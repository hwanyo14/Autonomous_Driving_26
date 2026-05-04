import torch
import torch.nn as nn


# class SoftVoxelizerOneAdd(nn.Module):
#     def __init__(self,
#                  point_cloud_range,
#                  voxel_size=0.2,
#                  occ_size=(512, 512, 40),
#                  as_prob=False,
#                  lambda_occ=1.0):
#         super().__init__()
#         self.point_cloud_range = point_cloud_range

#         if isinstance(voxel_size, (float, int)):
#             self.vx, self.vy, self.vz = float(voxel_size), float(voxel_size), float(voxel_size)
#         else:
#             self.vx, self.vy, self.vz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])

#         self.W, self.H, self.D = int(occ_size[0]), int(occ_size[1]), int(occ_size[2])
#         self.as_prob = as_prob
#         self.lambda_occ = float(lambda_occ)

#         # 8 corners offsets (dx, dy, dz)
#         # order: (0,0,0) (1,0,0) (0,1,0) (1,1,0) (0,0,1) (1,0,1) (0,1,1) (1,1,1)
#         self.register_buffer("dx8", torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long), persistent=False)
#         self.register_buffer("dy8", torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long), persistent=False)
#         self.register_buffer("dz8", torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long), persistent=False)

#     def forward(self, points_world: torch.Tensor) -> torch.Tensor:
#         """
#         points_world:
#           - [T, Nq, P, 3] 또는 [T, N, 3]
#         return:
#           occ: [T, 1, D, H, W]
#         """
#         assert points_world.dim() in (3, 4)

#         if points_world.dim() == 4:
#             T, Nq, P, _ = points_world.shape
#             pts = points_world.reshape(T, Nq * P, 3)
#         else:
#             T, N, _ = points_world.shape
#             pts = points_world

#         device = pts.device
#         dtype = pts.dtype

#         x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
#         x_min = torch.tensor(x_min, device=device, dtype=dtype)
#         y_min = torch.tensor(y_min, device=device, dtype=dtype)
#         z_min = torch.tensor(z_min, device=device, dtype=dtype)

#         vx = torch.tensor(self.vx, device=device, dtype=dtype)
#         vy = torch.tensor(self.vy, device=device, dtype=dtype)
#         vz = torch.tensor(self.vz, device=device, dtype=dtype)

#         occ = torch.zeros((T, self.D, self.H, self.W), device=device, dtype=dtype)
#         occ_flat = occ.view(T, -1)  # [T, D*H*W]

#         dx8 = self.dx8.to(device=device)
#         dy8 = self.dy8.to(device=device)
#         dz8 = self.dz8.to(device=device)

#         for t in range(T):
#             p = pts[t]  # [N, 3]
#             x = p[:, 0]
#             y = p[:, 1]
#             z = p[:, 2]

#             # world -> voxel continuous coords
#             u = (x - x_min) / vx  # x axis -> W
#             v = (y - y_min) / vy  # y axis -> H
#             w = (z - z_min) / vz  # z axis -> D

#             ix0 = torch.floor(u).to(torch.long)
#             iy0 = torch.floor(v).to(torch.long)
#             iz0 = torch.floor(w).to(torch.long)

#             fx = (u - ix0.to(dtype)).clamp(0.0, 1.0)
#             fy = (v - iy0.to(dtype)).clamp(0.0, 1.0)
#             fz = (w - iz0.to(dtype)).clamp(0.0, 1.0)

#             one_fx = 1.0 - fx
#             one_fy = 1.0 - fy
#             one_fz = 1.0 - fz

#             # [N, 8] indices
#             ix = ix0[:, None] + dx8[None, :]
#             iy = iy0[:, None] + dy8[None, :]
#             iz = iz0[:, None] + dz8[None, :]

#             # [N, 8] weights (trilinear)
#             wx = torch.where(dx8[None, :].bool(), fx[:, None], one_fx[:, None])
#             wy = torch.where(dy8[None, :].bool(), fy[:, None], one_fy[:, None])
#             wz = torch.where(dz8[None, :].bool(), fz[:, None], one_fz[:, None])
#             wgt = (wx * wy * wz).to(dtype)  # [N, 8]

#             # in-bounds mask
#             mask = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H) & (iz >= 0) & (iz < self.D)

#             # flatten valid
#             ixv = ix[mask]
#             iyv = iy[mask]
#             izv = iz[mask]
#             wv = wgt[mask]

#             # linear index in [0, D*H*W)
#             lin = (izv * (self.H * self.W) + iyv * self.W + ixv).to(torch.long)

#             # single index_add
#             occ_flat[t].index_add_(0, lin, wv)

#         occ = occ.unsqueeze(1)  # [T, 1, D, H, W]

#         if self.as_prob:
#             lam = torch.tensor(self.lambda_occ, device=device, dtype=dtype)
#             occ = 1.0 - torch.exp(-lam * occ)
        
#         return occ


class SoftVoxelizerOneAdd(nn.Module):
    def __init__(
        self,
        point_cloud_range,
        voxel_size=0.2,
        occ_size=(512, 512, 40),
        as_prob=False,
        lambda_occ=1.0,
        prob_eps=1e-6,
        # norm_mode="per_object",  # "none", "per_object", "per_frame"
        norm_mode="none",  # "none", "per_object", "per_frame"
        force_fp32=True,
        gaussian_truncate_sigma=3.0,
        gaussian_sigma_floor_vox=0.35,
    ):
        super().__init__()
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        if isinstance(voxel_size, (float, int)):
            vx, vy, vz = float(voxel_size), float(voxel_size), float(voxel_size)
        else:
            vx, vy, vz = float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2])

        self.W, self.H, self.D = int(occ_size[0]), int(occ_size[1]), int(occ_size[2])
        self.as_prob = as_prob
        self.lambda_occ = float(lambda_occ)
        self.prob_eps = float(prob_eps)
        self.norm_mode = norm_mode
        self.force_fp32 = force_fp32
        self.gaussian_truncate_sigma = float(gaussian_truncate_sigma)
        self.gaussian_sigma_floor_vox = float(gaussian_sigma_floor_vox)

        # buffers
        x_min, y_min, z_min, x_max, y_max, z_max = point_cloud_range
        self.register_buffer("pc_min", torch.tensor([x_min, y_min, z_min], dtype=torch.float32), persistent=False)
        self.register_buffer("vs", torch.tensor([vx, vy, vz], dtype=torch.float32), persistent=False)

        self.register_buffer("dx8", torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long), persistent=False)
        self.register_buffer("dy8", torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long), persistent=False)
        self.register_buffer("dz8", torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long), persistent=False)

        self.register_buffer("dx8b", self.dx8.bool(), persistent=False)
        self.register_buffer("dy8b", self.dy8.bool(), persistent=False)
        self.register_buffer("dz8b", self.dz8.bool(), persistent=False)

    def forward(
        self,
        points_world: torch.Tensor,
        weights: torch.Tensor = None,
        sigmas_world: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        points_world:
          - trilinear mode: [T, Nq, P, 3] 또는 [T, N, 3]
          - gaussian mode(sigmas_world not None): [T, Nq, 3]
        weights (optional):
          - trilinear mode: [T, Nq] 또는 [T, Nq, P] 또는 [T, N]
          - gaussian mode: [T, Nq]
        sigmas_world (optional):
          [T, Nq, 3] axis-wise sigma(m). 주어지면 Gaussian voxelization 수행.
        return:
          occ: [T, 1, D, H, W]
        """
        if sigmas_world is not None:
            return self._forward_gaussian(points_world, sigmas_world, weights=weights)

        assert points_world.dim() in (3, 4)

        if points_world.dim() == 4:
            T, Nq, P, _ = points_world.shape
            pts = points_world.reshape(T, Nq * P, 3)
            if weights is not None:
                if weights.dim() == 2:
                    # [T, Nq] -> [T, Nq, P] -> [T, Nq*P]
                    w_in = weights[:, :, None].expand(T, Nq, P).reshape(T, Nq * P)
                elif weights.dim() == 3:
                    # [T, Nq, P] -> [T, Nq*P]
                    w_in = weights.reshape(T, Nq * P)
                else:
                    raise ValueError("weights shape mismatch for points_world dim 4")
            else:
                w_in = None
        else:
            T, N, _ = points_world.shape
            pts = points_world
            if weights is not None:
                assert weights.dim() == 2 and weights.shape == (T, N)
                w_in = weights
            else:
                w_in = None

        device = pts.device
        out_dtype = pts.dtype

        # fp32로 누적하는 것이 안정적이다
        if self.force_fp32:
            pts_calc = pts.float()
            w_in_calc = w_in.float() if w_in is not None else None
        else:
            pts_calc = pts
            w_in_calc = w_in

        pc_min = self.pc_min.to(device=device, dtype=pts_calc.dtype)
        vs = self.vs.to(device=device, dtype=pts_calc.dtype)

        occ = torch.zeros((T, self.D, self.H, self.W), device=device, dtype=pts_calc.dtype)
        occ_flat = occ.view(T, -1)

        dx8 = self.dx8.to(device=device)
        dy8 = self.dy8.to(device=device)
        dz8 = self.dz8.to(device=device)
        dx8b = self.dx8b.to(device=device)
        dy8b = self.dy8b.to(device=device)
        dz8b = self.dz8b.to(device=device)

        for t in range(T):
            p = pts_calc[t]  # [N, 3]
            x = p[:, 0]
            y = p[:, 1]
            z = p[:, 2]

            u = (x - pc_min[0]) / vs[0]
            v = (y - pc_min[1]) / vs[1]
            w = (z - pc_min[2]) / vs[2]

            # floor는 비미분이지만, fx = u - floor(u)는 셀 내부에서 piecewise로 gradient가 흐른다
            ix0_f = torch.floor(u)
            iy0_f = torch.floor(v)
            iz0_f = torch.floor(w)

            ix0 = ix0_f.to(torch.long)
            iy0 = iy0_f.to(torch.long)
            iz0 = iz0_f.to(torch.long)

            fx = (u - ix0_f).clamp(0.0, 1.0)
            fy = (v - iy0_f).clamp(0.0, 1.0)
            fz = (w - iz0_f).clamp(0.0, 1.0)

            one_fx = 1.0 - fx
            one_fy = 1.0 - fy
            one_fz = 1.0 - fz

            ix = ix0[:, None] + dx8[None, :]
            iy = iy0[:, None] + dy8[None, :]
            iz = iz0[:, None] + dz8[None, :]

            wx = torch.where(dx8b[None, :], fx[:, None], one_fx[:, None])
            wy = torch.where(dy8b[None, :], fy[:, None], one_fy[:, None])
            wz = torch.where(dz8b[None, :], fz[:, None], one_fz[:, None])
            wgt = wx * wy * wz  # [N, 8]

            # 정규화 옵션
            if self.norm_mode == "per_object" and points_world.dim() == 4:
                # 객체당 포인트 수 P에 대해 불변하게 만든다
                wgt = wgt / float(P)
            elif self.norm_mode == "per_frame":
                wgt = wgt / float(p.shape[0])

            # 추가 가중치 적용
            if w_in_calc is not None:
                wgt = wgt * w_in_calc[t][:, None]

            mask = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H) & (iz >= 0) & (iz < self.D)

            ixv = ix[mask]
            iyv = iy[mask]
            izv = iz[mask]
            wv = wgt[mask]

            lin = (izv * (self.H * self.W) + iyv * self.W + ixv).to(torch.long)
            occ_flat[t].index_add_(0, lin, wv)

            # occ_list = []
            # V = self.D * self.H * self.W
            # for t in range(T):
            #     base = torch.zeros((V,), device=device, dtype=out_dtype)  # requires_grad=False여도 괜찮음
            #     base = base.index_add(0, lin, wv)  # out-of-place, wv가 grad를 가지면 base는 grad_fn이 생김
            #     occ_list.append(base)

            # occ = torch.stack(occ_list, dim=0).view(T, self.D, self.H, self.W)

        occ = occ.unsqueeze(1)  # [T, 1, D, H, W]
        return self._convert_occ_output(occ, out_dtype)

    def _convert_occ_output(self, occ: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        if self.as_prob:
            # 확률 변환은 fp32에서 안정적으로
            occ32 = occ.float()
            x = self.lambda_occ * occ32
            p32 = -torch.expm1(-x)  # 1 - exp(-x)
            p32 = p32.clamp_max(1.0 - self.prob_eps)
            return p32
        return occ.to(out_dtype)

    def _forward_gaussian(
        self,
        centers_world: torch.Tensor,
        sigmas_world: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if centers_world.dim() != 3:
            raise ValueError(f"gaussian centers must be [T,Nq,3], got {tuple(centers_world.shape)}")
        if sigmas_world.shape != centers_world.shape:
            raise ValueError(
                f"sigmas_world shape mismatch: {tuple(sigmas_world.shape)} vs {tuple(centers_world.shape)}"
            )

        T, Q, _ = centers_world.shape
        if weights is not None and (weights.dim() != 2 or weights.shape != (T, Q)):
            raise ValueError(
                f"gaussian mode weights must be [T,Nq], got {tuple(weights.shape)} for T={T},Q={Q}"
            )

        device = centers_world.device
        out_dtype = centers_world.dtype

        if self.force_fp32:
            centers = centers_world.float()
            sigmas = sigmas_world.float()
            w_in = weights.float() if weights is not None else None
        else:
            centers = centers_world
            sigmas = sigmas_world
            w_in = weights

        pc_min = self.pc_min.to(device=device, dtype=centers.dtype)
        vs = self.vs.to(device=device, dtype=centers.dtype)

        occ = torch.zeros((T, self.D, self.H, self.W), device=device, dtype=centers.dtype)
        occ_flat = occ.view(T, -1)

        cx = (centers[..., 0] - pc_min[0]) / vs[0] - 0.5
        cy = (centers[..., 1] - pc_min[1]) / vs[1] - 0.5
        cz = (centers[..., 2] - pc_min[2]) / vs[2] - 0.5

        sx = (sigmas[..., 0] / vs[0]).clamp(min=self.gaussian_sigma_floor_vox)
        sy = (sigmas[..., 1] / vs[1]).clamp(min=self.gaussian_sigma_floor_vox)
        sz = (sigmas[..., 2] / vs[2]).clamp(min=self.gaussian_sigma_floor_vox)

        trunc = max(1.0, float(self.gaussian_truncate_sigma))

        for t in range(T):
            for q in range(Q):
                cxi, cyi, czi = cx[t, q], cy[t, q], cz[t, q]
                sxi, syi, szi = sx[t, q], sy[t, q], sz[t, q]

                rx = max(1, int(torch.ceil(trunc * sxi).item()))
                ry = max(1, int(torch.ceil(trunc * syi).item()))
                rz = max(1, int(torch.ceil(trunc * szi).item()))

                x0 = max(0, int(torch.floor(cxi).item()) - rx)
                x1 = min(self.W - 1, int(torch.floor(cxi).item()) + rx)
                y0 = max(0, int(torch.floor(cyi).item()) - ry)
                y1 = min(self.H - 1, int(torch.floor(cyi).item()) + ry)
                z0 = max(0, int(torch.floor(czi).item()) - rz)
                z1 = min(self.D - 1, int(torch.floor(czi).item()) + rz)

                if (x1 < x0) or (y1 < y0) or (z1 < z0):
                    continue

                xs = torch.arange(x0, x1 + 1, device=device, dtype=centers.dtype)
                ys = torch.arange(y0, y1 + 1, device=device, dtype=centers.dtype)
                zs = torch.arange(z0, z1 + 1, device=device, dtype=centers.dtype)

                gx = torch.exp(-0.5 * ((xs - cxi) / sxi).pow(2))
                gy = torch.exp(-0.5 * ((ys - cyi) / syi).pow(2))
                gz = torch.exp(-0.5 * ((zs - czi) / szi).pow(2))
                wgt = gx[:, None, None] * gy[None, :, None] * gz[None, None, :]  # [X,Y,Z]

                if self.norm_mode == "per_frame":
                    wgt = wgt / float(max(1, Q))

                if w_in is not None:
                    wgt = wgt * w_in[t, q]

                xx, yy, zz = torch.meshgrid(
                    xs.to(torch.long), ys.to(torch.long), zs.to(torch.long), indexing="ij"
                )
                lin = (zz * (self.H * self.W) + yy * self.W + xx).reshape(-1).to(torch.long)
                occ_flat[t].index_add_(0, lin, wgt.reshape(-1))

        occ = occ.unsqueeze(1)  # [T,1,D,H,W]
        return self._convert_occ_output(occ, out_dtype)

    def forward_gaussian_mixture_grouped(
        self,
        mixture_centers_world_tkg3: torch.Tensor,
        mixture_sigmas_world_tkg3: torch.Tensor,
        mixture_weights_tkg: torch.Tensor,
        mixture_yaw_tkg: torch.Tensor,
        pair_weights_tk: torch.Tensor = None,
        pair_chunk_size: int = 8,
    ) -> torch.Tensor:
        """
        Grouped rotated Gaussian-mixture voxelization for matched pairs.
        Args:
            mixture_centers_world_tkg3: [T,K,G,3]
            mixture_sigmas_world_tkg3: [T,K,G,3]
            mixture_weights_tkg: [T,K,G] (typically softmax-normalized over G)
            mixture_yaw_tkg: [T,K,G] (z-axis yaw, radians)
            pair_weights_tk: optional [T,K] scalar weight per pair
            pair_chunk_size: chunk size over K for memory control
        Returns:
            occ_tk1zyx: [T,K,1,D,H,W], with p = 1 - exp(-sum_g w_g * G_g(x))
        """
        if mixture_centers_world_tkg3.dim() != 4:
            raise ValueError(
                f"mixture_centers_world_tkg3 must be [T,K,G,3], got {tuple(mixture_centers_world_tkg3.shape)}"
            )
        if mixture_sigmas_world_tkg3.shape != mixture_centers_world_tkg3.shape:
            raise ValueError(
                f"mixture_sigmas_world_tkg3 shape mismatch: {tuple(mixture_sigmas_world_tkg3.shape)} "
                f"vs {tuple(mixture_centers_world_tkg3.shape)}"
            )
        if mixture_weights_tkg.dim() != 3:
            raise ValueError(f"mixture_weights_tkg must be [T,K,G], got {tuple(mixture_weights_tkg.shape)}")
        if mixture_yaw_tkg.dim() != 3:
            raise ValueError(f"mixture_yaw_tkg must be [T,K,G], got {tuple(mixture_yaw_tkg.shape)}")
        if mixture_weights_tkg.shape != mixture_yaw_tkg.shape:
            raise ValueError(
                f"mixture_weights_tkg/mixture_yaw_tkg shape mismatch: "
                f"{tuple(mixture_weights_tkg.shape)} vs {tuple(mixture_yaw_tkg.shape)}"
            )
        if tuple(mixture_weights_tkg.shape[:3]) != tuple(mixture_centers_world_tkg3.shape[:3]):
            raise ValueError(
                "mixture_weights_tkg must match centers [T,K,G], got "
                f"{tuple(mixture_weights_tkg.shape)} vs {tuple(mixture_centers_world_tkg3.shape[:3])}"
            )

        T, K, G, _ = [int(v) for v in mixture_centers_world_tkg3.shape]
        if pair_weights_tk is not None:
            if pair_weights_tk.dim() != 2 or tuple(pair_weights_tk.shape) != (T, K):
                raise ValueError(f"pair_weights_tk must be [T,K], got {tuple(pair_weights_tk.shape)}")
        if T <= 0 or K <= 0 or G <= 0:
            return mixture_centers_world_tkg3.new_zeros((T, K, 1, self.D, self.H, self.W))

        device = mixture_centers_world_tkg3.device
        out_dtype = mixture_centers_world_tkg3.dtype
        if self.force_fp32:
            centers = mixture_centers_world_tkg3.float()
            sigmas = mixture_sigmas_world_tkg3.float()
            comp_weights = mixture_weights_tkg.float()
            yaws = mixture_yaw_tkg.float()
            pair_weights = pair_weights_tk.float() if pair_weights_tk is not None else None
        else:
            centers = mixture_centers_world_tkg3
            sigmas = mixture_sigmas_world_tkg3
            comp_weights = mixture_weights_tkg
            yaws = mixture_yaw_tkg
            pair_weights = pair_weights_tk

        pc_min = self.pc_min.to(device=device, dtype=centers.dtype)
        vs = self.vs.to(device=device, dtype=centers.dtype)
        trunc = max(1.0, float(self.gaussian_truncate_sigma))
        trunc2 = trunc * trunc
        chunk = max(1, int(pair_chunk_size))

        cx = (centers[..., 0] - pc_min[0]) / vs[0] - 0.5
        cy = (centers[..., 1] - pc_min[1]) / vs[1] - 0.5
        cz = (centers[..., 2] - pc_min[2]) / vs[2] - 0.5
        sx = (sigmas[..., 0] / vs[0]).clamp(min=self.gaussian_sigma_floor_vox)
        sy = (sigmas[..., 1] / vs[1]).clamp(min=self.gaussian_sigma_floor_vox)
        sz = (sigmas[..., 2] / vs[2]).clamp(min=self.gaussian_sigma_floor_vox)
        wt = comp_weights.clamp(min=0.0)
        yaw = yaws

        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        sx_rot = torch.sqrt((cos_y.pow(2) * sx.pow(2)) + (sin_y.pow(2) * sy.pow(2)))
        sy_rot = torch.sqrt((sin_y.pow(2) * sx.pow(2)) + (cos_y.pow(2) * sy.pow(2)))
        rx = torch.ceil(trunc * sx_rot).to(torch.long).clamp(min=1)
        ry = torch.ceil(trunc * sy_rot).to(torch.long).clamp(min=1)
        rz = torch.ceil(trunc * sz).to(torch.long).clamp(min=1)

        occ_tkzyx = centers.new_zeros((T, K, self.D, self.H, self.W))
        for t in range(T):
            for k0 in range(0, K, chunk):
                k1 = min(K, k0 + chunk)
                kc = k1 - k0
                if kc <= 0:
                    continue

                cxc = cx[t, k0:k1]   # [Kc,G]
                cyc = cy[t, k0:k1]
                czc = cz[t, k0:k1]
                sxc = sx[t, k0:k1]
                syc = sy[t, k0:k1]
                szc = sz[t, k0:k1]
                wtc = wt[t, k0:k1]
                cosc = cos_y[t, k0:k1]
                sinc = sin_y[t, k0:k1]
                rxc = rx[t, k0:k1]
                ryc = ry[t, k0:k1]
                rzc = rz[t, k0:k1]

                floor_cx = torch.floor(cxc)
                floor_cy = torch.floor(cyc)
                floor_cz = torch.floor(czc)

                x0_pair = (floor_cx.to(torch.long) - rxc).amin(dim=1).clamp(min=0, max=self.W - 1)
                x1_pair = (floor_cx.to(torch.long) + rxc).amax(dim=1).clamp(min=0, max=self.W - 1)
                y0_pair = (floor_cy.to(torch.long) - ryc).amin(dim=1).clamp(min=0, max=self.H - 1)
                y1_pair = (floor_cy.to(torch.long) + ryc).amax(dim=1).clamp(min=0, max=self.H - 1)
                z0_pair = (floor_cz.to(torch.long) - rzc).amin(dim=1).clamp(min=0, max=self.D - 1)
                z1_pair = (floor_cz.to(torch.long) + rzc).amax(dim=1).clamp(min=0, max=self.D - 1)

                valid_pair = (x1_pair >= x0_pair) & (y1_pair >= y0_pair) & (z1_pair >= z0_pair)
                if not bool(valid_pair.any().item()):
                    continue

                x0 = int(x0_pair.min().item())
                x1 = int(x1_pair.max().item())
                y0 = int(y0_pair.min().item())
                y1 = int(y1_pair.max().item())
                z0 = int(z0_pair.min().item())
                z1 = int(z1_pair.max().item())
                if (x1 < x0) or (y1 < y0) or (z1 < z0):
                    continue

                xs = torch.arange(x0, x1 + 1, device=device, dtype=centers.dtype)
                ys = torch.arange(y0, y1 + 1, device=device, dtype=centers.dtype)
                zs = torch.arange(z0, z1 + 1, device=device, dtype=centers.dtype)
                xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")

                dx = xx[None, None] - cxc[..., None, None, None]
                dy = yy[None, None] - cyc[..., None, None, None]
                dz = zz[None, None] - czc[..., None, None, None]

                xr = cosc[..., None, None, None] * dx + sinc[..., None, None, None] * dy
                yr = -sinc[..., None, None, None] * dx + cosc[..., None, None, None] * dy
                md2 = (
                    (xr / sxc[..., None, None, None]).pow(2)
                    + (yr / syc[..., None, None, None]).pow(2)
                    + (dz / szc[..., None, None, None]).pow(2)
                )
                gauss = torch.exp(-0.5 * md2)
                gauss = gauss * (md2 <= trunc2).to(gauss.dtype)
                lam_pair = (wtc[..., None, None, None] * gauss).sum(dim=1)
                p_pair = (-torch.expm1(-lam_pair)).clamp(0.0, 1.0)

                in_pair = (
                    (xx[None] >= x0_pair[:, None, None, None].to(xx.dtype))
                    & (xx[None] <= x1_pair[:, None, None, None].to(xx.dtype))
                    & (yy[None] >= y0_pair[:, None, None, None].to(yy.dtype))
                    & (yy[None] <= y1_pair[:, None, None, None].to(yy.dtype))
                    & (zz[None] >= z0_pair[:, None, None, None].to(zz.dtype))
                    & (zz[None] <= z1_pair[:, None, None, None].to(zz.dtype))
                )
                p_pair = p_pair * in_pair.to(p_pair.dtype)
                if pair_weights is not None:
                    p_pair = p_pair * pair_weights[t, k0:k1, None, None, None].clamp(min=0.0)
                p_pair = p_pair * valid_pair[:, None, None, None].to(p_pair.dtype)
                occ_tkzyx[t, k0:k1, z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = p_pair.permute(0, 3, 2, 1)

        return occ_tkzyx.unsqueeze(2).to(out_dtype)


if __name__ == "__main__":
    voxelizer = SoftVoxelizerOneAdd(
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size=0.2,
        occ_size=(512, 512, 40),
        as_prob=True,
        lambda_occ=1.0
    )

    points_world = torch.randn(6, 100, 256, 3)
    occ = voxelizer(points_world)
    print(occ.shape)  # should be [2, 1, 40, 512, 512]
