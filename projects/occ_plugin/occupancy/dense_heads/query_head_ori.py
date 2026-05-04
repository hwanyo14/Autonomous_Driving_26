import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

def sigmoid_to_world_from_range(logits: torch.Tensor,
                                point_cloud_range,
                                spatial_extent3d) -> torch.Tensor:
    """
    logits: (..., 3)  (unconstrained)
    point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    spatial_extent3d:  (x_range, y_range, z_range) = (x_max-x_min, y_max-y_min, z_max-z_min)

    return: (..., 3) world coords in the same unit as point_cloud_range (meters)
    """
    # make tensors on correct device/dtype
    pc_min = torch.tensor(point_cloud_range[:3], device=logits.device, dtype=logits.dtype)          # [3]
    extent = torch.tensor(spatial_extent3d, device=logits.device, dtype=logits.dtype)              # [3]

    return pc_min + torch.sigmoid(logits) * extent


def symmetric_offset_from_extent(offset_logits: torch.Tensor,
                                 spatial_extent3d,
                                 radius_ratio=(0.05, 0.05, 0.25)) -> torch.Tensor:
    """
    offset_logits: (..., 3) unconstrained
    spatial_extent3d: (x_range, y_range, z_range) e.g. (102.4, 102.4, 8.0)

    radius_ratio: extent 대비 최대 오프셋 비율
      - x,y는 0.05면 최대 약 5.12m (102.4 * 0.05)
      - z는 0.25면 최대 2.0m (8.0 * 0.25)

    return: (..., 3) offsets in meters, range approx [-radius, +radius]
    """
    extent = torch.tensor(spatial_extent3d, device=offset_logits.device, dtype=offset_logits.dtype)  # [3]
    ratio = torch.tensor(radius_ratio, device=offset_logits.device, dtype=offset_logits.dtype)       # [3]

    # radius in meters for each axis
    radius = extent * ratio  # [3]

    # map logits -> [-1, 1]
    unit = 2.0 * torch.sigmoid(offset_logits) - 1.0
    return unit * radius

def build_points_world(center_logits: torch.Tensor,
                       offset_logits: torch.Tensor,
                       point_cloud_range,
                       spatial_extent3d,
                       radius_ratio=(0.05, 0.05, 0.25),
                       clamp_to_range: bool = True) -> torch.Tensor:
    """
    center_logits: [F, Nq, 3]
    offset_logits: [F, Nq, P, 3]
    returns points_world: [F, Nq, P, 3]
    """
    centers = sigmoid_to_world_from_range(center_logits, point_cloud_range, spatial_extent3d)        # [F, Nq, 3]
    offsets = symmetric_offset_from_extent(offset_logits, spatial_extent3d, radius_ratio)            # [F, Nq, P, 3]
    points = centers[:, :, None, :] + offsets                                                        # [F, Nq, P, 3]

    if clamp_to_range:
        pc_min = torch.tensor(point_cloud_range[:3], device=points.device, dtype=points.dtype)       # [3]
        pc_max = torch.tensor(point_cloud_range[3:], device=points.device, dtype=points.dtype)       # [3]
        points = torch.max(torch.min(points, pc_max), pc_min)

    return points


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0, use_ln=True):
        super().__init__()
        assert num_layers >= 1
        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(in_dim) if use_ln else nn.Identity()

    def forward(self, x):
        x = self.ln(x)
        return self.net(x)


class CenterHead(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 out_dim=3):
        super(CenterHead, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim

        self.mlp = MLP(in_dim=self.embed_dim,
                       hidden_dim=self.embed_dim,
                       out_dim=self.out_dim,
                       num_layers=2,
                       dropout=0.0,
                       use_ln=True)

    def forward(self, query_inst):
        center_logits = self.mlp(query_inst)  # [T, num_queries, out_dim]
        return center_logits
    

class OffsetHead(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_pcd=256,
                 out_dim=3):
        super(OffsetHead, self).__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_pcd = num_pcd

        self.mlp = MLP(in_dim=self.embed_dim,
                       hidden_dim=self.embed_dim,
                       out_dim=self.out_dim * self.num_pcd,
                       num_layers=2,
                       dropout=0.0,
                       use_ln=True)

    def forward(self, query_inst):
        offset_logits = self.mlp(query_inst)  # [T, num_queries, out_dim * num_pcd]
        return offset_logits
    

class QueryHead(nn.Module):
    def __init__(self,
                 embed_dim=128,
                 num_past_frames=3,
                 num_future_frames=6,
                 center_out_dim=3,
                 offset_out_dim=3,
                 num_pcd=256,
                 point_cloud_range=None,
                 spatial_extent3d=None):
        super(QueryHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.point_cloud_range = point_cloud_range
        self.spatial_extent3d = spatial_extent3d
        self.num_pcd = num_pcd
        self.iter = 0
        self.debug_vis_every = 8
        self.debug_vis_dir = "./work_dirs/query_debug_vis"

        self.query_projection = nn.Sequential(
            nn.LayerNorm(self.embed_dim * self.num_past_frames),
            nn.Linear(self.embed_dim * self.num_past_frames, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.embed_dim * self.num_future_frames)
        )

        self.center_head = CenterHead(embed_dim=self.embed_dim,
                                      out_dim=center_out_dim)
        self.offset_head = OffsetHead(embed_dim=self.embed_dim,
                                      out_dim=offset_out_dim,
                                      num_pcd=num_pcd)

    def forward(self, query_inst):
        '''
            query_inst: [T, num_queries, embed_dim]
        '''
        query_inst = query_inst.permute(1, 0, 2)  # [num_queries, T, embed_dim]
        query_inst = query_inst.reshape(query_inst.shape[0], -1)  # [num_queries, T * embed_dim]
        query_inst = self.query_projection(query_inst)  # [num_queries, T_future * embed_dim]
        query_inst = query_inst.reshape(query_inst.shape[0], self.num_future_frames, self.embed_dim)  # [num_queries, T_future, embed_dim]
        query_inst = query_inst.permute(1, 0, 2)  # [T_future, num_queries, embed_dim]

        center_logits = self.center_head(query_inst)  # [T, num_queries, center_out_dim]


        offset_logits = self.offset_head(query_inst)  # [T, num_queries, offset_out_dim * num_pcd]
        offset_logits = offset_logits.view(query_inst.shape[0], query_inst.shape[1], -1, 3)  # [T, num_queries, num_pcd, 3]

        points_world = build_points_world(center_logits, offset_logits,
                                          point_cloud_range=self.point_cloud_range,
                                          spatial_extent3d=self.spatial_extent3d,
                                          radius_ratio=(0.05, 0.05, 0.25),
                                          clamp_to_range=True)

        return points_world, center_logits, offset_logits

    def _is_main_process(self) -> bool:
        if not dist.is_available():
            return True
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    @torch.no_grad()
    def _maybe_save_points_gt_vis(
        self,
        points_world: torch.Tensor,
        gt_occ_gmo: torch.Tensor,
        step: int,
        max_frames: int = 6,
        voxel_center_offset: float = 0.5,
    ) -> None:
        vis_every = int(getattr(self, "debug_vis_every", 0))
        if vis_every <= 0:
            return
        if (step % vis_every) != 0:
            return
        if not self._is_main_process():
            return
        if points_world is None:
            return
        if (self.point_cloud_range is None) or (self.spatial_extent3d is None):
            return

        try:
            import os
            from PIL import Image, ImageDraw
        except Exception:
            return

        if points_world.dim() == 4:
            pts = points_world
        elif points_world.dim() == 5:
            # fallback for batched layout
            pts = points_world[0]
        else:
            return

        if gt_occ_gmo.dim() != 4:
            return

        # gt_occ_gmo: [T, Z, H, W]
        T_gt, Z, H, W = gt_occ_gmo.shape
        T_pred = pts.shape[0]
        T = int(min(T_gt, T_pred, max_frames))
        if T <= 0:
            return

        pts = pts[:T].to(torch.float32)
        gt = gt_occ_gmo[:T]

        # world -> voxel-center index with the same convention as DT sampling.
        pc_min = pts.new_tensor(self.point_cloud_range[:3])  # [x_min,y_min,z_min]
        extent = pts.new_tensor(self.spatial_extent3d)       # [x_range,y_range,z_range]
        grid_size = pts.new_tensor([float(W), float(H), float(Z)])
        voxel_size = extent / grid_size

        off = float(voxel_center_offset)
        ix = (pts[..., 0] - pc_min[0]) / voxel_size[0] - off
        iy = (pts[..., 1] - pc_min[1]) / voxel_size[1] - off
        iz = (pts[..., 2] - pc_min[2]) / voxel_size[2] - off
        valid = (ix >= 0.0) & (ix <= float(W - 1)) & (iy >= 0.0) & (iy <= float(H - 1)) & (iz >= 0.0) & (iz <= float(Z - 1))

        ix_i = torch.round(ix).to(torch.long).cpu().numpy()
        iy_i = torch.round(iy).to(torch.long).cpu().numpy()
        valid_np = valid.cpu().numpy()

        gt_np = gt.cpu().numpy()
        bev_list = []
        overlay_list = []
        stats = []

        for t in range(T):
            # [Z, H, W] -> BEV [H, W]
            bev = (gt_np[t] == 1).any(axis=0).astype(np.bool_)

            gt_rgb = np.zeros((H, W, 3), dtype=np.uint8)
            gt_rgb[bev] = np.array([240, 240, 240], dtype=np.uint8)

            ov = np.zeros((H, W, 3), dtype=np.uint8)
            ov[bev] = np.array([40, 180, 40], dtype=np.uint8)

            xt = ix_i[t].reshape(-1)
            yt = iy_i[t].reshape(-1)
            vt = valid_np[t].reshape(-1)
            if vt.any():
                xv = xt[vt]
                yv = yt[vt]
                # draw predicted points in red (3x3)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        xx = np.clip(xv + dx, 0, W - 1)
                        yy = np.clip(yv + dy, 0, H - 1)
                        ov[yy, xx] = np.array([255, 70, 70], dtype=np.uint8)
                vratio = float(vt.mean())
            else:
                vratio = 0.0

            bev_list.append(gt_rgb)
            overlay_list.append(ov)
            stats.append(vratio)

        gap = 4
        text_h = 18
        row_h = H
        canvas_w = T * W + (T - 1) * gap
        canvas_h = text_h + row_h + gap + row_h
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for t in range(T):
            x0 = t * (W + gap)
            canvas[text_h:text_h + row_h, x0:x0 + W] = bev_list[t]
            y1 = text_h + row_h + gap
            canvas[y1:y1 + row_h, x0:x0 + W] = overlay_list[t]

        img = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.text((2, 1), "row1: gt_occ GMO BEV  row2: overlay (green=gt, red=pred points)", fill=(255, 255, 255))
        for t in range(T):
            x0 = t * (W + gap)
            draw.text((x0 + 2, text_h + 2), f"t={t}", fill=(255, 255, 255))
            y1 = text_h + row_h + gap
            draw.text((x0 + 2, y1 + 2), f"valid={stats[t]:.3f}", fill=(255, 255, 255))

        vis_dir = str(getattr(self, "debug_vis_dir", "./work_dirs/query_debug_vis"))
        os.makedirs(vis_dir, exist_ok=True)
        out_path = os.path.join(vis_dir, f"iter_{int(step):06d}.png")
        img.save(out_path)

    @torch.no_grad()
    def _maybe_save_prob_grid_vis(
        self,
        pred_occ_prob: torch.Tensor,
        points_world: torch.Tensor,
        gt_occ_gmo: torch.Tensor,
        step: int,
        max_frames: int = 6,
        voxel_center_offset: float = 0.5,
        prob_threshold: float = 0.5,
    ) -> None:
        """
        확률 기반 BEV 시각화.
        - row1: GT GMO BEV
        - row2: Pred occupancy prob BEV (z축 max-pooling)
        - row3: Overlay (green=gt, yellow=pred>=thr, red=query points)
        """
        vis_every = int(getattr(self, "debug_vis_every", 0))
        if vis_every <= 0:
            return
        if (step % vis_every) != 0:
            return
        if not self._is_main_process():
            return
        if pred_occ_prob is None or points_world is None:
            return
        if (self.point_cloud_range is None) or (self.spatial_extent3d is None):
            return

        try:
            import os
            from PIL import Image, ImageDraw
        except Exception:
            return

        if points_world.dim() == 4:
            pts = points_world
        elif points_world.dim() == 5:
            pts = points_world[0]
        else:
            return

        if pred_occ_prob.dim() == 5 and pred_occ_prob.shape[1] == 1:
            pred = pred_occ_prob[:, 0]
        elif pred_occ_prob.dim() == 4:
            pred = pred_occ_prob
        else:
            return

        if gt_occ_gmo.dim() != 4:
            return

        # gt_occ_gmo follows [T, Z, X, Y] in this branch.
        T_gt, Z, X, Y = gt_occ_gmo.shape
        T_pred = pts.shape[0]
        T_prob = pred.shape[0]
        if pred.shape[1] != Z:
            return

        # pred can be [T,Z,Y,X] (voxelizer) or [T,Z,X,Y] depending source.
        pred_is_zyx = (pred.shape[2] == Y and pred.shape[3] == X)
        pred_is_zxy = (pred.shape[2] == X and pred.shape[3] == Y)
        if not (pred_is_zyx or pred_is_zxy):
            return

        T = int(min(T_gt, T_pred, T_prob, max_frames))
        if T <= 0:
            return

        pts = pts[:T].to(torch.float32)
        gt = gt_occ_gmo[:T]
        pred = pred[:T].to(torch.float32).clamp(0.0, 1.0)

        pc_min = pts.new_tensor(self.point_cloud_range[:3])
        extent = pts.new_tensor(self.spatial_extent3d)
        grid_size = pts.new_tensor([float(X), float(Y), float(Z)])
        voxel_size = extent / grid_size

        off = float(voxel_center_offset)
        ix = (pts[..., 0] - pc_min[0]) / voxel_size[0] - off
        iy = (pts[..., 1] - pc_min[1]) / voxel_size[1] - off
        iz = (pts[..., 2] - pc_min[2]) / voxel_size[2] - off
        valid = (ix >= 0.0) & (ix <= float(X - 1)) & (iy >= 0.0) & (iy <= float(Y - 1)) & (iz >= 0.0) & (iz <= float(Z - 1))

        ix_i = torch.round(ix).to(torch.long).cpu().numpy()
        iy_i = torch.round(iy).to(torch.long).cpu().numpy()
        valid_np = valid.cpu().numpy()

        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()

        row_gt = []
        row_prob = []
        row_overlay = []
        stats_valid = []
        stats_prob = []
        stats_bin = []

        thr = float(prob_threshold)
        for t in range(T):
            # gt: [Z, X, Y] -> [Y, X] (rollback: keep old gt/point orientation)
            gt_bev = (gt_np[t] == 1).any(axis=0).T.astype(np.bool_)
            pred_xy = pred_np[t].max(axis=0).astype(np.float32)
            # fix only yellow(pred): align to [Y, X] regardless of pred tensor layout.
            pred_bev = pred_xy if pred_is_zyx else pred_xy.T
            pred_bin = (pred_bev >= thr)

            gt_rgb = np.zeros((Y, X, 3), dtype=np.uint8)
            gt_rgb[gt_bev] = np.array([240, 240, 240], dtype=np.uint8)

            p8 = np.clip(pred_bev * 255.0, 0.0, 255.0).astype(np.uint8)
            prob_rgb = np.zeros((Y, X, 3), dtype=np.uint8)
            prob_rgb[..., 0] = p8
            prob_rgb[..., 1] = (p8.astype(np.float32) * 0.6).astype(np.uint8)
            prob_rgb[..., 2] = (p8.astype(np.float32) * 0.15).astype(np.uint8)
            # row2에도 GT를 블렌딩해 정합을 바로 확인할 수 있게 한다.
            prob_ov = prob_rgb.copy()
            if gt_bev.any():
                gt_color = np.array([40, 180, 40], dtype=np.float32)
                blended = 0.6 * prob_ov[gt_bev].astype(np.float32) + 0.4 * gt_color
                prob_ov[gt_bev] = np.clip(blended, 0.0, 255.0).astype(np.uint8)

            ov = np.zeros((Y, X, 3), dtype=np.uint8)
            ov[gt_bev] = np.array([40, 180, 40], dtype=np.uint8)
            ov[pred_bin] = np.array([255, 215, 40], dtype=np.uint8)

            xt = ix_i[t].reshape(-1)
            yt = iy_i[t].reshape(-1)
            vt = valid_np[t].reshape(-1)
            if vt.any():
                xv = xt[vt]
                yv = yt[vt]
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        xx = np.clip(xv + dx, 0, X - 1)
                        yy = np.clip(yv + dy, 0, Y - 1)
                        ov[yy, xx] = np.array([255, 70, 70], dtype=np.uint8)
                vratio = float(vt.mean())
            else:
                vratio = 0.0

            row_gt.append(gt_rgb)
            row_prob.append(prob_ov)
            row_overlay.append(ov)
            stats_valid.append(vratio)
            stats_prob.append(float(pred_bev.mean()))
            stats_bin.append(float(pred_bin.mean()))

        gap = 4
        text_h = 18
        row_h = Y
        canvas_w = T * X + (T - 1) * gap
        canvas_h = text_h + row_h + gap + row_h + gap + row_h
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for t in range(T):
            x0 = t * (X + gap)
            y0 = text_h
            y1 = y0 + row_h + gap
            y2 = y1 + row_h + gap
            canvas[y0:y0 + row_h, x0:x0 + X] = row_gt[t]
            canvas[y1:y1 + row_h, x0:x0 + X] = row_prob[t]
            canvas[y2:y2 + row_h, x0:x0 + X] = row_overlay[t]

        img = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.text((2, 1), "row1: gt_occ GMO BEV  row2: pred_prob BEV + GT overlay  row3: overlay(green=gt,yellow=pred>=thr,red=points)", fill=(255, 255, 255))

        for t in range(T):
            x0 = t * (X + gap)
            y0 = text_h
            y1 = y0 + row_h + gap
            y2 = y1 + row_h + gap
            draw.text((x0 + 2, y0 + 2), f"t={t}", fill=(255, 255, 255))
            draw.text((x0 + 2, y1 + 2), f"meanP={stats_prob[t]:.3f}", fill=(255, 255, 255))
            draw.text((x0 + 2, y2 + 2), f"bin={stats_bin[t]:.3f} valid={stats_valid[t]:.3f}", fill=(255, 255, 255))

        vis_dir = str(getattr(self, "debug_vis_dir", "./work_dirs/query_debug_vis"))
        os.makedirs(vis_dir, exist_ok=True)
        out_path = os.path.join(vis_dir, f"iter_{int(step):06d}_prob.png")
        img.save(out_path)
    
    def gt_occ_to_gmo_binary(self, gt_occ: torch.Tensor,
                            gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
                            ignore_index: int = 255) -> torch.Tensor:
        """
        gt_occ: (..., ) 정수 라벨 텐서
        예: (B, T, Z, X, Y) 또는 (B, T, H, W) 등 어떤 shape도 가능
        return:
        동일 shape, 값은 {0, 1, ignore_index}
        """
        gt = gt_occ.to(torch.long)

        out = torch.zeros_like(gt, dtype=torch.long)

        # ignore는 보존
        ignore_mask = (gt == ignore_index)

        # torch.isin 없이도 동작하도록 구현 (버전 안전)
        gmo_ids_t = gt.new_tensor(gmo_ids, dtype=torch.long)
        gmo_mask = (gt.unsqueeze(-1) == gmo_ids_t).any(dim=-1)

        out[gmo_mask] = 1
        out[ignore_mask] = ignore_index
        return out
    
    def compute_gmo_loss(self,
                         pred_occ: torch.Tensor,
                         gt_occ: torch.Tensor,
                         ignore_index: int = 255,
                         occ_dt=None,
                         points_world: torch.Tensor = None,
                         loss_weight=1.0) -> torch.Tensor:
        # pred_occ: [6, 1, 40, 512, 512] with prob values
        # gt_occ: [7] list of [1, 512, 512, 40] with class labels
        pred_occ = pred_occ.squeeze(1) # [6, 40, 512, 512]
        gt_occ = torch.stack(gt_occ, dim=0)[-6:].squeeze(1).permute(0,3,1,2)  # [6, 40, 512, 512]

        gt_occ_gmo = self.gt_occ_to_gmo_binary(gt_occ)

        valid = (gt_occ_gmo != ignore_index)
        target_fg = (gt_occ_gmo == 1).float()

        # if self.iter % 50 == 0:
            # breakpoint()

        eps = 1e-6
        p = pred_occ.clamp(eps, 1.0 - eps)     # prob
        t = target_fg                         # 0/1

        # 마스크
        pos_mask = valid & (t > 0.5)
        neg_mask = valid & (t < 0.5)

        # elementwise BCE (shape 유지, boolean indexing 큰 벡터 생성 피함)
        loss_map = F.binary_cross_entropy(p, t, reduction="none")

        pos_cnt = pos_mask.sum().clamp(min=1).float()
        neg_cnt = neg_mask.sum().clamp(min=1).float()

        loss_pos = (loss_map * pos_mask.float()).sum() / pos_cnt
        loss_neg = (loss_map * neg_mask.float()).sum() / neg_cnt

        # 균형 합산
        loss = 0.5 * loss_pos + 0.5 * loss_neg
        loss_dict = {}
        loss_dict['loss_gmo_bce'] = loss * loss_weight

        self.iter += 1
        self._maybe_save_prob_grid_vis(
            pred_occ_prob=pred_occ,
            points_world=points_world,
            gt_occ_gmo=gt_occ_gmo,
            step=self.iter,
            max_frames=6,
            voxel_center_offset=0.5,
            prob_threshold=0.5,
        )

        return loss_dict

    def compute_query_point_dt_loss(
        self,
        points_world: torch.Tensor,
        occ_dt,
        loss_weight: float = 0.1,
        dt_clip_max: float = None,
        voxel_center_offset: float = 0.5,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = True,
    ):
        """
        DT 기반 query-point loss (pred point -> GT occupancy DT).

        Args:
            points_world:
                [T, Nq, P, 3] or [T, N, 3], world xyz(m).
            occ_dt:
                [B, T, X, Y, Z] or [T, X, Y, Z], DT in meters.
            loss_weight:
                final scalar weight.
            dt_clip_max:
                optional DT upper clipping (meters).
            voxel_center_offset:
                DT grid is defined on voxel centers. Use 0.5 for center-aligned
                world->index mapping: idx = (x-min)/vx - 0.5.
        Returns:
            dict(loss_query_dt=...)
        """
        if occ_dt is None:
            z = points_world.sum() * 0.0
            return {"loss_query_dt": z}

        if not torch.is_tensor(occ_dt):
            if isinstance(occ_dt, np.ndarray):
                occ_dt = torch.from_numpy(occ_dt)
            else:
                raise TypeError(f"occ_dt must be torch.Tensor/np.ndarray, got {type(occ_dt)}")

        if points_world.dim() == 4:
            # [T, Nq, P, 3] -> [T, N, 3]
            T_pred = points_world.shape[0]
            pts_tn3 = points_world.reshape(T_pred, -1, 3)
        elif points_world.dim() == 3:
            # [T, N, 3]
            T_pred = points_world.shape[0]
            pts_tn3 = points_world
        else:
            raise ValueError(f"points_world must be [T,Nq,P,3] or [T,N,3], got {points_world.shape}")

        # collated by pipeline: [B, T, X, Y, Z], current query branch uses B=1.
        if occ_dt.dim() == 6 and occ_dt.shape[2] == 1:
            occ_dt = occ_dt.squeeze(2)
        if occ_dt.dim() == 5:
            if occ_dt.shape[0] != 1:
                raise ValueError(
                    f"current query path expects batch=1 for occ_dt, got {occ_dt.shape}"
                )
            occ_dt = occ_dt[0]
        elif occ_dt.dim() != 4:
            raise ValueError(f"occ_dt must be [B,T,X,Y,Z] or [T,X,Y,Z], got {occ_dt.shape}")

        if occ_dt.shape[0] < T_pred:
            raise ValueError(f"occ_dt time dim too short: {occ_dt.shape[0]} < {T_pred}")
        if occ_dt.shape[0] != T_pred:
            occ_dt = occ_dt[-T_pred:]

        occ_dt = occ_dt.to(device=points_world.device, dtype=torch.float32).contiguous()  # [T, X, Y, Z]
        if dt_clip_max is not None:
            occ_dt = occ_dt.clamp(max=float(dt_clip_max))

        # grid_sample expects input [N,C,D,H,W] with coord order (x,y,z) over (W,H,D).
        # our DT axis is [X,Y,Z], so permute to [Z,Y,X] as [D,H,W].
        dt_tzyx = occ_dt.permute(0, 3, 2, 1).unsqueeze(1).contiguous()  # [T,1,Z,Y,X]

        X = int(occ_dt.shape[1])
        Y = int(occ_dt.shape[2])
        Z = int(occ_dt.shape[3])

        pts = pts_tn3.to(dtype=torch.float32)
        pc_min = pts.new_tensor(self.point_cloud_range[:3])   # [x_min,y_min,z_min]
        extent = pts.new_tensor(self.spatial_extent3d)        # [x_range,y_range,z_range]
        grid_size = pts.new_tensor([float(X), float(Y), float(Z)])
        voxel_size = extent / grid_size

        # world xyz -> continuous voxel-center index in [x,y,z] axes.
        # ex) world at center of voxel k maps to index k when offset=0.5.
        off = float(voxel_center_offset)
        ix = (pts[..., 0] - pc_min[0]) / voxel_size[0] - off
        iy = (pts[..., 1] - pc_min[1]) / voxel_size[1] - off
        iz = (pts[..., 2] - pc_min[2]) / voxel_size[2] - off

        # voxel index -> normalized coords for grid_sample
        if X > 1:
            gx = (ix / float(X - 1)) * 2.0 - 1.0
        else:
            gx = torch.zeros_like(ix)
        if Y > 1:
            gy = (iy / float(Y - 1)) * 2.0 - 1.0
        else:
            gy = torch.zeros_like(iy)
        if Z > 1:
            gz = (iz / float(Z - 1)) * 2.0 - 1.0
        else:
            gz = torch.zeros_like(iz)

        sample_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(2).unsqueeze(2)  # [T,N,1,1,3]
        sampled_dt = F.grid_sample(
            dt_tzyx,
            sample_grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # [T,1,N,1,1]
        sampled_dt = sampled_dt[:, 0, :, 0, 0]  # [T,N]

        loss = sampled_dt.mean() * float(loss_weight)

        return {"loss_query_dt": loss}

    def compute_query_point_repel_loss(
        self,
        points_world: torch.Tensor,
        loss_weight: float = 0.01,
        k: int = 64,
        margin_m: float = 0.5,
    ):
        """
        Toy 코드의 repel_points와 동일한 의도:
        가까이 뭉친 query points에 약한 repulsion 페널티를 부여.

        Args:
            points_world:
                [T, Nq, P, 3] or [T, N, 3], world xyz(m).
            loss_weight:
                final scalar weight.
            k:
                frame마다 랜덤 샘플할 점 개수.
            margin_m:
                pair distance가 이 값보다 작을 때만 페널티.
        Returns:
            dict(loss_query_repel=...)
        """
        if points_world.dim() == 4:
            pts_tn3 = points_world.reshape(points_world.shape[0], -1, 3)
        elif points_world.dim() == 3:
            pts_tn3 = points_world
        else:
            raise ValueError(f"points_world must be [T,Nq,P,3] or [T,N,3], got {points_world.shape}")

        T, N, _ = pts_tn3.shape
        if N <= 1:
            z = pts_tn3.sum() * 0.0
            return {"loss_query_repel": z}

        kk = min(int(k), int(N))
        if kk <= 1:
            z = pts_tn3.sum() * 0.0
            return {"loss_query_repel": z}

        idx = torch.randint(0, N, (T, kk), device=pts_tn3.device)
        p = pts_tn3.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3)).to(torch.float32)  # [T,kk,3]

        d = torch.cdist(p, p)  # [T,kk,kk]
        eye = torch.eye(kk, device=p.device, dtype=torch.bool).unsqueeze(0)
        d = d.masked_fill(eye, 1e9)

        pen = F.relu(float(margin_m) - d)
        loss = pen.mean() * float(loss_weight)
        return {"loss_query_repel": loss}

    def compute_query_intra_repel_loss(
        self,
        points_world: torch.Tensor,
        loss_weight: float = 0.01,
        margin_m: float = 0.5,
        sample_k: int = 32,
        num_points_per_query: int = None,
        chunk_queries: int = 128,
    ):
        """
        Backward-compat name.
        실제 동작은 query center 간(inter-query) repulsion loss.
        프레임별로 query center들 사이 거리가 margin_m보다 가까우면 페널티를 준다.

        Args:
            points_world:
                [T, Q, P, 3] 또는 [T, N, 3] (N=Q*P).
            loss_weight:
                final scalar weight.
            margin_m:
                pair distance가 이 값보다 작을 때만 페널티.
            sample_k:
                프레임당 사용할 query center 수 (Q에서 랜덤 샘플).
                0 또는 음수면 전체 Q 사용.
            num_points_per_query:
                points_world가 [T,N,3]일 때 query당 point 수(P). None이면 self.num_pcd 사용.
            chunk_queries:
                메모리 절약용 frame chunk size.
        Returns:
            dict(loss_query_intra_repel=...)
        """
        if points_world.dim() == 4:
            centers_tq3 = points_world.mean(dim=2)
        elif points_world.dim() == 3:
            T, N, _ = points_world.shape
            P = int(num_points_per_query if num_points_per_query is not None else self.num_pcd)
            if P <= 0:
                raise ValueError(f"num_points_per_query must be positive, got {P}")
            if (N % P) != 0:
                raise ValueError(f"N={N} not divisible by P={P} for inter-query center repel")
            Q = N // P
            centers_tq3 = points_world.view(T, Q, P, 3).mean(dim=2)
        else:
            raise ValueError(f"points_world must be [T,Q,P,3] or [T,N,3], got {points_world.shape}")

        T, Q, _ = centers_tq3.shape
        if Q <= 1:
            z = centers_tq3.sum() * 0.0
            return {"loss_query_intra_repel": z}

        k = int(sample_k)
        if k <= 0 or k > Q:
            k = Q
        if k <= 1:
            z = centers_tq3.sum() * 0.0
            return {"loss_query_intra_repel": z}

        # frame별 query center 랜덤 샘플링
        if k < Q:
            # without replacement per frame
            idx = torch.rand(T, Q, device=centers_tq3.device).argsort(dim=1)[:, :k]
            centers = centers_tq3.gather(1, idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            centers = centers_tq3

        centers = centers.to(torch.float32)  # [T, k, 3]
        tri = torch.triu(torch.ones((k, k), device=centers.device, dtype=torch.bool), diagonal=1)
        pair_count = int(tri.sum().item())
        if pair_count <= 0:
            z = centers.sum() * 0.0
            return {"loss_query_intra_repel": z}

        step = max(1, int(chunk_queries))
        total_pen = centers.new_tensor(0.0)
        total_cnt = 0

        for s in range(0, T, step):
            c = centers[s:s + step]            # [mc, k, 3]
            d = torch.cdist(c, c)              # [mc, k, k]
            pen = F.relu(float(margin_m) - d[:, tri])  # [mc, pair_count]
            total_pen = total_pen + pen.sum()
            total_cnt += pen.numel()

        if total_cnt == 0:
            z = centers.sum() * 0.0
            return {"loss_query_intra_repel": z}

        loss = (total_pen / float(total_cnt)) * float(loss_weight)
        return {"loss_query_intra_repel": loss}

    def compute_query_gt2p_unlabeled_loss(
        self,
        points_world: torch.Tensor,
        gt_occ,
        gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        ignore_index: int = 255,
        loss_weight: float = 0.1,
        max_gt_samples: int = 512,
        pred_use_centers: bool = True,
        num_points_per_query: int = None,
        tau_softmin: float = 0.6,
        chunk_gt: int = 256,
    ):
        """
        라벨-무관(unlabeled) GT->Pred coverage loss.
        - GT에서 GMO 양성 voxel을 샘플링하고
        - 각 GT 샘플이 가장 가까운 예측점으로부터 떨어진 거리를 줄인다.

        Args:
            points_world:
                [T, Q, P, 3] 또는 [T, N, 3], world xyz(m).
            gt_occ:
                보통 list length>=T, each [1, X, Y, Z] 정수 라벨.
                (텐서 입력도 일부 형태는 허용)
            pred_use_centers:
                True면 query당 center(평균점)만 사용해 오버헤드 절감.
            max_gt_samples:
                frame당 GT 양성 샘플 상한.
            tau_softmin:
                softmin 온도. <=0 이면 hard-min 사용.
            chunk_gt:
                GT chunk 크기.
        Returns:
            dict(loss_query_gt2p_unlabeled=...)
        """
        if (self.point_cloud_range is None) or (self.spatial_extent3d is None):
            raise ValueError("point_cloud_range/spatial_extent3d must be set for gt2p loss")

        if points_world.dim() == 4:
            # [T,Q,P,3]
            if pred_use_centers:
                # offsets 분포 영향 줄이기 위해 query별 평균점 사용
                pred_tn3 = points_world.mean(dim=2)  # [T,Q,3]
            else:
                pred_tn3 = points_world.reshape(points_world.shape[0], -1, 3)  # [T,Q*P,3]
        elif points_world.dim() == 3:
            # [T,N,3]
            if pred_use_centers:
                T, N, _ = points_world.shape
                P = int(num_points_per_query if num_points_per_query is not None else self.num_pcd)
                if P <= 0:
                    raise ValueError(f"num_points_per_query must be positive, got {P}")
                if (N % P) != 0:
                    raise ValueError(f"N={N} not divisible by P={P} for center-only gt2p")
                pred_tn3 = points_world.view(T, N // P, P, 3).mean(dim=2)  # [T,Q,3]
            else:
                pred_tn3 = points_world  # [T,N,3]
        else:
            raise ValueError(f"points_world must be [T,Q,P,3] or [T,N,3], got {points_world.shape}")

        T_pred = int(pred_tn3.shape[0])

        # gt_occ -> [T, X, Y, Z]
        if isinstance(gt_occ, (list, tuple)):
            gt_occ_t = torch.stack(gt_occ, dim=0)
        elif torch.is_tensor(gt_occ):
            gt_occ_t = gt_occ
        else:
            raise TypeError(f"gt_occ must be list/tuple/tensor, got {type(gt_occ)}")

        # Common layouts:
        # - [T,1,X,Y,Z] (list stacked)
        # - [1,T,1,X,Y,Z] (batched single-sample)
        # - [T,X,Y,Z]
        if gt_occ_t.dim() == 6 and gt_occ_t.shape[0] == 1 and gt_occ_t.shape[2] == 1:
            gt_occ_t = gt_occ_t[0, :, 0]  # [T,X,Y,Z]
        elif gt_occ_t.dim() == 5 and gt_occ_t.shape[1] == 1:
            gt_occ_t = gt_occ_t[:, 0]      # [T,X,Y,Z]
        elif gt_occ_t.dim() == 4:
            pass
        else:
            raise ValueError(f"Unsupported gt_occ shape for gt2p: {tuple(gt_occ_t.shape)}")

        if gt_occ_t.shape[0] < T_pred:
            raise ValueError(f"gt_occ time dim too short: {gt_occ_t.shape[0]} < {T_pred}")
        if gt_occ_t.shape[0] != T_pred:
            gt_occ_t = gt_occ_t[-T_pred:]

        # [T,X,Y,Z] -> [T,Z,X,Y] (same convention used in compute_gmo_loss)
        gt_occ_t = gt_occ_t.permute(0, 3, 1, 2).contiguous()
        gt_bin = self.gt_occ_to_gmo_binary(gt_occ_t, gmo_ids=gmo_ids, ignore_index=ignore_index)

        Z = int(gt_bin.shape[1])
        X = int(gt_bin.shape[2])
        Y = int(gt_bin.shape[3])

        pred_f = pred_tn3.to(torch.float32)
        pc_min = pred_f.new_tensor(self.point_cloud_range[:3])
        extent = pred_f.new_tensor(self.spatial_extent3d)
        grid_size = pred_f.new_tensor([float(X), float(Y), float(Z)])
        voxel_size = extent / grid_size  # [vx,vy,vz]

        tau = float(tau_softmin)
        max_gt = int(max_gt_samples)
        step = max(1, int(chunk_gt))

        loss_acc = pred_f.new_tensor(0.0)
        valid_frames = 0

        for t in range(T_pred):
            pos = (gt_bin[t] == 1).nonzero(as_tuple=False)  # [M,3] in (z,x,y)
            if pos.numel() == 0:
                continue

            if max_gt > 0 and pos.shape[0] > max_gt:
                perm = torch.randperm(pos.shape[0], device=pos.device)[:max_gt]
                pos = pos[perm]

            pos = pos.to(device=pred_f.device)
            z = pos[:, 0].to(torch.float32)
            x = pos[:, 1].to(torch.float32)
            y = pos[:, 2].to(torch.float32)

            gt_xyz = torch.stack(
                [
                    pc_min[0] + (x + 0.5) * voxel_size[0],
                    pc_min[1] + (y + 0.5) * voxel_size[1],
                    pc_min[2] + (z + 0.5) * voxel_size[2],
                ],
                dim=-1,
            )  # [M,3]

            pred_xyz = pred_f[t]  # [N,3]
            if pred_xyz.numel() == 0:
                continue

            sum_t = pred_xyz.new_tensor(0.0)
            cnt_t = 0
            for s in range(0, gt_xyz.shape[0], step):
                g = gt_xyz[s:s + step]  # [mc,3]
                d = torch.cdist(g.unsqueeze(0), pred_xyz.unsqueeze(0)).squeeze(0)  # [mc,N]

                if tau > 0.0:
                    dist = -tau * torch.logsumexp(-d / tau, dim=1)  # [mc]
                else:
                    dist = d.min(dim=1).values

                sum_t = sum_t + dist.sum()
                cnt_t += int(dist.numel())

            if cnt_t > 0:
                loss_acc = loss_acc + (sum_t / float(cnt_t))
                valid_frames += 1

        if valid_frames == 0:
            z = pred_tn3.sum() * 0.0
            return {"loss_query_gt2p_unlabeled": z}

        loss = (loss_acc / float(valid_frames)) * float(loss_weight)
        return {"loss_query_gt2p_unlabeled": loss}


    @torch.no_grad()
    def _sample_rows(self, idx: torch.Tensor, max_rows: int) -> torch.Tensor:
        # idx: (K, d)
        if max_rows is None or idx.shape[0] <= max_rows:
            return idx
        perm = torch.randperm(idx.shape[0], device=idx.device)[:max_rows]
        return idx[perm]

    # def compute_point_gt_knn_loss(
    #         self,
    #         points_world: torch.Tensor,
    #         gt_occ: torch.Tensor,
    #         voxelizer,
    #         gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
    #         ignore_index: int = 255,
    #         use_centers_only: bool = True,
    #         num_points_per_query: int = None,
    #         max_pred: int = 512,
    #         max_gt: int = 4096,
    #         chunk_gt: int = 1024,
    #         sqrt_dist: bool = True,
    #         eps: float = 1e-6,
    #     ) -> torch.Tensor:
    #         """
    #         목적: overlap=0 이어도 query center가 GT 양성 영역 쪽으로 이동하게 만드는 attractive loss.

    #         points_world:
    #         (B, T, N, 3) where N = num_queries * num_points_per_query
    #         또는 (B, T, num_queries, 3) 같은 center만 모아둔 텐서도 허용(그 경우 use_centers_only=False로 두면 됨)

    #         gt_occ:
    #         (B, T, D, H, W) 정수 라벨 occupancy (ignore_index 포함 가능)

    #         voxelizer:
    #         point_cloud_range, voxel_size를 가지고 있어야 함
    #         """
    #         device = points_world.device

    #         # 1) GT를 GMO binary로 변환
    #         gt_occ = torch.stack(gt_occ, dim=0)[-6:].squeeze(1).permute(0,3,1,2)  # [6, 40, 512, 512]
    #         gt_bin = self.gt_occ_to_gmo_binary(gt_occ, gmo_ids=gmo_ids, ignore_index=ignore_index)
    #         # gt_bin: same shape, {0,1,ignore}

    #         # 2) centers만 쓰기 (연산 줄이기)
    #         if use_centers_only:
    #             if points_world.dim() != 4:
    #                 raise ValueError(f"points_world must be (B,T,N,3), got {points_world.shape}")
    #             if num_points_per_query is None:
    #                 num_points_per_query = getattr(self, "num_points", None)
    #             if num_points_per_query is None:
    #                 raise ValueError("num_points_per_query is required when use_centers_only=True")

    #             B, T, N, _ = points_world.shape
    #             step = int(num_points_per_query)
    #             if N % step != 0:
    #                 raise ValueError(f"N={N} is not divisible by num_points_per_query={step}")

    #             center_idx = torch.arange(0, N, step, device=device)
    #             pred_centers = points_world.index_select(2, center_idx)  # (B,T,Nq,3)
    #         else:
    #             pred_centers = points_world  # assume already (B,T,Nq,3) or (B,T,N,3)

    #         # voxelizer params
    #         pcr = voxelizer.point_cloud_range
    #         vsize = voxelizer.voxel_size
    #         vsize = voxelizer.voxel_size

    #         if isinstance(vsize, (float, int)):
    #             vx = vy = vz = float(vsize)
    #         else:
    #             vx, vy, vz = float(vsize[0]), float(vsize[1]), float(vsize[2])

    #         min_x, min_y, min_z = float(pcr[0]), float(pcr[1]), float(pcr[2])

    #         B = gt_bin.shape[0]
    #         T = gt_bin.shape[1]

    #         losses = []
    #         for b in range(B):
    #             for t in range(T):
    #                 g = gt_bin[b, t]  # (D,H,W)
    #                 pos_mask = (g == 1)

    #                 # GT 양성이 없으면 스킵
    #                 if not pos_mask.any():
    #                     continue

    #                 # (K,3) with order (z,x,y)
    #                 pos_idx = pos_mask.nonzero(as_tuple=False)

    #                 # GT 샘플링으로 비용 제한
    #                 if max_gt is not None and pos_idx.shape[0] > max_gt:
    #                     with torch.no_grad():
    #                         perm = torch.randperm(pos_idx.shape[0], device=device)[:max_gt]
    #                     pos_idx = pos_idx[perm]

    #                 iz = pos_idx[:, 0].to(torch.float32)
    #                 ix = pos_idx[:, 1].to(torch.float32)
    #                 iy = pos_idx[:, 2].to(torch.float32)

    #                 gx = min_x + (ix + 0.5) * vx
    #                 gy = min_y + (iy + 0.5) * vy
    #                 gz = min_z + (iz + 0.5) * vz
    #                 gt_xyz = torch.stack([gx, gy, gz], dim=-1)  # (K,3)

    #                 pred = pred_centers[b, t]  # (Np,3)

    #                 # pred 샘플링으로 비용 제한
    #                 if max_pred is not None and pred.shape[0] > max_pred:
    #                     with torch.no_grad():
    #                         perm = torch.randperm(pred.shape[0], device=device)[:max_pred]
    #                     pred = pred[perm]

    #                 # KNN: pred -> nearest GT
    #                 # 메모리 줄이기 위해 gt를 chunk로 쪼갬
    #                 min_d2 = torch.full((pred.shape[0],), float("inf"), device=device, dtype=torch.float32)

    #                 for s in range(0, gt_xyz.shape[0], chunk_gt):
    #                     gt_chunk = gt_xyz[s : s + chunk_gt]  # (C,3)
    #                     d2 = (pred[:, None, :] - gt_chunk[None, :, :]).pow(2).sum(dim=-1)  # (Np,C)
    #                     min_d2 = torch.minimum(min_d2, d2.min(dim=1).values)

    #                 if sqrt_dist:
    #                     dist = torch.sqrt(min_d2 + eps)
    #                     loss_bt = dist.mean()
    #                 else:
    #                     loss_bt = min_d2.mean()

    #                 losses.append(loss_bt)

    #         if len(losses) == 0:
    #             # GT pos가 전부 없던 경우
    #             return pred_centers.sum() * 0.0

    #         return torch.stack(losses).mean()

    def compute_gmo_center_knn_loss(
        self,
        center_logits: torch.Tensor,   # [T_pred, Nq, 3] (unconstrained)
        gt_occ,                        # list length >= T_pred, each [1, H, W, Z] (int labels)
        voxelizer,
        gmo_ids=(2, 3, 4, 5, 6, 7, 9, 10),
        ignore_index: int = 255,
        loss_weight: float = 0.1,
        max_gt: int = 4096,
        chunk_gt: int = 1024,
        max_pred: int = None,          # 보통 Nq=100이라 None 권장
        sqrt_dist: bool = True,
        eps: float = 1e-6,
    ):
        """
        목적:
          예측 query center들이 GT GMO 양성 영역(복셀 center 집합) 근처로 이동하도록 유도.

        정의(프레임별):
          pred_centers[t] = sigmoid(center_logits[t])를 world 범위로 변환한 [Nq,3]
          gt_points[t]    = GT에서 gmo 양성 복셀들의 world center 좌표 [K,3]
          loss[t] = mean_i min_j || pred_centers_i - gt_points_j ||  (KNN, pred->gt)

        반환:
          dict(loss_gmo_center_knn=...)
        """
        device = center_logits.device

        # 1) pred centers (world)
        pred_centers = sigmoid_to_world_from_range(
            center_logits, self.point_cloud_range, self.spatial_extent3d
        )  # [T_pred, Nq, 3]
        T_pred = pred_centers.shape[0]

        # 2) GT를 compute_gmo_loss와 동일한 방식으로 정렬: [T_pred, D, H, W]
        # gt_occ: [T_gt] list of [1,H,W,Z]
        gt_occ_t = torch.stack(gt_occ, dim=0)[-T_pred:].squeeze(1).permute(0, 3, 1, 2)  # [T_pred, Z, H, W]
        gt_bin = self.gt_occ_to_gmo_binary(gt_occ_t, gmo_ids=gmo_ids, ignore_index=ignore_index)

        # 3) voxelizer에서 world 변환 파라미터 확보
        # voxelizer.pc_min = [x_min,y_min,z_min], voxelizer.vs = [vx,vy,vz]
        if hasattr(voxelizer, "pc_min") and hasattr(voxelizer, "vs"):
            pc_min = voxelizer.pc_min.to(device=device, dtype=torch.float32)
            vs = voxelizer.vs.to(device=device, dtype=torch.float32)
            x_min, y_min, z_min = pc_min[0], pc_min[1], pc_min[2]
            vx, vy, vz = vs[0], vs[1], vs[2]
        else:
            pcr = voxelizer.point_cloud_range
            vsize = voxelizer.voxel_size
            if isinstance(vsize, (float, int)):
                vx = vy = vz = float(vsize)
            else:
                vx, vy, vz = float(vsize[0]), float(vsize[1]), float(vsize[2])
            x_min, y_min, z_min = float(pcr[0]), float(pcr[1]), float(pcr[2])
            x_min = torch.tensor(x_min, device=device, dtype=torch.float32)
            y_min = torch.tensor(y_min, device=device, dtype=torch.float32)
            z_min = torch.tensor(z_min, device=device, dtype=torch.float32)
            vx = torch.tensor(vx, device=device, dtype=torch.float32)
            vy = torch.tensor(vy, device=device, dtype=torch.float32)
            vz = torch.tensor(vz, device=device, dtype=torch.float32)

        losses = []
        for t in range(T_pred):
            g = gt_bin[t]              # [D,H,W]
            pos = (g == 1)

            if not pos.any():
                continue

            idx = pos.nonzero(as_tuple=False)  # [K,3] in (d,h,w)

            # GT 샘플링
            if max_gt is not None and idx.shape[0] > max_gt:
                with torch.no_grad():
                    perm = torch.randperm(idx.shape[0], device=device)[:max_gt]
                idx = idx[perm]

            d = idx[:, 0].to(torch.float32)
            h = idx[:, 1].to(torch.float32)
            w = idx[:, 2].to(torch.float32)

            # (d,h,w) -> world (x,y,z)
            gx = x_min + (w + 0.5) * vx
            gy = y_min + (h + 0.5) * vy
            gz = z_min + (d + 0.5) * vz
            gt_xyz = torch.stack([gx, gy, gz], dim=-1)  # [K,3]

            pred = pred_centers[t].to(torch.float32)    # [Nq,3]

            if max_pred is not None and pred.shape[0] > max_pred:
                with torch.no_grad():
                    perm = torch.randperm(pred.shape[0], device=device)[:max_pred]
                pred = pred[perm]

            # pred -> nearest GT (chunked)
            min_d2 = torch.full((pred.shape[0],), float("inf"), device=device, dtype=torch.float32)
            for s in range(0, gt_xyz.shape[0], chunk_gt):
                gt_chunk = gt_xyz[s:s + chunk_gt]  # [C,3]
                d2 = (pred[:, None, :] - gt_chunk[None, :, :]).pow(2).sum(dim=-1)  # [Np,C]
                min_d2 = torch.minimum(min_d2, d2.min(dim=1).values)

            if sqrt_dist:
                loss_t = torch.sqrt(min_d2 + eps).mean()
            else:
                loss_t = min_d2.mean()

            losses.append(loss_t)

        if len(losses) == 0:
            # 모든 프레임에서 GT 양성이 없으면 0 loss
            z = center_logits.sum() * 0.0
            return {"loss_gmo_center_knn": z}

        loss = torch.stack(losses).mean() * float(loss_weight)
        return {"loss_gmo_center_knn": loss}




if __name__ == "__main__":
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    spatial_extent3d = [102.4, 102.4, 8.0]

    net = QueryHead(embed_dim=128, center_out_dim=3, offset_out_dim=3, num_pcd=256,
                    point_cloud_range=point_cloud_range,
                    spatial_extent3d=spatial_extent3d)
    
    query = torch.randn(3, 100, 128)  # [T, num_queries, embed_dim]
    points_world, center_logits, offset_logits = net(query)

    print(points_world.shape)  # [3, 100, 256, 3]
    print(center_logits.shape)  # [3, 100, 3]
    print(offset_logits.shape)  # [3, 100, 256, 3]
