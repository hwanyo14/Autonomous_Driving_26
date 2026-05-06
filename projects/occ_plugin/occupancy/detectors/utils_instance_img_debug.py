import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class EfficientOCFInstanceImgDebugMixin:
    """Instance-aware image projection and visualization debug helpers."""

    _BOX_EDGES = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    @staticmethod
    def _to_display_u8(img_chw: torch.Tensor):
        x = img_chw.detach().to(torch.float32).cpu()
        if x.dim() != 3:
            raise ValueError(f"Expected CHW image tensor, got shape={tuple(x.shape)}")
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] > 3:
            x = x[:3]

        x_min = float(x.min().item())
        x_max = float(x.max().item())
        if x_max <= x_min + 1e-6:
            x = torch.zeros_like(x)
        else:
            x = (x - x_min) / (x_max - x_min)
        return (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()

    @staticmethod
    def _instance_color(instance_id: int, cls_id: int = None) -> Tuple[int, int, int]:
        class_palette = {
            0: (180, 180, 180),
            2: (231, 76, 60),
            3: (243, 156, 18),
            4: (46, 204, 113),
            5: (52, 152, 219),
            6: (155, 89, 182),
            7: (26, 188, 156),
            9: (241, 196, 15),
            10: (230, 126, 34),
        }
        if cls_id is not None and int(cls_id) in class_palette:
            return class_palette[int(cls_id)]
        iid = int(instance_id)
        return (
            int((37 * iid + 17) % 255),
            int((97 * iid + 29) % 255),
            int((53 * iid + 71) % 255),
        )

    def _build_boxes_from_instance_occ(
        self,
        seg_instance_txyz: torch.Tensor,
        max_instances: int,
    ):
        if seg_instance_txyz is None or seg_instance_txyz.dim() != 4:
            return [], []

        x_size, y_size, z_size = [int(v) for v in seg_instance_txyz.shape[1:]]
        pc = torch.as_tensor(self.point_cloud_range, dtype=torch.float32, device=seg_instance_txyz.device)
        voxel = torch.tensor(
            [
                (pc[3] - pc[0]) / float(x_size),
                (pc[4] - pc[1]) / float(y_size),
                (pc[5] - pc[2]) / float(z_size),
            ],
            dtype=torch.float32,
            device=seg_instance_txyz.device,
        )
        pc_min = pc[:3]

        frame_boxes: List[Dict[int, torch.Tensor]] = []
        global_counts: Dict[int, int] = {}
        for t in range(int(seg_instance_txyz.shape[0])):
            occ_t = seg_instance_txyz[t].to(torch.long)
            inst_ids = torch.unique(occ_t)
            keep = (inst_ids > 0) & (inst_ids != 255)
            inst_ids = inst_ids[keep]
            cur_boxes = {}
            for iid_t in inst_ids.tolist():
                iid = int(iid_t)
                pts = torch.nonzero(occ_t == iid, as_tuple=False)
                if pts.numel() <= 0:
                    continue
                mins = pts.min(dim=0).values.to(torch.float32)
                maxs = pts.max(dim=0).values.to(torch.float32) + 1.0
                wmin = pc_min + mins * voxel
                wmax = pc_min + maxs * voxel
                corners = torch.tensor(
                    [
                        [wmin[0], wmin[1], wmin[2]],
                        [wmax[0], wmin[1], wmin[2]],
                        [wmax[0], wmax[1], wmin[2]],
                        [wmin[0], wmax[1], wmin[2]],
                        [wmin[0], wmin[1], wmax[2]],
                        [wmax[0], wmin[1], wmax[2]],
                        [wmax[0], wmax[1], wmax[2]],
                        [wmin[0], wmax[1], wmax[2]],
                    ],
                    dtype=torch.float32,
                    device=seg_instance_txyz.device,
                )
                cur_boxes[iid] = corners
                global_counts[iid] = int(global_counts.get(iid, 0)) + int(pts.shape[0])
            frame_boxes.append(cur_boxes)

        ranked = sorted(global_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        selected_ids = [int(iid) for iid, _ in ranked[: max(1, int(max_instances))]]

        filtered = []
        for cur in frame_boxes:
            filtered.append({iid: cur[iid] for iid in selected_ids if iid in cur})
        return filtered, selected_ids

    @staticmethod
    def _build_class_map(seg_cls_inst_tcxzy: torch.Tensor, selected_ids: List[int]):
        if seg_cls_inst_tcxzy is None or seg_cls_inst_tcxzy.dim() != 5 or len(selected_ids) <= 0:
            return {}

        cls_txyz = seg_cls_inst_tcxzy[:, 0].to(torch.long)
        inst_txyz = seg_cls_inst_tcxzy[:, 1].to(torch.long)
        out = {}
        for iid in selected_ids:
            vox = cls_txyz[inst_txyz == int(iid)]
            if vox.numel() <= 0:
                continue
            vox = vox[(vox > 0) & (vox != 255)]
            if vox.numel() <= 0:
                continue
            vals, cnts = torch.unique(vox, return_counts=True)
            out[int(iid)] = int(vals[torch.argmax(cnts)].item())
        return out

    def _build_lidar_frame_to_present(
        self,
        future_egomotion_seq44: torch.Tensor,
        frame_indices: List[int],
        present_global_idx: int,
    ):
        if len(frame_indices) <= 0:
            return []
        with torch.cuda.amp.autocast(enabled=False):
            ego = future_egomotion_seq44.to(torch.float32)
            eye = torch.eye(4, dtype=ego.dtype, device=ego.device)
            present_g = int(present_global_idx)
            seq_len = int(ego.shape[0])

            def compose_forward(start_g: int, end_g: int):
                # Compose transforms from frame start_g -> end_g, using adjacent g->g+1 matrices.
                if end_g <= start_g:
                    return eye.clone()
                out = eye.clone()
                for g in range(int(start_g), int(end_g)):
                    if g < 0 or g >= seq_len:
                        continue
                    out = ego[g] @ out
                return out

            out = []
            for g in frame_indices:
                g = int(g)
                if g == present_g:
                    out.append(eye.clone())
                elif g < present_g:
                    out.append(compose_forward(g, present_g))
                else:
                    present_to_g = compose_forward(present_g, g)
                    out.append(torch.inverse(present_to_g))
            return out

    @staticmethod
    def _project_corners_to_aug_uv(corners_lidar: torch.Tensor, rot, trans, intrins, post_rot, post_trans):
        pts = corners_lidar.to(torch.float32)
        cam = pts - trans.to(torch.float32).view(1, 3)
        cam = (torch.inverse(rot.to(torch.float32)) @ cam.t()).t()
        depth = cam[:, 2]
        valid_depth = depth > 1e-5
        uv = (intrins.to(torch.float32) @ cam.t()).t()
        uv = uv[:, :2] / uv[:, 2:3].clamp(min=1e-6)
        uv = (post_rot.to(torch.float32)[:2, :2] @ uv.t()).t() + post_trans.to(torch.float32)[:2].view(1, 2)
        return uv, depth, valid_depth

    @staticmethod
    def _project_points_to_aug_uv(points_lidar: torch.Tensor, rot, trans, intrins, post_rot, post_trans):
        pts = points_lidar.to(torch.float32)
        cam = pts - trans.to(torch.float32).view(1, 3)
        cam = (torch.inverse(rot.to(torch.float32)) @ cam.t()).t()
        depth = cam[:, 2]
        valid_depth = depth > 1e-5
        uv = (intrins.to(torch.float32) @ cam.t()).t()
        uv = uv[:, :2] / uv[:, 2:3].clamp(min=1e-6)
        uv = (post_rot.to(torch.float32)[:2, :2] @ uv.t()).t() + post_trans.to(torch.float32)[:2].view(1, 2)
        return uv, depth, valid_depth

    @torch.no_grad()
    def build_query_inst_depth_targets(
        self,
        gt_inst_center_world_tn3: torch.Tensor,
        gt_inst_center_valid_tn: torch.Tensor,
        gt_inst_ids_n: torch.Tensor,
        future_egomotion: torch.Tensor,
        query_match_inputs: dict,
    ) -> dict:
        if (
            (not isinstance(query_match_inputs, dict))
            or (not torch.is_tensor(gt_inst_center_world_tn3))
            or (not torch.is_tensor(gt_inst_center_valid_tn))
            or gt_inst_center_world_tn3.dim() != 3
            or gt_inst_center_valid_tn.dim() != 2
            or gt_inst_center_world_tn3.shape[:2] != gt_inst_center_valid_tn.shape
        ):
            return None
        rots_tn33 = query_match_inputs.get("rots_tn33", None)
        trans_tn3 = query_match_inputs.get("trans_tn3", None)
        intrins_tn33 = query_match_inputs.get("intrins_tn33", None)
        post_rots_tn33 = query_match_inputs.get("post_rots_tn33", None)
        post_trans_tn3 = query_match_inputs.get("post_trans_tn3", None)
        img_h = int(query_match_inputs.get("img_h", 0))
        img_w = int(query_match_inputs.get("img_w", 0))
        frame_start = int(query_match_inputs.get("frame_start_idx", 0))
        frame_end = int(query_match_inputs.get("frame_end_idx_exclusive", 0))
        if (
            (not torch.is_tensor(rots_tn33))
            or (not torch.is_tensor(trans_tn3))
            or (not torch.is_tensor(intrins_tn33))
            or (not torch.is_tensor(post_rots_tn33))
            or (not torch.is_tensor(post_trans_tn3))
            or img_h <= 1
            or img_w <= 1
        ):
            return None

        proj_device = rots_tn33.device
        t_hist = int(
            min(
                gt_inst_center_world_tn3.shape[0],
                gt_inst_center_valid_tn.shape[0],
                rots_tn33.shape[0],
                trans_tn3.shape[0],
                intrins_tn33.shape[0],
                post_rots_tn33.shape[0],
                post_trans_tn3.shape[0],
            )
        )
        if t_hist <= 0:
            return None
        present_global_idx = int(getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1)))
        query_present_only = bool(getattr(self, "query_present_only", False))
        frame_indices_hist = list(range(frame_start, max(frame_start, frame_end)))
        if len(frame_indices_hist) < t_hist:
            frame_indices_hist = list(
                range(int(self.time_receptive_field - t_hist), int(self.time_receptive_field))
            )
        frame_indices_hist = frame_indices_hist[:t_hist]
        selected_src_idx = list(range(t_hist))
        if query_present_only:
            present_candidates = [
                int(i) for i, g in enumerate(frame_indices_hist) if int(g) == int(present_global_idx)
            ]
            if len(present_candidates) > 0:
                selected_src_idx = [present_candidates[0]]
            else:
                fallback_idx = int(present_global_idx - frame_start)
                fallback_idx = max(0, min(t_hist - 1, fallback_idx))
                selected_src_idx = [fallback_idx]
        t_out = int(len(selected_src_idx))
        if t_out <= 0:
            return None
        src_idx_t = torch.as_tensor(selected_src_idx, device=proj_device, dtype=torch.long)
        output_global_idx_t = torch.as_tensor(
            [int(frame_indices_hist[int(i)]) for i in selected_src_idx],
            device=proj_device,
            dtype=torch.long,
        )

        centers_tn3 = gt_inst_center_world_tn3[:t_hist].to(device=proj_device, dtype=torch.float32)
        valid_tn = gt_inst_center_valid_tn[:t_hist].to(device=proj_device, dtype=torch.bool)
        n_inst = int(centers_tn3.shape[1])
        n_cam = int(rots_tn33.shape[1])
        if torch.is_tensor(gt_inst_ids_n) and gt_inst_ids_n.numel() == n_inst:
            inst_ids_n = gt_inst_ids_n.to(device=proj_device, dtype=torch.long).contiguous()
        else:
            inst_ids_n = torch.arange(n_inst, device=proj_device, dtype=torch.long)

        num_bins = int(getattr(self, "query_inst_depth_num_bins", 64))
        if num_bins <= 0:
            return None
        range_mode = str(getattr(self, "query_inst_depth_range_mode", "dbound")).lower()
        depth_min = float(getattr(self, "query_inst_depth_min", 0.0))
        depth_max = float(getattr(self, "query_inst_depth_max", 0.0))
        if range_mode == "dbound":
            dbound = None
            if hasattr(self, "img_view_transformer") and hasattr(self.img_view_transformer, "grid_config"):
                dbound = self.img_view_transformer.grid_config.get("dbound", None)
            if isinstance(dbound, (list, tuple)) and len(dbound) >= 2:
                depth_min = float(dbound[0])
                depth_max = float(dbound[1])
        if (not (depth_max > depth_min)) or (not torch.isfinite(torch.tensor(depth_min))) or (not torch.isfinite(torch.tensor(depth_max))):
            return None
        bin_size = (depth_max - depth_min) / float(num_bins)
        if not (bin_size > 0.0):
            return None

        depth_bin_tcn = torch.full(
            (t_out, n_cam, n_inst),
            fill_value=-1,
            device=proj_device,
            dtype=torch.long,
        )
        depth_valid_tcn = torch.zeros((t_out, n_cam, n_inst), device=proj_device, dtype=torch.bool)
        depth_value_tcn = torch.zeros((t_out, n_cam, n_inst), device=proj_device, dtype=torch.float32)
        if n_inst <= 0 or n_cam <= 0:
            return {
                "gt_inst_depth_bin_tcn": depth_bin_tcn,
                "gt_inst_depth_valid_tcn": depth_valid_tcn,
                "gt_inst_depth_value_tcn": depth_value_tcn,
                "gt_inst_ids_n": inst_ids_n,
                "attn_t_idx_t": src_idx_t,
                "query_output_global_frame_indices": output_global_idx_t,
                "query_present_global_idx": depth_value_tcn.new_tensor(
                    [present_global_idx], dtype=torch.long
                ),
                "present_source_local_idx": depth_value_tcn.new_tensor(
                    [int(selected_src_idx[0])], dtype=torch.long
                ),
                "dbg_query_inst_depth_target_total_count": depth_value_tcn.new_tensor(0.0),
                "dbg_query_inst_depth_target_valid_count": depth_value_tcn.new_tensor(0.0),
                "dbg_query_inst_depth_target_inst_count": depth_value_tcn.new_tensor(float(n_inst)),
                "dbg_query_inst_depth_target_num_bins": depth_value_tcn.new_tensor(float(num_bins)),
                "dbg_query_inst_depth_target_depth_min": depth_value_tcn.new_tensor(depth_min),
                "dbg_query_inst_depth_target_depth_max": depth_value_tcn.new_tensor(depth_max),
            }

        if not torch.is_tensor(future_egomotion):
            return {
                "gt_inst_depth_bin_tcn": depth_bin_tcn,
                "gt_inst_depth_valid_tcn": depth_valid_tcn,
                "gt_inst_depth_value_tcn": depth_value_tcn,
                "gt_inst_ids_n": inst_ids_n,
                "attn_t_idx_t": src_idx_t,
                "query_output_global_frame_indices": output_global_idx_t,
                "query_present_global_idx": depth_value_tcn.new_tensor(
                    [present_global_idx], dtype=torch.long
                ),
                "present_source_local_idx": depth_value_tcn.new_tensor(
                    [int(selected_src_idx[0])], dtype=torch.long
                ),
                "dbg_query_inst_depth_target_total_count": depth_value_tcn.new_tensor(float(t_out * n_cam * n_inst)),
                "dbg_query_inst_depth_target_valid_count": depth_value_tcn.new_tensor(0.0),
                "dbg_query_inst_depth_target_inst_count": depth_value_tcn.new_tensor(float(n_inst)),
                "dbg_query_inst_depth_target_num_bins": depth_value_tcn.new_tensor(float(num_bins)),
                "dbg_query_inst_depth_target_depth_min": depth_value_tcn.new_tensor(depth_min),
                "dbg_query_inst_depth_target_depth_max": depth_value_tcn.new_tensor(depth_max),
            }

        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[0]
        ego = ego.to(device=proj_device, dtype=torch.float32)
        frame_indices = [int(frame_indices_hist[int(i)]) for i in selected_src_idx]
        lidar_t_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=frame_indices,
            present_global_idx=present_global_idx,
        )
        if len(lidar_t_to_present) != t_out:
            return None

        for out_t_idx in range(t_out):
            src_t_idx = int(selected_src_idx[out_t_idx])
            valid_inst_t = valid_tn[src_t_idx]
            if not bool(valid_inst_t.any().item()):
                continue
            inst_idx_t = torch.nonzero(valid_inst_t, as_tuple=False).squeeze(1)
            centers_present = centers_tn3[src_t_idx].index_select(0, inst_idx_t)
            tf_inv = torch.inverse(lidar_t_to_present[out_t_idx]).to(device=proj_device, dtype=torch.float32)
            centers_h = torch.cat(
                [
                    centers_present,
                    torch.ones((centers_present.shape[0], 1), device=proj_device, dtype=torch.float32),
                ],
                dim=1,
            )
            centers_lidar_t = (tf_inv @ centers_h.t()).t()[:, :3]
            for cam in range(n_cam):
                uv, depth, valid_depth = self._project_points_to_aug_uv(
                    points_lidar=centers_lidar_t,
                    rot=rots_tn33[src_t_idx, cam],
                    trans=trans_tn3[src_t_idx, cam],
                    intrins=intrins_tn33[src_t_idx, cam],
                    post_rot=post_rots_tn33[src_t_idx, cam],
                    post_trans=post_trans_tn3[src_t_idx, cam],
                )
                valid = (
                    valid_depth
                    & torch.isfinite(depth)
                    & torch.isfinite(uv).all(dim=-1)
                    & (uv[:, 0] >= 0.0)
                    & (uv[:, 0] <= float(img_w - 1))
                    & (uv[:, 1] >= 0.0)
                    & (uv[:, 1] <= float(img_h - 1))
                )
                if not bool(valid.any().item()):
                    continue
                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                depth_valid = depth[valid_idx]
                in_range = (depth_valid >= depth_min) & (depth_valid < depth_max)
                if not bool(in_range.any().item()):
                    continue
                valid_idx = valid_idx[in_range]
                depth_valid = depth_valid[in_range]
                inst_cols = inst_idx_t.index_select(0, valid_idx)
                bin_idx = torch.floor((depth_valid - depth_min) / bin_size).to(torch.long)
                bin_idx = torch.clamp(bin_idx, min=0, max=num_bins - 1)
                depth_bin_tcn[out_t_idx, cam, inst_cols] = bin_idx
                depth_valid_tcn[out_t_idx, cam, inst_cols] = True
                depth_value_tcn[out_t_idx, cam, inst_cols] = depth_valid.to(torch.float32)

        valid_count = depth_valid_tcn.to(torch.float32).sum()
        out = {
            "gt_inst_depth_bin_tcn": depth_bin_tcn,
            "gt_inst_depth_valid_tcn": depth_valid_tcn,
            "gt_inst_depth_value_tcn": depth_value_tcn,
            "gt_inst_ids_n": inst_ids_n,
            "attn_t_idx_t": src_idx_t,
            "query_output_global_frame_indices": output_global_idx_t,
            "query_present_global_idx": depth_value_tcn.new_tensor(
                [present_global_idx], dtype=torch.long
            ),
            "present_source_local_idx": depth_value_tcn.new_tensor(
                [int(selected_src_idx[0])], dtype=torch.long
            ),
            "dbg_query_inst_depth_target_total_count": depth_value_tcn.new_tensor(float(t_out * n_cam * n_inst)),
            "dbg_query_inst_depth_target_valid_count": valid_count,
            "dbg_query_inst_depth_target_inst_count": depth_value_tcn.new_tensor(float(n_inst)),
            "dbg_query_inst_depth_target_num_bins": depth_value_tcn.new_tensor(float(num_bins)),
            "dbg_query_inst_depth_target_depth_min": depth_value_tcn.new_tensor(depth_min),
            "dbg_query_inst_depth_target_depth_max": depth_value_tcn.new_tensor(depth_max),
        }
        if bool(valid_count.item() > 0):
            valid_depths = depth_value_tcn[depth_valid_tcn]
            out["dbg_query_inst_depth_target_valid_min"] = valid_depths.min()
            out["dbg_query_inst_depth_target_valid_max"] = valid_depths.max()
            out["dbg_query_inst_depth_target_valid_mean"] = valid_depths.mean()
        else:
            z = depth_value_tcn.new_tensor(0.0)
            out["dbg_query_inst_depth_target_valid_min"] = z
            out["dbg_query_inst_depth_target_valid_max"] = z
            out["dbg_query_inst_depth_target_valid_mean"] = z
        return out

    @staticmethod
    def _convex_hull(points_xy: torch.Tensor):
        pts = points_xy.to(torch.float32).detach().cpu().tolist()
        if len(pts) <= 1:
            return pts
        pts = sorted(set((float(x), float(y)) for x, y in pts))
        if len(pts) <= 2:
            return pts

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        return lower[:-1] + upper[:-1]

    @staticmethod
    def _polygon_to_mask(points_xy, height: int, width: int):
        if len(points_xy) < 3:
            return None
        try:
            from PIL import Image, ImageDraw
            import numpy as np
        except Exception:
            return None
        mask_img = Image.new("L", (int(width), int(height)), color=0)
        draw = ImageDraw.Draw(mask_img)
        draw.polygon([(float(x), float(y)) for x, y in points_xy], fill=1)
        mask_np = np.array(mask_img, dtype=np.uint8)
        return torch.from_numpy(mask_np.astype("bool"))

    @staticmethod
    def _feature_overlay(base_u8, masks_nhw: torch.Tensor, colors: List[Tuple[int, int, int]]):
        import numpy as np

        out = base_u8.astype(np.float32) / 255.0
        n_inst = int(masks_nhw.shape[0]) if masks_nhw is not None else 0
        for i in range(n_inst):
            m = masks_nhw[i].detach().cpu().numpy() > 0
            if not m.any():
                continue
            color = np.array(colors[i], dtype=np.float32) / 255.0
            out[m] = 0.65 * out[m] + 0.35 * color[None]
        return (out.clip(0.0, 1.0) * 255.0).astype("uint8")

    @staticmethod
    def _compute_instance_similarity_from_resnet(
        resnet_nchw: torch.Tensor,
        masks_cam_nhw: torch.Tensor,
    ):
        # resnet_nchw: [Ncam, C, Hr, Wr], masks_cam_nhw: [Ncam, Ninst, Hm, Wm]
        if (not torch.is_tensor(resnet_nchw)) or (not torch.is_tensor(masks_cam_nhw)):
            return None, None
        if resnet_nchw.dim() != 4 or masks_cam_nhw.dim() != 4:
            return None, None
        n_cam = int(min(resnet_nchw.shape[0], masks_cam_nhw.shape[0]))
        n_inst = int(masks_cam_nhw.shape[1])
        if n_cam <= 0 or n_inst <= 0:
            return None, None

        c_dim = int(resnet_nchw.shape[1])
        feat_sum = torch.zeros((n_inst, c_dim), dtype=torch.float32, device=resnet_nchw.device)
        feat_count = torch.zeros((n_inst,), dtype=torch.float32, device=resnet_nchw.device)

        for cam in range(n_cam):
            feat = resnet_nchw[cam].to(torch.float32)  # [C, Hr, Wr]
            hr, wr = int(feat.shape[-2]), int(feat.shape[-1])
            masks = masks_cam_nhw[cam].to(torch.float32)  # [N, Hm, Wm]
            masks_up = F.interpolate(
                masks.unsqueeze(1),
                size=(hr, wr),
                mode="nearest",
            )[:, 0] > 0.5  # [N, Hr, Wr]
            masks_flat = masks_up.view(n_inst, -1).to(torch.float32)  # [N, P]
            denom = masks_flat.sum(dim=1, keepdim=True)  # [N,1]
            valid = denom[:, 0] > 0.5
            if not bool(valid.any().item()):
                continue
            feat_flat = feat.view(c_dim, -1).transpose(0, 1).contiguous()  # [P,C]
            pooled = (masks_flat @ feat_flat) / denom.clamp(min=1.0)  # [N,C]
            feat_sum[valid] += pooled[valid]
            feat_count[valid] += 1.0

        valid_inst = feat_count > 0.0
        if not bool(valid_inst.any().item()):
            return None, valid_inst
        feat_avg = torch.zeros_like(feat_sum)
        feat_avg[valid_inst] = feat_sum[valid_inst] / feat_count[valid_inst].unsqueeze(1).clamp(min=1.0)
        feat_norm = F.normalize(feat_avg, dim=1, eps=1e-6)
        sim = feat_norm @ feat_norm.t()  # [N,N]
        invalid_mask = (~valid_inst).to(sim.device)
        sim[invalid_mask, :] = 0.0
        sim[:, invalid_mask] = 0.0
        return sim, valid_inst

    @staticmethod
    def _render_similarity_matrix(
        sim_nn: torch.Tensor,
        valid_n: torch.Tensor,
        n_inst: int,
        selected_ids: List[int] = None,
    ):
        if sim_nn is None or valid_n is None:
            return None
        try:
            import numpy as np
            from PIL import Image, ImageDraw
        except Exception:
            return None

        n = int(n_inst)
        if n <= 0:
            return None
        sim = sim_nn.detach().to(torch.float32).cpu().clamp(-1.0, 1.0).numpy()
        valid = valid_n.detach().to(torch.bool).cpu().numpy()

        # -1 (blue) .. 0 (white) .. +1 (red)
        r = ((sim + 1.0) * 0.5 * 255.0)
        b = ((1.0 - sim) * 0.5 * 255.0)
        g = (255.0 - (abs(sim) * 160.0))
        heat = np.stack([r, g, b], axis=-1).clip(0.0, 255.0).astype(np.uint8)
        for i in range(n):
            if not bool(valid[i]):
                heat[i, :, :] = np.array([80, 80, 80], dtype=np.uint8)
                heat[:, i, :] = np.array([80, 80, 80], dtype=np.uint8)

        cell = max(12, min(28, int(360 / max(1, n))))
        mat_img = Image.fromarray(heat, mode="RGB").resize((n * cell, n * cell), resample=Image.NEAREST)
        label_ids = selected_ids if isinstance(selected_ids, (list, tuple)) else list(range(n))
        label_ids = [int(v) for v in label_ids[:n]]

        max_id_chars = max(2, max(len(str(v)) for v in label_ids)) if len(label_ids) > 0 else 2
        top_pad = 28
        bottom_pad = 20
        left_pad = max(48, 10 + 7 * max_id_chars)
        canvas = Image.new("RGB", (left_pad + mat_img.width, top_pad + mat_img.height + bottom_pad), color=(20, 20, 20))
        canvas.paste(mat_img, (left_pad, top_pad))
        draw = ImageDraw.Draw(canvas)
        valid_cnt = int(valid.sum())
        draw.text((2, 2), f"ResNet cosine sim (N={n}, valid={valid_cnt})", fill=(255, 255, 255))

        # Axis labels (row/column instance ids) when matrix is not too dense.
        if n <= 40:
            for i in range(n):
                y = int(top_pad + i * cell + 2)
                draw.text((2, y), f"id{label_ids[i]}", fill=(220, 220, 220))
            for j in range(n):
                x = int(left_pad + j * cell + 2)
                draw.text((x, top_pad - 13), str(label_ids[j]), fill=(220, 220, 220))

        # Explicit legend so row/column-to-id mapping remains readable.
        preview = ",".join(str(v) for v in label_ids[:30])
        if n > 30:
            preview += ",..."
        draw.text((2, top_pad + mat_img.height + 2), f"row/col ids: [{preview}]", fill=(200, 200, 200))

        # Overlay numeric values when cells are large enough to remain legible.
        if (cell >= 18) and (n <= 20):
            for i in range(n):
                for j in range(n):
                    if (not bool(valid[i])) or (not bool(valid[j])):
                        txt = "NA"
                    else:
                        txt = f"{float(sim[i, j]):.2f}"
                    px = int(left_pad + j * cell)
                    py = int(top_pad + i * cell)
                    # Choose text color from cell luminance for readability.
                    rr, gg, bb = [int(v) for v in heat[i, j].tolist()]
                    lum = (0.299 * rr) + (0.587 * gg) + (0.114 * bb)
                    fg = (0, 0, 0) if lum > 150 else (255, 255, 255)
                    draw.text((px + 2, py + 2), txt, fill=fg)
        return canvas

    def pool_gt_instance_context_features(
        self,
        segmentation_instance3d_txyz: torch.Tensor,
        future_egomotion,
        gt_inst_ids_n: torch.Tensor,
        query_match_inputs: dict,
        fallback_segmentation_instance3d_txyz: torch.Tensor = None,
    ):
        """
        Pool instance-wise features from context_seq using projected GT masks.

        Returns:
            gt_inst_img_feat_tnd: [T_hist, N_inst, Cctx]
            gt_inst_img_feat_valid_tn: [T_hist, N_inst]
            gt_inst_img_feat_ids_n: [N_inst]
        """
        if (not isinstance(query_match_inputs, dict)):
            return None, None, None

        context_seq_tnchw = query_match_inputs.get("context_seq_tnchw", None)
        if (not torch.is_tensor(context_seq_tnchw)) or context_seq_tnchw.dim() != 5:
            return None, None, None

        proj = self._project_gt_instances_to_cam_masks(
            segmentation_instance3d_txyz=segmentation_instance3d_txyz,
            future_egomotion=future_egomotion,
            gt_inst_ids_n=gt_inst_ids_n,
            query_match_inputs=query_match_inputs,
            fallback_segmentation_instance3d_txyz=fallback_segmentation_instance3d_txyz,
        )
        if not isinstance(proj, dict):
            return None, None, None
        cam_mask_tnnhw = proj.get("gt_inst_cam_mask_tnnhw", None)
        inst_ids_n = proj.get("gt_inst_ids_n", None)
        if (
            (not torch.is_tensor(cam_mask_tnnhw))
            or cam_mask_tnnhw.dim() != 5
            or (not torch.is_tensor(inst_ids_n))
        ):
            return None, None, None

        t_hist = min(int(context_seq_tnchw.shape[0]), int(cam_mask_tnnhw.shape[0]))
        n_inst = int(cam_mask_tnnhw.shape[1])
        n_cam = min(int(context_seq_tnchw.shape[1]), int(cam_mask_tnnhw.shape[2]))
        c_ctx = int(context_seq_tnchw.shape[2])
        out_feat = context_seq_tnchw.new_zeros((t_hist, n_inst, c_ctx), dtype=torch.float32)
        out_valid = torch.zeros((t_hist, n_inst), device=context_seq_tnchw.device, dtype=torch.bool)
        if t_hist <= 0 or n_inst <= 0 or n_cam <= 0:
            return out_feat, out_valid, inst_ids_n

        masks = cam_mask_tnnhw[:t_hist, :, :n_cam].to(device=context_seq_tnchw.device, dtype=torch.bool)
        feats = context_seq_tnchw[:t_hist, :n_cam].to(torch.float32)
        for t in range(t_hist):
            for n_idx in range(n_inst):
                pooled_cam = []
                for cam in range(n_cam):
                    m = masks[t, n_idx, cam]
                    if not bool(m.any().item()):
                        continue
                    feat_chw = feats[t, cam]
                    denom = m.to(feat_chw.dtype).sum().clamp(min=1.0)
                    pooled = (feat_chw * m.to(feat_chw.dtype).unsqueeze(0)).sum(dim=(1, 2)) / denom
                    if torch.isfinite(pooled).all():
                        pooled_cam.append(pooled)
                if len(pooled_cam) > 0:
                    out_feat[t, n_idx] = torch.stack(pooled_cam, dim=0).mean(dim=0)
                    out_valid[t, n_idx] = True
        return out_feat, out_valid, inst_ids_n.to(device=context_seq_tnchw.device, dtype=torch.long)

    def _project_gt_instances_to_cam_masks(
        self,
        segmentation_instance3d_txyz: torch.Tensor,
        future_egomotion,
        gt_inst_ids_n: torch.Tensor,
        query_match_inputs: dict,
        attn_t_idx_t: torch.Tensor = None,
        seg_t_idx_t: torch.Tensor = None,
        frame_idx_override: Optional[List[int]] = None,
        fallback_segmentation_instance3d_txyz: torch.Tensor = None,
        max_points_per_inst: int = None,
    ) -> dict:
        if (
            (not isinstance(query_match_inputs, dict))
        ):
            return None

        context_seq_tnchw = query_match_inputs.get("context_seq_tnchw", None)
        rots_tn33 = query_match_inputs.get("rots_tn33", None)
        trans_tn3 = query_match_inputs.get("trans_tn3", None)
        intrins_tn33 = query_match_inputs.get("intrins_tn33", None)
        post_rots_tn33 = query_match_inputs.get("post_rots_tn33", None)
        post_trans_tn3 = query_match_inputs.get("post_trans_tn3", None)
        img_h = int(query_match_inputs.get("img_h", 0))
        img_w = int(query_match_inputs.get("img_w", 0))
        frame_start = int(query_match_inputs.get("frame_start_idx", 0))
        frame_end = int(query_match_inputs.get("frame_end_idx_exclusive", 0))
        feat_h = int(query_match_inputs.get("feat_h", 0))
        feat_w = int(query_match_inputs.get("feat_w", 0))
        n_cam_cfg = int(query_match_inputs.get("n_cam", 0))
        has_context = torch.is_tensor(context_seq_tnchw) and context_seq_tnchw.dim() == 5
        if (
            (not torch.is_tensor(rots_tn33))
            or (not torch.is_tensor(trans_tn3))
            or (not torch.is_tensor(intrins_tn33))
            or (not torch.is_tensor(post_rots_tn33))
            or (not torch.is_tensor(post_trans_tn3))
            or img_h <= 1
            or img_w <= 1
        ):
            return None

        has_primary = torch.is_tensor(segmentation_instance3d_txyz) and segmentation_instance3d_txyz.dim() == 4
        has_fallback = torch.is_tensor(fallback_segmentation_instance3d_txyz) and fallback_segmentation_instance3d_txyz.dim() == 4
        if (not has_primary) and (not has_fallback):
            return None

        t_src = int(segmentation_instance3d_txyz.shape[0]) if has_primary else int(fallback_segmentation_instance3d_txyz.shape[0])
        if has_context:
            feat_h = int(context_seq_tnchw.shape[-2])
            feat_w = int(context_seq_tnchw.shape[-1])
            n_cam = int(context_seq_tnchw.shape[1])
            proj_device = context_seq_tnchw.device
            t_cam_full = int(
                min(
                    context_seq_tnchw.shape[0],
                    rots_tn33.shape[0],
                    trans_tn3.shape[0],
                    intrins_tn33.shape[0],
                    post_rots_tn33.shape[0],
                    post_trans_tn3.shape[0],
                )
            )
        else:
            n_cam = n_cam_cfg if n_cam_cfg > 0 else int(rots_tn33.shape[1])
            proj_device = rots_tn33.device
            t_cam_full = int(
                min(
                    rots_tn33.shape[0],
                    trans_tn3.shape[0],
                    intrins_tn33.shape[0],
                    post_rots_tn33.shape[0],
                    post_trans_tn3.shape[0],
                )
            )
        if feat_h <= 0 or feat_w <= 0 or n_cam <= 0:
            return None
        t_seg_full = int(t_src)
        t_hist_full = int(min(t_cam_full, t_seg_full))
        if t_cam_full <= 0 or t_seg_full <= 0 or t_hist_full <= 0:
            return None
        seg_hist = (
            segmentation_instance3d_txyz[:t_seg_full].to(device=proj_device, dtype=torch.long)
            if has_primary else None
        )
        seg_fb_hist = (
            fallback_segmentation_instance3d_txyz[:t_seg_full].to(device=proj_device, dtype=torch.long)
            if has_fallback else None
        )

        if torch.is_tensor(gt_inst_ids_n) and gt_inst_ids_n.numel() > 0:
            inst_ids_n = gt_inst_ids_n.to(device=proj_device, dtype=torch.long).contiguous()
        else:
            uniq_src = seg_hist if seg_hist is not None else seg_fb_hist
            uniq = torch.unique(uniq_src)
            keep = (uniq > 0) & (uniq != 255)
            inst_ids_n = torch.sort(uniq[keep]).values.to(torch.long)

        if torch.is_tensor(attn_t_idx_t) and attn_t_idx_t.numel() > 0:
            attn_idx = attn_t_idx_t.to(device=proj_device, dtype=torch.long).reshape(-1)
            keep = (attn_idx >= 0) & (attn_idx < t_cam_full)
            attn_idx = attn_idx[keep]
        else:
            attn_idx = torch.arange(t_hist_full, device=proj_device, dtype=torch.long)
        if torch.is_tensor(seg_t_idx_t) and seg_t_idx_t.numel() > 0:
            seg_idx = seg_t_idx_t.to(device=proj_device, dtype=torch.long).reshape(-1)
            pair_count = int(min(attn_idx.numel(), seg_idx.numel()))
            attn_idx = attn_idx[:pair_count]
            seg_idx = seg_idx[:pair_count]
            keep = (seg_idx >= 0) & (seg_idx < t_seg_full)
            attn_idx = attn_idx[keep]
            seg_idx = seg_idx[keep]
        else:
            seg_idx = attn_idx.clone()
            keep = (seg_idx >= 0) & (seg_idx < t_seg_full)
            attn_idx = attn_idx[keep]
            seg_idx = seg_idx[keep]
        t_out = int(min(attn_idx.numel(), seg_idx.numel()))

        n_inst = int(inst_ids_n.numel())

        def _empty_result():
            return {
                "gt_inst_cam_mask_tnnhw": torch.zeros(
                    (t_out, n_inst, n_cam, feat_h, feat_w),
                    device=proj_device,
                    dtype=torch.bool,
                ),
                "gt_inst_cam_valid_tnn": torch.zeros(
                    (t_out, n_inst, n_cam),
                    device=proj_device,
                    dtype=torch.bool,
                ),
                "gt_inst_ids_n": inst_ids_n,
                "attn_t_idx_t": attn_idx,
                "seg_t_idx_t": seg_idx,
            }

        if t_out <= 0:
            return _empty_result()
        if n_inst <= 0 or n_cam <= 0:
            return _empty_result()

        if not torch.is_tensor(future_egomotion):
            return _empty_result()
        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[0]
        ego = ego.to(torch.float32)

        frame_indices_hist = list(range(frame_start, max(frame_start, frame_end)))
        if len(frame_indices_hist) < t_cam_full:
            frame_indices_hist = list(range(int(self.time_receptive_field - t_cam_full), int(self.time_receptive_field)))
        frame_indices_hist = frame_indices_hist[:t_cam_full]

        frame_override = None
        if isinstance(frame_idx_override, (list, tuple)):
            frame_override = [int(v) for v in frame_idx_override]
        elif torch.is_tensor(frame_idx_override):
            frame_override = [int(v) for v in frame_idx_override.reshape(-1).tolist()]
        if isinstance(frame_override, list):
            if len(frame_override) == t_out:
                frame_indices_used = frame_override
            elif len(frame_override) >= t_cam_full:
                frame_indices_used = [int(frame_override[int(i)]) for i in attn_idx.tolist()]
            else:
                frame_indices_used = [int(frame_indices_hist[int(i)]) for i in attn_idx.tolist()]
        else:
            frame_indices_used = [int(frame_indices_hist[int(i)]) for i in attn_idx.tolist()]

        lidar_t_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=frame_indices_used,
            present_global_idx=int(self.time_receptive_field - 1),
        )
        if len(lidar_t_to_present) != t_out:
            return _empty_result()

        out = _empty_result()
        out_mask = out["gt_inst_cam_mask_tnnhw"]
        out_valid = out["gt_inst_cam_valid_tnn"]

        occ_shape_src = seg_hist if seg_hist is not None else seg_fb_hist
        x_size, y_size, z_size = [int(v) for v in occ_shape_src.shape[1:]]
        pc = torch.as_tensor(self.point_cloud_range, dtype=torch.float32, device=proj_device)
        voxel = torch.tensor(
            [
                (pc[3] - pc[0]) / float(x_size),
                (pc[4] - pc[1]) / float(y_size),
                (pc[5] - pc[2]) / float(z_size),
            ],
            dtype=torch.float32,
            device=proj_device,
        )
        pc_min = pc[:3]
        id_to_col = {int(iid.item()): idx for idx, iid in enumerate(inst_ids_n.to(torch.long))}
        max_pts = int(
            getattr(self, "query_gt_occ_inst_cam_proj_max_points_per_inst", 8192)
            if max_points_per_inst is None else max_points_per_inst
        )
        max_pts = max(1, max_pts)
        dilate_radius = int(getattr(self, "query_gt_occ_inst_cam_proj_mask_dilate", 1))
        dilate_radius = max(1, dilate_radius)
        dilate_kernel = (2 * dilate_radius) - 1

        def _bbox_fallback_mask(occ_t, tf_inv, iid, src_t, n_idx):
            pts = torch.nonzero(occ_t == int(iid), as_tuple=False)
            if pts.numel() <= 0:
                return
            mins = pts.min(dim=0).values.to(torch.float32)
            maxs = pts.max(dim=0).values.to(torch.float32) + 1.0
            wmin = pc_min + mins * voxel
            wmax = pc_min + maxs * voxel
            corners_present = torch.stack(
                [
                    torch.stack([wmin[0], wmin[1], wmin[2]]),
                    torch.stack([wmax[0], wmin[1], wmin[2]]),
                    torch.stack([wmax[0], wmax[1], wmin[2]]),
                    torch.stack([wmin[0], wmax[1], wmin[2]]),
                    torch.stack([wmin[0], wmin[1], wmax[2]]),
                    torch.stack([wmax[0], wmin[1], wmax[2]]),
                    torch.stack([wmax[0], wmax[1], wmax[2]]),
                    torch.stack([wmin[0], wmax[1], wmax[2]]),
                ],
                dim=0,
            )
            corners_h = torch.cat(
                [corners_present, torch.ones((8, 1), dtype=torch.float32, device=corners_present.device)],
                dim=1,
            )
            corners_t = (tf_inv @ corners_h.t()).t()[:, :3]
            for cam in range(n_cam):
                uv, _depth, valid = self._project_corners_to_aug_uv(
                    corners_lidar=corners_t,
                    rot=rots_tn33[src_t, cam],
                    trans=trans_tn3[src_t, cam],
                    intrins=intrins_tn33[src_t, cam],
                    post_rot=post_rots_tn33[src_t, cam],
                    post_trans=post_trans_tn3[src_t, cam],
                )
                if int(valid.sum().item()) < 2:
                    continue
                uv_valid = uv[valid]
                if int(uv_valid.shape[0]) >= 3:
                    hull = self._convex_hull(uv_valid)
                else:
                    hull = []
                if len(hull) < 3 and int(uv_valid.shape[0]) >= 2:
                    xmin = float(uv_valid[:, 0].min().item())
                    xmax = float(uv_valid[:, 0].max().item())
                    ymin = float(uv_valid[:, 1].min().item())
                    ymax = float(uv_valid[:, 1].max().item())
                    hull = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                if len(hull) < 3:
                    continue
                hull_feat = [
                    (
                        (float(x) / max(1.0, float(img_w - 1))) * float(feat_w - 1),
                        (float(y) / max(1.0, float(img_h - 1))) * float(feat_h - 1),
                    )
                    for x, y in hull
                ]
                mask = self._polygon_to_mask(hull_feat, feat_h, feat_w)
                if mask is None:
                    continue
                mask = mask.to(device=proj_device, dtype=torch.bool)
                if not bool(mask.any().item()):
                    continue
                out_mask[out_t, n_idx, cam] |= mask
                out_valid[out_t, n_idx, cam] = True

        for out_t in range(t_out):
            src_t = int(attn_idx[out_t].item())
            seg_t = int(seg_idx[out_t].item())
            occ_t = seg_hist[seg_t] if seg_hist is not None else None
            occ_fb_t = seg_fb_hist[seg_t] if seg_fb_hist is not None else None
            tf_inv = torch.inverse(lidar_t_to_present[out_t]).to(device=proj_device, dtype=torch.float32)
            for iid, n_idx in id_to_col.items():
                projected = False
                if occ_t is not None:
                    pts = torch.nonzero(occ_t == int(iid), as_tuple=False)
                    if pts.numel() > 0:
                        projected = True
                        if int(pts.shape[0]) > max_pts:
                            step = max(1, int((int(pts.shape[0]) + max_pts - 1) // max_pts))
                            pts = pts[::step]
                        pts_world = pc_min[None, :] + (pts.to(torch.float32) + 0.5) * voxel[None, :]
                        pts_h = torch.cat(
                            [pts_world, torch.ones((pts_world.shape[0], 1), dtype=torch.float32, device=pts_world.device)],
                            dim=1,
                        )
                        pts_t = (tf_inv @ pts_h.t()).t()[:, :3]
                        for cam in range(n_cam):
                            uv, _depth, valid = self._project_points_to_aug_uv(
                                points_lidar=pts_t,
                                rot=rots_tn33[src_t, cam],
                                trans=trans_tn3[src_t, cam],
                                intrins=intrins_tn33[src_t, cam],
                                post_rot=post_rots_tn33[src_t, cam],
                                post_trans=post_trans_tn3[src_t, cam],
                            )
                            valid = (
                                valid
                                & torch.isfinite(uv).all(dim=-1)
                                & (uv[:, 0] >= 0.0)
                                & (uv[:, 0] <= float(img_w - 1))
                                & (uv[:, 1] >= 0.0)
                                & (uv[:, 1] <= float(img_h - 1))
                            )
                            if not bool(valid.any().item()):
                                continue
                            uv = uv[valid]
                            fx = torch.round(
                                (uv[:, 0] / max(1.0, float(img_w - 1))) * float(feat_w - 1)
                            ).to(torch.long)
                            fy = torch.round(
                                (uv[:, 1] / max(1.0, float(img_h - 1))) * float(feat_h - 1)
                            ).to(torch.long)
                            keep = (
                                (fx >= 0) & (fx < int(feat_w))
                                & (fy >= 0) & (fy < int(feat_h))
                            )
                            if not bool(keep.any().item()):
                                continue
                            fx = fx[keep]
                            fy = fy[keep]
                            cur = out_mask[out_t, n_idx, cam]
                            cur[fy, fx] = True
                            if dilate_radius > 1:
                                cur = F.max_pool2d(
                                    cur.to(torch.float32)[None, None],
                                    kernel_size=dilate_kernel,
                                    stride=1,
                                    padding=dilate_radius - 1,
                                )[0, 0] > 0.5
                                out_mask[out_t, n_idx, cam] = cur
                            out_valid[out_t, n_idx, cam] = bool(out_mask[out_t, n_idx, cam].any().item())

                if (not projected) and (occ_fb_t is not None):
                    _bbox_fallback_mask(
                        occ_t=occ_fb_t,
                        tf_inv=tf_inv,
                        iid=iid,
                        src_t=src_t,
                        n_idx=n_idx,
                    )

        return out

    def build_query_attn_bbox_targets(
        self,
        segmentation_instance3d_txyz: torch.Tensor,
        future_egomotion,
        gt_inst_ids_n: torch.Tensor,
        query_match_inputs: dict,
        fallback_segmentation_instance3d_txyz: torch.Tensor = None,
    ) -> dict:
        """
        Build GT attention supervision targets on context feature grid.

        Returns dict with:
            gt_inst_mask_tnhw: [T, Ninst, H, W] bool
            gt_inst_valid_tn: [T, Ninst] bool
            gt_inst_ids_n: [Ninst] long
            union_mask_thw: [T, H, W] bool
            inverse_union_dist_thw: [T, H, W] float, each frame sums to 1
            gt_inst_cam_mask_tnnhw: [T, Ninst, Ncam, H, W] bool
            gt_inst_cam_valid_tnn: [T, Ninst, Ncam] bool
        """
        attn_t_idx_t = None
        if bool(getattr(self, "query_present_only", False)) and isinstance(query_match_inputs, dict):
            frame_start = int(query_match_inputs.get("frame_start_idx", 0))
            frame_end = int(query_match_inputs.get("frame_end_idx_exclusive", frame_start))
            rots_tn33 = query_match_inputs.get("rots_tn33", None)
            t_cam = int(rots_tn33.shape[0]) if torch.is_tensor(rots_tn33) and rots_tn33.dim() >= 1 else 0
            frame_indices = list(range(frame_start, max(frame_start, frame_end)))
            if len(frame_indices) < t_cam:
                frame_indices = list(range(int(self.time_receptive_field - t_cam), int(self.time_receptive_field)))
            frame_indices = frame_indices[:t_cam]
            present_global_idx = int(getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1)))
            present_local = None
            for idx, global_idx in enumerate(frame_indices):
                if int(global_idx) == int(present_global_idx):
                    present_local = int(idx)
                    break
            if present_local is None and t_cam > 0:
                present_local = max(0, min(t_cam - 1, int(present_global_idx - frame_start)))
            if present_local is not None:
                device = rots_tn33.device if torch.is_tensor(rots_tn33) else None
                attn_t_idx_t = torch.as_tensor([present_local], device=device, dtype=torch.long)
        gt_proj = self._project_gt_instances_to_cam_masks(
            segmentation_instance3d_txyz=segmentation_instance3d_txyz,
            future_egomotion=future_egomotion,
            gt_inst_ids_n=gt_inst_ids_n,
            query_match_inputs=query_match_inputs,
            attn_t_idx_t=attn_t_idx_t,
            fallback_segmentation_instance3d_txyz=fallback_segmentation_instance3d_txyz,
        )
        if not isinstance(gt_proj, dict):
            return None

        cam_mask_tnnhw = gt_proj.get("gt_inst_cam_mask_tnnhw", None)
        cam_valid_tnn = gt_proj.get("gt_inst_cam_valid_tnn", None)
        gt_ids_n = gt_proj.get("gt_inst_ids_n", None)
        if (
            (not torch.is_tensor(cam_mask_tnnhw))
            or cam_mask_tnnhw.dim() != 5
            or (not torch.is_tensor(cam_valid_tnn))
            or cam_valid_tnn.dim() != 3
            or (not torch.is_tensor(gt_ids_n))
        ):
            return None

        out_mask = cam_mask_tnnhw.any(dim=2)
        out_valid = cam_valid_tnn.any(dim=2)
        union_mask = out_mask.any(dim=1)
        t_hist = int(out_mask.shape[0])
        feat_h = int(out_mask.shape[-2])
        feat_w = int(out_mask.shape[-1])
        inverse_union_dist = torch.full(
            (t_hist, feat_h, feat_w),
            fill_value=(1.0 / float(max(1, feat_h * feat_w))),
            device=out_mask.device,
            dtype=torch.float32,
        )
        for t in range(t_hist):
            inv = ~union_mask[t]
            inv_f = inv.to(torch.float32)
            denom = inv_f.sum()
            if bool((denom > 0).item()):
                inverse_union_dist[t] = inv_f / denom.clamp(min=1.0)

        return {
            "gt_inst_mask_tnhw": out_mask,
            "gt_inst_valid_tn": out_valid,
            "gt_inst_ids_n": gt_ids_n,
            "union_mask_thw": union_mask,
            "inverse_union_dist_thw": inverse_union_dist,
            "gt_inst_cam_mask_tnnhw": cam_mask_tnnhw,
            "gt_inst_cam_valid_tnn": cam_valid_tnn,
            "attn_t_idx_t": gt_proj.get("attn_t_idx_t", None),
        }

    def _project_query_gaussians_to_cam_maps(
        self,
        mixture_centers_world_tqg3: torch.Tensor,
        mixture_sigmas_world_tqg3: torch.Tensor,
        mixture_yaw_tqg: torch.Tensor,
        mixture_weights_tqg: torch.Tensor,
        future_egomotion,
        query_match_inputs: dict,
        query_idx_subset: torch.Tensor = None,
        attn_t_idx_t: torch.Tensor = None,
        query_t_idx_t: torch.Tensor = None,
        chunk: int = 32,
    ) -> dict:
        del mixture_yaw_tqg  # Current CAM target uses axis-aligned projected sigma.

        if (
            (not isinstance(query_match_inputs, dict))
            or (not torch.is_tensor(mixture_centers_world_tqg3))
            or (not torch.is_tensor(mixture_sigmas_world_tqg3))
            or (not torch.is_tensor(mixture_weights_tqg))
            or mixture_centers_world_tqg3.dim() != 4
            or tuple(mixture_centers_world_tqg3.shape) != tuple(mixture_sigmas_world_tqg3.shape)
            or tuple(mixture_centers_world_tqg3.shape[:3]) != tuple(mixture_weights_tqg.shape)
        ):
            return None

        context_seq_tnchw = query_match_inputs.get("context_seq_tnchw", None)
        rots_tn33 = query_match_inputs.get("rots_tn33", None)
        trans_tn3 = query_match_inputs.get("trans_tn3", None)
        intrins_tn33 = query_match_inputs.get("intrins_tn33", None)
        post_rots_tn33 = query_match_inputs.get("post_rots_tn33", None)
        post_trans_tn3 = query_match_inputs.get("post_trans_tn3", None)
        img_h = int(query_match_inputs.get("img_h", 0))
        img_w = int(query_match_inputs.get("img_w", 0))
        frame_start = int(query_match_inputs.get("frame_start_idx", 0))
        frame_end = int(query_match_inputs.get("frame_end_idx_exclusive", 0))
        feat_h = int(query_match_inputs.get("feat_h", 0))
        feat_w = int(query_match_inputs.get("feat_w", 0))
        n_cam_cfg = int(query_match_inputs.get("n_cam", 0))
        has_context = torch.is_tensor(context_seq_tnchw) and context_seq_tnchw.dim() == 5
        if (
            (not torch.is_tensor(rots_tn33))
            or (not torch.is_tensor(trans_tn3))
            or (not torch.is_tensor(intrins_tn33))
            or (not torch.is_tensor(post_rots_tn33))
            or (not torch.is_tensor(post_trans_tn3))
            or img_h <= 1
            or img_w <= 1
        ):
            return None

        if has_context:
            feat_h = int(context_seq_tnchw.shape[-2])
            feat_w = int(context_seq_tnchw.shape[-1])
            n_cam = int(context_seq_tnchw.shape[1])
            proj_device = context_seq_tnchw.device
            t_hist = int(
                min(
                    context_seq_tnchw.shape[0],
                    rots_tn33.shape[0],
                    trans_tn3.shape[0],
                    intrins_tn33.shape[0],
                    post_rots_tn33.shape[0],
                    post_trans_tn3.shape[0],
                )
            )
        else:
            n_cam = n_cam_cfg if n_cam_cfg > 0 else int(rots_tn33.shape[1])
            proj_device = rots_tn33.device
            t_hist = int(
                min(
                    rots_tn33.shape[0],
                    trans_tn3.shape[0],
                    intrins_tn33.shape[0],
                    post_rots_tn33.shape[0],
                    post_trans_tn3.shape[0],
                )
            )
        t_query = int(mixture_centers_world_tqg3.shape[0])
        q_total = int(mixture_centers_world_tqg3.shape[1])
        g_count = int(mixture_centers_world_tqg3.shape[2])
        if feat_h <= 0 or feat_w <= 0 or n_cam <= 0:
            return None
        if t_hist <= 0 or t_query <= 0 or q_total <= 0 or g_count <= 0 or n_cam <= 0:
            return None

        if torch.is_tensor(query_idx_subset) and query_idx_subset.numel() > 0:
            query_idx_q = query_idx_subset.to(device=proj_device, dtype=torch.long).reshape(-1)
            q_keep = (query_idx_q >= 0) & (query_idx_q < q_total)
            query_idx_q = query_idx_q[q_keep]
        else:
            query_idx_q = torch.arange(q_total, device=proj_device, dtype=torch.long)
        q_count = int(query_idx_q.numel())
        if q_count <= 0:
            return None

        if torch.is_tensor(attn_t_idx_t) and torch.is_tensor(query_t_idx_t):
            attn_t_idx = attn_t_idx_t.to(device=proj_device, dtype=torch.long).reshape(-1)
            query_t_idx = query_t_idx_t.to(device=proj_device, dtype=torch.long).reshape(-1)
            t_out = min(int(attn_t_idx.numel()), int(query_t_idx.numel()))
            attn_t_idx = attn_t_idx[:t_out]
            query_t_idx = query_t_idx[:t_out]
        elif (attn_t_idx_t is None) and (query_t_idx_t is None):
            frame_mode = str(getattr(self, "query_attn_cam_frame_mode", "overlap_only")).lower()
            if frame_mode != "overlap_only":
                return None
            if bool(getattr(self, "query_present_only", False)):
                present_global_idx = int(getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1)))
                frame_indices_tmp = list(range(frame_start, max(frame_start, frame_end)))
                if len(frame_indices_tmp) < t_hist:
                    frame_indices_tmp = list(
                        range(int(self.time_receptive_field - t_hist), int(self.time_receptive_field))
                    )
                frame_indices_tmp = frame_indices_tmp[:t_hist]
                present_local = 0
                for idx, global_idx in enumerate(frame_indices_tmp):
                    if int(global_idx) == int(present_global_idx):
                        present_local = int(idx)
                        break
                present_local = max(0, min(t_hist - 1, int(present_local)))
                attn_t_idx = torch.tensor([present_local], device=proj_device, dtype=torch.long)
                query_t_idx = torch.tensor([0], device=proj_device, dtype=torch.long)
            elif t_hist >= 2:
                attn_t_idx = torch.tensor(
                    [t_hist - 2, t_hist - 1],
                    device=proj_device,
                    dtype=torch.long,
                )
            else:
                attn_t_idx = torch.tensor([t_hist - 1], device=proj_device, dtype=torch.long)
            query_t_idx = torch.arange(
                0,
                min(int(attn_t_idx.numel()), t_query),
                device=proj_device,
                dtype=torch.long,
            )
            attn_t_idx = attn_t_idx[: int(query_t_idx.numel())]
        else:
            return None

        t_keep = (
            (attn_t_idx >= 0) & (attn_t_idx < t_hist)
            & (query_t_idx >= 0) & (query_t_idx < t_query)
        )
        attn_t_idx = attn_t_idx[t_keep]
        query_t_idx = query_t_idx[t_keep]
        t_out = int(attn_t_idx.numel())
        if t_out <= 0:
            return None

        if not torch.is_tensor(future_egomotion):
            return None
        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[0]
        ego = ego.to(torch.float32)
        frame_indices = list(range(frame_start, max(frame_start, frame_end)))
        if len(frame_indices) < t_hist:
            frame_indices = list(range(int(self.time_receptive_field - t_hist), int(self.time_receptive_field)))
        frame_indices = frame_indices[:t_hist]
        frame_indices_used = [int(frame_indices[int(v)]) for v in attn_t_idx.tolist()]
        lidar_t_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=frame_indices_used,
            present_global_idx=int(self.time_receptive_field - 1),
        )
        if len(lidar_t_to_present) != t_out:
            return None

        grid_y, grid_x = torch.meshgrid(
            torch.arange(feat_h, device=proj_device, dtype=torch.float32),
            torch.arange(feat_w, device=proj_device, dtype=torch.float32),
            indexing="ij",
        )
        out_map = torch.zeros(
            (t_out, q_count, n_cam, feat_h, feat_w),
            device=proj_device,
            dtype=torch.float32,
        )
        out_valid_tqn = torch.zeros((t_out, q_count, n_cam), device=proj_device, dtype=torch.bool)
        proj_total_count = out_map.new_tensor(0.0)
        proj_valid_depth_count = out_map.new_tensor(0.0)
        proj_inbound_count = out_map.new_tensor(0.0)

        centers = mixture_centers_world_tqg3.to(device=proj_device, dtype=torch.float32)
        sigmas = mixture_sigmas_world_tqg3.to(device=proj_device, dtype=torch.float32)
        weights = mixture_weights_tqg.to(device=proj_device, dtype=torch.float32).clamp(min=0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        trunc_sigma = float(getattr(self, "query_attn_cam_gaussian_truncate_sigma", 3.0))
        chunk = max(1, int(chunk))

        for out_t in range(t_out):
            t_attn = int(attn_t_idx[out_t].item())
            t_query_idx = int(query_t_idx[out_t].item())
            tf_inv = torch.inverse(lidar_t_to_present[out_t]).to(
                device=proj_device,
                dtype=torch.float32,
            )

            center_qg = centers[t_query_idx].index_select(0, query_idx_q)  # [Qs,G,3]
            sigma_qg = sigmas[t_query_idx].index_select(0, query_idx_q)    # [Qs,G,3]
            weight_qg = weights[t_query_idx].index_select(0, query_idx_q)  # [Qs,G]

            center_flat = center_qg.reshape(-1, 3)
            sigma_flat = sigma_qg.reshape(-1, 3)
            weight_flat = weight_qg.reshape(-1)
            q_idx_flat = (
                torch.arange(q_count, device=proj_device, dtype=torch.long)
                .unsqueeze(1)
                .expand(q_count, g_count)
                .reshape(-1)
            )

            pts = torch.cat(
                [
                    center_flat,
                    center_flat + torch.stack(
                        [
                            sigma_flat[:, 0],
                            torch.zeros_like(sigma_flat[:, 0]),
                            torch.zeros_like(sigma_flat[:, 0]),
                        ],
                        dim=1,
                    ),
                    center_flat + torch.stack(
                        [
                            torch.zeros_like(sigma_flat[:, 1]),
                            sigma_flat[:, 1],
                            torch.zeros_like(sigma_flat[:, 1]),
                        ],
                        dim=1,
                    ),
                    center_flat + torch.stack(
                        [
                            torch.zeros_like(sigma_flat[:, 2]),
                            torch.zeros_like(sigma_flat[:, 2]),
                            sigma_flat[:, 2],
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            )  # [4*(QG),3]
            pts_h = torch.cat(
                [pts, torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)],
                dim=1,
            )
            pts_t = (tf_inv @ pts_h.t()).t()[:, :3]
            n_flat = int(center_flat.shape[0])
            p0 = pts_t[:n_flat]
            px = pts_t[n_flat : (2 * n_flat)]
            py = pts_t[(2 * n_flat) : (3 * n_flat)]
            pz = pts_t[(3 * n_flat) :]

            for cam in range(n_cam):
                proj_total_count += float(center_flat.shape[0])
                uv0, _d0, v0 = self._project_corners_to_aug_uv(
                    corners_lidar=p0,
                    rot=rots_tn33[t_attn, cam],
                    trans=trans_tn3[t_attn, cam],
                    intrins=intrins_tn33[t_attn, cam],
                    post_rot=post_rots_tn33[t_attn, cam],
                    post_trans=post_trans_tn3[t_attn, cam],
                )
                uvx, _dx, vx = self._project_corners_to_aug_uv(
                    corners_lidar=px,
                    rot=rots_tn33[t_attn, cam],
                    trans=trans_tn3[t_attn, cam],
                    intrins=intrins_tn33[t_attn, cam],
                    post_rot=post_rots_tn33[t_attn, cam],
                    post_trans=post_trans_tn3[t_attn, cam],
                )
                uvy, _dy, vy = self._project_corners_to_aug_uv(
                    corners_lidar=py,
                    rot=rots_tn33[t_attn, cam],
                    trans=trans_tn3[t_attn, cam],
                    intrins=intrins_tn33[t_attn, cam],
                    post_rot=post_rots_tn33[t_attn, cam],
                    post_trans=post_trans_tn3[t_attn, cam],
                )
                uvz, _dz, vz = self._project_corners_to_aug_uv(
                    corners_lidar=pz,
                    rot=rots_tn33[t_attn, cam],
                    trans=trans_tn3[t_attn, cam],
                    intrins=intrins_tn33[t_attn, cam],
                    post_rot=post_rots_tn33[t_attn, cam],
                    post_trans=post_trans_tn3[t_attn, cam],
                )
                valid = (
                    v0
                    & vx
                    & vy
                    & vz
                    & torch.isfinite(uv0).all(dim=-1)
                    & torch.isfinite(uvx).all(dim=-1)
                    & torch.isfinite(uvy).all(dim=-1)
                    & torch.isfinite(uvz).all(dim=-1)
                )
                proj_valid_depth_count += valid.to(torch.float32).sum()
                if not bool(valid.any().item()):
                    continue

                uv0 = uv0[valid]
                uvx = uvx[valid]
                uvy = uvy[valid]
                uvz = uvz[valid]
                q_idx = q_idx_flat[valid]
                w_comp = weight_flat[valid]

                uc = (uv0[:, 0] / max(1.0, float(img_w - 1))) * float(feat_w - 1)
                vc = (uv0[:, 1] / max(1.0, float(img_h - 1))) * float(feat_h - 1)
                su = (
                    torch.maximum(
                        (uvx[:, 0] - uv0[:, 0]).abs(),
                        (uvz[:, 0] - uv0[:, 0]).abs(),
                    )
                    / max(1.0, float(img_w - 1))
                ) * float(feat_w - 1)
                sv = (
                    torch.maximum(
                        (uvy[:, 1] - uv0[:, 1]).abs(),
                        (uvz[:, 1] - uv0[:, 1]).abs(),
                    )
                    / max(1.0, float(img_h - 1))
                ) * float(feat_h - 1)
                su = su.clamp(min=0.5)
                sv = sv.clamp(min=0.5)

                in_bound = (
                    (uc >= 0.0) & (uc <= float(feat_w - 1))
                    & (vc >= 0.0) & (vc <= float(feat_h - 1))
                    & torch.isfinite(su)
                    & torch.isfinite(sv)
                    & (w_comp > 0.0)
                )
                proj_inbound_count += in_bound.to(torch.float32).sum()
                if not bool(in_bound.any().item()):
                    continue
                uc = uc[in_bound]
                vc = vc[in_bound]
                su = su[in_bound]
                sv = sv[in_bound]
                q_idx = q_idx[in_bound]
                w_comp = w_comp[in_bound]

                tgt_qhw = out_map[out_t, :, cam]
                for st in range(0, int(uc.shape[0]), chunk):
                    ed = min(st + chunk, int(uc.shape[0]))
                    uc_c = uc[st:ed].view(-1, 1, 1)
                    vc_c = vc[st:ed].view(-1, 1, 1)
                    su_c = su[st:ed].view(-1, 1, 1)
                    sv_c = sv[st:ed].view(-1, 1, 1)
                    w_c = w_comp[st:ed].view(-1, 1, 1)
                    q_c = q_idx[st:ed]
                    dx = (grid_x.unsqueeze(0) - uc_c) / su_c
                    dy = (grid_y.unsqueeze(0) - vc_c) / sv_c
                    gmap = torch.exp(-0.5 * (dx.pow(2) + dy.pow(2)))
                    if trunc_sigma > 0.0:
                        in_gate = (dx.abs() <= trunc_sigma) & (dy.abs() <= trunc_sigma)
                        gmap = gmap * in_gate.to(gmap.dtype)
                    gmap = gmap * w_c
                    tgt_qhw.index_add_(0, q_c, gmap)

                q_unique = torch.unique(q_idx)
                out_valid_tqn[out_t, q_unique, cam] = True

        return {
            "gauss_proj_tqnhw": out_map,
            "gauss_valid_tqn": out_valid_tqn,
            "gauss_valid_tq": out_valid_tqn.any(dim=2),
            "attn_t_idx_t": attn_t_idx,
            "query_t_idx_t": query_t_idx,
            "query_idx_q": query_idx_q,
            "dbg_query_attn_cam_proj_total_count": proj_total_count,
            "dbg_query_attn_cam_proj_valid_depth_count": proj_valid_depth_count,
            "dbg_query_attn_cam_proj_inbound_count": proj_inbound_count,
            "dbg_query_attn_cam_valid_qcam_count": out_valid_tqn.to(torch.float32).sum(),
        }

    @torch.no_grad()
    def build_query_attn_cam_gaussian_targets(
        self,
        mixture_centers_world_tqg3: torch.Tensor,
        mixture_sigmas_world_tqg3: torch.Tensor,
        mixture_yaw_tqg: torch.Tensor,
        mixture_weights_tqg: torch.Tensor,
        future_egomotion,
        query_match_inputs: dict,
    ) -> dict:
        proj = self._project_query_gaussians_to_cam_maps(
            mixture_centers_world_tqg3=mixture_centers_world_tqg3,
            mixture_sigmas_world_tqg3=mixture_sigmas_world_tqg3,
            mixture_yaw_tqg=mixture_yaw_tqg,
            mixture_weights_tqg=mixture_weights_tqg,
            future_egomotion=future_egomotion,
            query_match_inputs=query_match_inputs,
        )
        if not isinstance(proj, dict):
            return None
        raw_map = proj.get("gauss_proj_tqnhw", None)
        if (not torch.is_tensor(raw_map)) or raw_map.dim() != 5:
            return None
        denom = raw_map.sum(dim=(-1, -2), keepdim=True)
        valid_prob = denom > 0.0
        prob_map = torch.where(valid_prob, raw_map / denom.clamp(min=1e-6), raw_map)
        out = dict(proj)
        out["gauss_target_tqnhw"] = prob_map
        return out

    @torch.no_grad()
    def maybe_save_query_cam_gaussian_vis(
        self,
        debug_bundle: dict,
        query_vis_bundle: dict,
        future_egomotion: torch.Tensor,
        step: int,
        gt_segmentation_instance3d: torch.Tensor = None,
    ) -> None:
        if not bool(getattr(self, "debug_query_cam_gaussian_vis_enabled", False)):
            return
        vis_every = int(getattr(self, "debug_query_cam_gaussian_vis_every", 0))
        if vis_every <= 0:
            return
        if (int(step) % vis_every) != 0:
            return
        if not self._is_main_process():
            return
        if (not isinstance(debug_bundle, dict)) or (not isinstance(query_vis_bundle, dict)):
            return
        if not torch.is_tensor(future_egomotion):
            return

        imgs = debug_bundle.get("imgs_seq_bt", None)
        rots = debug_bundle.get("rots_seq_bt", None)
        trans = debug_bundle.get("trans_seq_bt", None)
        intrins = debug_bundle.get("intrins_seq_bt", None)
        post_rots = debug_bundle.get("post_rots_seq_bt", None)
        post_trans = debug_bundle.get("post_trans_seq_bt", None)
        if any(v is None for v in (imgs, rots, trans, intrins, post_rots, post_trans)):
            return
        if imgs.dim() != 6:
            return
        if int(imgs.shape[0]) <= 0:
            return

        def _pick_bundle(prefix: str):
            c = query_vis_bundle.get(f"{prefix}_mixture_centers_tqg3", None)
            s = query_vis_bundle.get(f"{prefix}_mixture_sigmas_tqg3", None)
            w = query_vis_bundle.get(f"{prefix}_mixture_weights_tqg", None)
            q_idx = query_vis_bundle.get(f"{prefix}_query_idx_q", None)
            score_q = query_vis_bundle.get(f"{prefix}_score_q", None)
            if (
                torch.is_tensor(c)
                and torch.is_tensor(s)
                and torch.is_tensor(w)
                and c.dim() == 4
                and tuple(c.shape) == tuple(s.shape)
                and tuple(c.shape[:3]) == tuple(w.shape)
            ):
                return c, s, w, q_idx, score_q
            return None, None, None, None, None

        overlay_requested = bool(
            getattr(self, "debug_query_cam_gaussian_vis_gt_overlay_enabled", False)
        )
        prefix_order = ["matched", "selected", "candidate"] if overlay_requested else ["selected", "candidate", "matched"]
        picked_prefix = None
        mix_centers = None
        mix_sigmas = None
        mix_weights = None
        query_idx_q = None
        score_q = None
        for prefix in prefix_order:
            c, s, w, q_idx, sc = _pick_bundle(prefix)
            if c is not None:
                mix_centers = c
                mix_sigmas = s
                mix_weights = w
                query_idx_q = q_idx
                score_q = sc
                picked_prefix = prefix
                break
        if mix_centers is None:
            return

        b0 = 0
        t_all = int(min(imgs.shape[1], mix_centers.shape[0]))
        t_cap = int(min(t_all, int(getattr(self, "debug_query_cam_gaussian_vis_max_frames", 2))))
        if t_cap <= 0:
            return
        n_cam = int(imgs.shape[2])
        if n_cam <= 0:
            return

        mix_centers = mix_centers[-t_cap:].to(torch.float32)
        mix_sigmas = mix_sigmas[-t_cap:].to(torch.float32).clamp(min=1e-3)
        mix_weights = mix_weights[-t_cap:].to(torch.float32).clamp(min=0.0)
        mix_weights = mix_weights / mix_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        q_count = int(mix_centers.shape[1])
        g_count = int(mix_centers.shape[2])
        if q_count <= 0 or g_count <= 0:
            return

        if torch.is_tensor(query_idx_q) and int(query_idx_q.numel()) == q_count:
            query_idx_q = query_idx_q.to(dtype=torch.long, device=mix_centers.device).reshape(-1)
        else:
            query_idx_q = torch.arange(q_count, device=mix_centers.device, dtype=torch.long)

        if torch.is_tensor(score_q) and int(score_q.numel()) == q_count:
            score_q = score_q.to(device=mix_centers.device, dtype=torch.float32).reshape(-1)
        else:
            score_q = None

        vis_global_frame_indices = debug_bundle.get("vis_global_frame_indices", None)
        if isinstance(vis_global_frame_indices, (list, tuple)):
            vis_global_frame_indices = [int(v) for v in vis_global_frame_indices]
        else:
            vis_global_frame_indices = list(range(int(imgs.shape[1])))
        present_global_frame_idx = int(
            debug_bundle.get("present_global_frame_idx", int(self.time_receptive_field - 1))
        )
        if bool(getattr(self, "query_present_only", False)) and int(mix_centers.shape[0]) == 1:
            present_local = None
            for idx, global_idx in enumerate(vis_global_frame_indices):
                if int(global_idx) == int(present_global_frame_idx):
                    present_local = int(idx)
                    break
            if present_local is not None and 0 <= present_local < int(imgs.shape[1]):
                imgs = imgs[:, present_local:present_local + 1].contiguous()
                rots = rots[:, present_local:present_local + 1].contiguous()
                trans = trans[:, present_local:present_local + 1].contiguous()
                intrins = intrins[:, present_local:present_local + 1].contiguous()
                post_rots = post_rots[:, present_local:present_local + 1].contiguous()
                post_trans = post_trans[:, present_local:present_local + 1].contiguous()
                vis_global_frame_indices = [int(present_global_frame_idx)]

        matched_inst_idx_q = None
        matched_gt_ids_q = None
        if picked_prefix == "matched":
            inst_idx = query_vis_bundle.get("matched_inst_idx_q", None)
            gt_ids = query_vis_bundle.get("matched_gt_ids_q", None)
            if torch.is_tensor(inst_idx) and int(inst_idx.numel()) == q_count:
                matched_inst_idx_q = inst_idx.to(device=mix_centers.device, dtype=torch.long).reshape(-1)
            if torch.is_tensor(gt_ids) and int(gt_ids.numel()) == q_count:
                matched_gt_ids_q = gt_ids.to(device=mix_centers.device, dtype=torch.long).reshape(-1)

        topk_matched = int(getattr(self, "debug_query_cam_gaussian_vis_topk_matched", 0))
        if picked_prefix == "matched" and topk_matched > 0 and q_count > topk_matched:
            if score_q is not None:
                keep_idx = torch.topk(
                    score_q,
                    k=topk_matched,
                    largest=True,
                    sorted=True,
                ).indices
            else:
                keep_idx = torch.arange(topk_matched, device=mix_centers.device, dtype=torch.long)
            mix_centers = mix_centers.index_select(1, keep_idx)
            mix_sigmas = mix_sigmas.index_select(1, keep_idx)
            mix_weights = mix_weights.index_select(1, keep_idx)
            query_idx_q = query_idx_q.index_select(0, keep_idx)
            if score_q is not None:
                score_q = score_q.index_select(0, keep_idx)
            if torch.is_tensor(matched_inst_idx_q):
                matched_inst_idx_q = matched_inst_idx_q.index_select(0, keep_idx)
            if torch.is_tensor(matched_gt_ids_q):
                matched_gt_ids_q = matched_gt_ids_q.index_select(0, keep_idx)
            q_count = int(mix_centers.shape[1])

        q_keep = int(getattr(self, "debug_query_cam_gaussian_vis_max_queries", 50))
        q_keep = max(1, q_keep)
        if q_count > q_keep:
            if score_q is not None:
                keep_idx = torch.topk(
                    score_q,
                    k=q_keep,
                    largest=True,
                    sorted=True,
                ).indices
            else:
                keep_idx = torch.arange(q_keep, device=mix_centers.device, dtype=torch.long)
            mix_centers = mix_centers.index_select(1, keep_idx)
            mix_sigmas = mix_sigmas.index_select(1, keep_idx)
            mix_weights = mix_weights.index_select(1, keep_idx)
            query_idx_q = query_idx_q.index_select(0, keep_idx)
            if score_q is not None:
                score_q = score_q.index_select(0, keep_idx)
            if torch.is_tensor(matched_inst_idx_q):
                matched_inst_idx_q = matched_inst_idx_q.index_select(0, keep_idx)
            if torch.is_tensor(matched_gt_ids_q):
                matched_gt_ids_q = matched_gt_ids_q.index_select(0, keep_idx)
            q_count = int(mix_centers.shape[1])

        if len(vis_global_frame_indices) >= t_cap:
            vis_global_frame_indices = vis_global_frame_indices[-t_cap:]
        else:
            vis_global_frame_indices = list(range(t_cap))

        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[b0]
        ego = ego.to(torch.float32)
        lidar_t_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=vis_global_frame_indices,
            present_global_idx=present_global_frame_idx,
        )
        if len(lidar_t_to_present) != t_cap:
            return

        try:
            from PIL import Image, ImageDraw
        except Exception:
            return

        vis_dir = str(
            getattr(self, "debug_query_cam_gaussian_vis_dir", "./work_dirs/query_cam_gaussian_vis")
        )
        os.makedirs(vis_dir, exist_ok=True)

        trunc_sigma = float(getattr(self, "query_attn_cam_gaussian_truncate_sigma", 3.0))
        trunc_sigma = max(1.0, trunc_sigma)
        img_h = int(imgs.shape[-2])
        img_w = int(imgs.shape[-1])

        q_local_flat = (
            torch.arange(q_count, device=mix_centers.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(q_count, g_count)
            .reshape(-1)
        )
        g_local_flat = (
            torch.arange(g_count, device=mix_centers.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(q_count, g_count)
            .reshape(-1)
        )
        q_id_flat = query_idx_q[q_local_flat]
        draw_gt_overlay = (
            overlay_requested
            and picked_prefix == "matched"
            and torch.is_tensor(matched_gt_ids_q)
            and int(matched_gt_ids_q.numel()) == q_count
            and torch.is_tensor(gt_segmentation_instance3d)
            and gt_segmentation_instance3d.dim() == 4
            and int(gt_segmentation_instance3d.shape[0]) >= t_cap
        )
        gt_id_flat = None
        frame_gt_points: List[Dict[int, torch.Tensor]] = []
        if draw_gt_overlay:
            matched_gt_ids_q = matched_gt_ids_q.to(device=mix_centers.device, dtype=torch.long).reshape(-1)
            valid_gt_q = matched_gt_ids_q > 0
            if not bool(valid_gt_q.any().item()):
                draw_gt_overlay = False
            else:
                gt_id_flat = matched_gt_ids_q[q_local_flat]
                gt_ids_unique = torch.unique(matched_gt_ids_q[valid_gt_q]).to(torch.long)
                seg_vis_txyz = gt_segmentation_instance3d[-t_cap:].to(
                    device=mix_centers.device, dtype=torch.long
                )
                x_size, y_size, z_size = [int(v) for v in seg_vis_txyz.shape[1:]]
                pc = torch.as_tensor(
                    self.point_cloud_range, dtype=torch.float32, device=seg_vis_txyz.device
                )
                voxel = torch.tensor(
                    [
                        (pc[3] - pc[0]) / float(x_size),
                        (pc[4] - pc[1]) / float(y_size),
                        (pc[5] - pc[2]) / float(z_size),
                    ],
                    dtype=torch.float32,
                    device=seg_vis_txyz.device,
                )
                pc_min = pc[:3]
                gt_ids_list = [int(v) for v in gt_ids_unique.tolist()]
                max_gt_vis_pts = int(
                    getattr(
                        self,
                        "query_gt_occ_inst_cam_vis_max_points_per_inst",
                        getattr(self, "query_gt_occ_inst_cam_proj_max_points_per_inst", 8192),
                    )
                )
                max_gt_vis_pts = max(1, max_gt_vis_pts)
                for t_idx in range(t_cap):
                    occ_t = seg_vis_txyz[t_idx]
                    cur_points = {}
                    for gt_id in gt_ids_list:
                        pts_occ = torch.nonzero(occ_t == int(gt_id), as_tuple=False)
                        if pts_occ.numel() <= 0:
                            continue
                        if int(pts_occ.shape[0]) > max_gt_vis_pts:
                            step = max(1, int((int(pts_occ.shape[0]) + max_gt_vis_pts - 1) // max_gt_vis_pts))
                            pts_occ = pts_occ[::step]
                        pts_world = pc_min[None, :] + (pts_occ.to(torch.float32) + 0.5) * voxel[None, :]
                        cur_points[int(gt_id)] = pts_world
                    frame_gt_points.append(cur_points)

        for t in range(t_cap):
            tf_inv = torch.inverse(lidar_t_to_present[t]).to(device=mix_centers.device, dtype=torch.float32)
            center_qg = mix_centers[t]   # [Q,G,3]
            sigma_qg = mix_sigmas[t]     # [Q,G,3]
            weight_qg = mix_weights[t]   # [Q,G]
            max_comp_q = torch.argmax(weight_qg, dim=-1)  # [Q]

            center_flat = center_qg.reshape(-1, 3)
            sigma_flat = sigma_qg.reshape(-1, 3)
            weight_flat = weight_qg.reshape(-1)

            pts = torch.cat(
                [
                    center_flat,
                    center_flat + torch.stack(
                        [
                            sigma_flat[:, 0],
                            torch.zeros_like(sigma_flat[:, 0]),
                            torch.zeros_like(sigma_flat[:, 0]),
                        ],
                        dim=1,
                    ),
                    center_flat + torch.stack(
                        [
                            torch.zeros_like(sigma_flat[:, 1]),
                            sigma_flat[:, 1],
                            torch.zeros_like(sigma_flat[:, 1]),
                        ],
                        dim=1,
                    ),
                    center_flat + torch.stack(
                        [
                            torch.zeros_like(sigma_flat[:, 2]),
                            torch.zeros_like(sigma_flat[:, 2]),
                            sigma_flat[:, 2],
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            )  # [4*(QG),3]
            pts_h = torch.cat(
                [pts, torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)],
                dim=1,
            )
            pts_t = (tf_inv @ pts_h.t()).t()[:, :3]
            n_flat = int(center_flat.shape[0])
            p0 = pts_t[:n_flat]
            px = pts_t[n_flat : (2 * n_flat)]
            py = pts_t[(2 * n_flat) : (3 * n_flat)]
            pz = pts_t[(3 * n_flat) :]

            cell_imgs = []
            qdraw_count = 0
            gdraw_count = 0
            gt_draw_count = 0
            gt_points_t = {}
            if draw_gt_overlay and t < len(frame_gt_points):
                for gt_id, pts_present in frame_gt_points[t].items():
                    pts_h = torch.cat(
                        [
                            pts_present.to(torch.float32),
                            torch.ones((int(pts_present.shape[0]), 1), dtype=torch.float32, device=pts_present.device),
                        ],
                        dim=1,
                    )
                    gt_points_t[int(gt_id)] = (tf_inv @ pts_h.t()).t()[:, :3]
            for cam in range(n_cam):
                base_u8 = self._to_display_u8(imgs[b0, -t_cap + t, cam])
                raw_img = Image.fromarray(base_u8.copy(), mode="RGB")
                draw = ImageDraw.Draw(raw_img, mode="RGBA")

                uv0, _d0, v0 = self._project_corners_to_aug_uv(
                    corners_lidar=p0,
                    rot=rots[b0, -t_cap + t, cam],
                    trans=trans[b0, -t_cap + t, cam],
                    intrins=intrins[b0, -t_cap + t, cam],
                    post_rot=post_rots[b0, -t_cap + t, cam],
                    post_trans=post_trans[b0, -t_cap + t, cam],
                )
                uvx, _dx, vx = self._project_corners_to_aug_uv(
                    corners_lidar=px,
                    rot=rots[b0, -t_cap + t, cam],
                    trans=trans[b0, -t_cap + t, cam],
                    intrins=intrins[b0, -t_cap + t, cam],
                    post_rot=post_rots[b0, -t_cap + t, cam],
                    post_trans=post_trans[b0, -t_cap + t, cam],
                )
                uvy, _dy, vy = self._project_corners_to_aug_uv(
                    corners_lidar=py,
                    rot=rots[b0, -t_cap + t, cam],
                    trans=trans[b0, -t_cap + t, cam],
                    intrins=intrins[b0, -t_cap + t, cam],
                    post_rot=post_rots[b0, -t_cap + t, cam],
                    post_trans=post_trans[b0, -t_cap + t, cam],
                )
                uvz, _dz, vz = self._project_corners_to_aug_uv(
                    corners_lidar=pz,
                    rot=rots[b0, -t_cap + t, cam],
                    trans=trans[b0, -t_cap + t, cam],
                    intrins=intrins[b0, -t_cap + t, cam],
                    post_rot=post_rots[b0, -t_cap + t, cam],
                    post_trans=post_trans[b0, -t_cap + t, cam],
                )
                valid = (
                    v0
                    & vx
                    & vy
                    & vz
                    & torch.isfinite(uv0).all(dim=-1)
                    & torch.isfinite(uvx).all(dim=-1)
                    & torch.isfinite(uvy).all(dim=-1)
                    & torch.isfinite(uvz).all(dim=-1)
                    & (weight_flat > 0.0)
                )
                if bool(valid.any().item()):
                    uv0 = uv0[valid]
                    uvx = uvx[valid]
                    uvy = uvy[valid]
                    uvz = uvz[valid]
                    wv = weight_flat[valid]
                    qloc = q_local_flat[valid]
                    gloc = g_local_flat[valid]
                    qids = q_id_flat[valid]

                    uc = uv0[:, 0]
                    vc = uv0[:, 1]
                    su = torch.maximum(
                        (uvx[:, 0] - uv0[:, 0]).abs(),
                        (uvz[:, 0] - uv0[:, 0]).abs(),
                    ).clamp(min=1.0)
                    sv = torch.maximum(
                        (uvy[:, 1] - uv0[:, 1]).abs(),
                        (uvz[:, 1] - uv0[:, 1]).abs(),
                    ).clamp(min=1.0)
                    in_bound = (
                        (uc >= 0.0)
                        & (uc <= float(img_w - 1))
                        & (vc >= 0.0)
                        & (vc <= float(img_h - 1))
                        & torch.isfinite(su)
                        & torch.isfinite(sv)
                    )
                    if bool(in_bound.any().item()):
                        uc = uc[in_bound]
                        vc = vc[in_bound]
                        su = su[in_bound]
                        sv = sv[in_bound]
                        wv = wv[in_bound]
                        qloc = qloc[in_bound]
                        gloc = gloc[in_bound]
                        qids = qids[in_bound]
                        gt_ids_vis = None
                        if draw_gt_overlay and (gt_id_flat is not None):
                            gt_ids_vis = gt_id_flat[valid][in_bound]

                        labeled_q = set()
                        for i in range(int(uc.numel())):
                            qid = int(qids[i].item())
                            ql = int(qloc[i].item())
                            gl = int(gloc[i].item())
                            gt_id = int(gt_ids_vis[i].item()) if torch.is_tensor(gt_ids_vis) else -1
                            if gl == int(max_comp_q[ql].item()):
                                qdraw_count += 1
                            gdraw_count += 1
                            color = self._instance_color(qid)
                            alpha_fill = int(max(24.0, min(110.0, 22.0 + (140.0 * float(wv[i].item())))))
                            rx = float(su[i].item() * trunc_sigma)
                            ry = float(sv[i].item() * trunc_sigma)
                            cx = float(uc[i].item())
                            cy = float(vc[i].item())
                            draw.ellipse(
                                [cx - rx, cy - ry, cx + rx, cy + ry],
                                outline=(color[0], color[1], color[2], 220),
                                fill=(color[0], color[1], color[2], alpha_fill),
                                width=2,
                            )
                            draw.ellipse(
                                [cx - 2.0, cy - 2.0, cx + 2.0, cy + 2.0],
                                fill=(255, 255, 255, 220),
                            )
                            if (gl == int(max_comp_q[ql].item())) and (qid not in labeled_q):
                                labeled_q.add(qid)
                                lx = int(max(2, min(img_w - 40, round(cx + 3.0))))
                                ly = int(max(2, min(img_h - 12, round(cy - 3.0))))
                                draw.rectangle([lx - 1, ly - 1, lx + 34, ly + 10], fill=(0, 0, 0, 180))
                                if gt_id > 0:
                                    draw.text(
                                        (lx, ly),
                                        f"q={qid} g={gt_id}",
                                        fill=(color[0], color[1], color[2], 255),
                                    )
                                else:
                                    draw.text((lx, ly), f"q={qid}", fill=(color[0], color[1], color[2], 255))

                if draw_gt_overlay and len(gt_points_t) > 0:
                    for gt_id, pts_t in gt_points_t.items():
                        uv_gt, _depth_gt, valid_gt = self._project_points_to_aug_uv(
                            points_lidar=pts_t,
                            rot=rots[b0, -t_cap + t, cam],
                            trans=trans[b0, -t_cap + t, cam],
                            intrins=intrins[b0, -t_cap + t, cam],
                            post_rot=post_rots[b0, -t_cap + t, cam],
                            post_trans=post_trans[b0, -t_cap + t, cam],
                        )
                        valid_gt = (
                            valid_gt
                            & torch.isfinite(uv_gt).all(dim=-1)
                            & (uv_gt[:, 0] >= 0.0)
                            & (uv_gt[:, 0] <= float(img_w - 1))
                            & (uv_gt[:, 1] >= 0.0)
                            & (uv_gt[:, 1] <= float(img_h - 1))
                        )
                        if int(valid_gt.sum().item()) <= 0:
                            continue
                        uv_valid = uv_gt[valid_gt]
                        color_gt = self._instance_color(int(gt_id))
                        if int(uv_valid.shape[0]) >= 3:
                            hull = self._convex_hull(uv_valid)
                            if len(hull) >= 3:
                                draw.polygon(
                                    hull,
                                    fill=(color_gt[0], color_gt[1], color_gt[2], 36),
                                    outline=(color_gt[0], color_gt[1], color_gt[2], 180),
                                )
                        draw_step = max(1, int((int(uv_valid.shape[0]) + 255) // 256))
                        uv_draw = uv_valid[::draw_step]
                        for pt_idx in range(int(uv_draw.shape[0])):
                            px_pt = float(uv_draw[pt_idx, 0].item())
                            py_pt = float(uv_draw[pt_idx, 1].item())
                            draw.ellipse(
                                [px_pt - 1.0, py_pt - 1.0, px_pt + 1.0, py_pt + 1.0],
                                fill=(color_gt[0], color_gt[1], color_gt[2], 180),
                            )
                        gt_draw_count += 1

                cell_imgs.append(raw_img)

            target_w = 320
            target_h = max(64, int(round(float(img_h) * float(target_w) / float(max(1, img_w)))))
            gap = 4
            pad_top = 28
            canvas_w = n_cam * target_w + (n_cam - 1) * gap
            canvas_h = pad_top + target_h
            canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (4, 4),
                (
                    f"iter={int(step)} t_local={int(t)} t_global={int(vis_global_frame_indices[t])} "
                    f"bundle={picked_prefix} query={q_count} mix={g_count} "
                    f"draw_q={int(qdraw_count)} draw_g={int(gdraw_count)} gt_draw={int(gt_draw_count)}"
                ),
                fill=(255, 255, 255),
            )
            for cam in range(n_cam):
                x0 = cam * (target_w + gap)
                cell = cell_imgs[cam].resize((target_w, target_h), resample=Image.BILINEAR)
                canvas.paste(cell, (x0, pad_top))
                draw.text((x0 + 2, pad_top + 2), f"cam={cam}", fill=(255, 255, 255))

            out_path = os.path.join(
                vis_dir, f"iter_{int(step):06d}_t{int(t):02d}_query_cam_gaussian.png"
            )
            canvas.save(out_path)

    @torch.no_grad()
    def maybe_save_instance_img_debug(
        self,
        debug_bundle: dict,
        segmentation_instance3d: torch.Tensor,
        segmentation_cls_instance3d: torch.Tensor,
        future_egomotion: torch.Tensor,
        step: int,
    ) -> None:
        vis_every = int(getattr(self, "debug_instance_img_vis_every", 0))
        if vis_every <= 0:
            return
        if (int(step) % vis_every) != 0:
            return
        if not self._is_main_process():
            return
        if debug_bundle is None:
            return
        if segmentation_instance3d is None or future_egomotion is None:
            return

        imgs = debug_bundle.get("imgs_seq_bt", None)
        rots = debug_bundle.get("rots_seq_bt", None)
        trans = debug_bundle.get("trans_seq_bt", None)
        intrins = debug_bundle.get("intrins_seq_bt", None)
        post_rots = debug_bundle.get("post_rots_seq_bt", None)
        post_trans = debug_bundle.get("post_trans_seq_bt", None)
        context = debug_bundle.get("context_seq_btnchw", None)
        resnet_last = debug_bundle.get("resnet_last_seq_btnchw", None)
        if any(v is None for v in (imgs, rots, trans, intrins, post_rots, post_trans, context)):
            return

        if imgs.dim() != 6 or context.dim() != 6:
            return
        if imgs.shape[0] <= 0 or context.shape[0] <= 0:
            return

        b0 = 0
        vis_global_frame_indices = debug_bundle.get("vis_global_frame_indices", None)
        present_global_frame_idx = int(debug_bundle.get("present_global_frame_idx", int(self.time_receptive_field - 1)))
        t_cap = int(min(
            imgs.shape[1],
            context.shape[1],
            segmentation_instance3d.shape[0],
            int(getattr(self, "debug_instance_img_vis_max_frames", 3)),
            int(self.n_future_frames_plus),
        ))
        if t_cap <= 0:
            return
        n_cam = int(min(imgs.shape[2], context.shape[2]))
        if n_cam <= 0:
            return

        if isinstance(vis_global_frame_indices, (list, tuple)):
            vis_global_frame_indices = [int(v) for v in vis_global_frame_indices]
        else:
            vis_global_frame_indices = list(range(int(imgs.shape[1])))
        if len(vis_global_frame_indices) >= t_cap:
            vis_global_frame_indices = vis_global_frame_indices[-t_cap:]
        else:
            vis_global_frame_indices = list(range(t_cap))

        seg_instance_vis = segmentation_instance3d[-t_cap:].contiguous()
        seg_cls_vis = None
        if torch.is_tensor(segmentation_cls_instance3d):
            seg_cls_vis = segmentation_cls_instance3d[-t_cap:].contiguous()

        frame_boxes, selected_ids = self._build_boxes_from_instance_occ(
            seg_instance_txyz=seg_instance_vis,
            max_instances=int(getattr(self, "debug_instance_img_vis_max_instances", 24)),
        )
        if len(selected_ids) <= 0:
            return
        cls_map = self._build_class_map(
            seg_cls_vis,
            selected_ids=selected_ids,
        )
        colors = [self._instance_color(iid, cls_map.get(iid, None)) for iid in selected_ids]
        id_to_col = {iid: idx for idx, iid in enumerate(selected_ids)}

        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[b0]
        ego = ego.to(torch.float32)
        lidar_t_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=vis_global_frame_indices,
            present_global_idx=present_global_frame_idx,
        )
        if len(lidar_t_to_present) != t_cap:
            return

        try:
            from PIL import Image, ImageDraw
            import numpy as np
        except Exception:
            return

        vis_dir = str(getattr(self, "debug_instance_img_vis_dir", "./work_dirs/instance_img_debug_vis"))
        os.makedirs(vis_dir, exist_ok=True)

        for t in range(t_cap):
            raw_cells = []
            feat_cells = []
            pooled_valid = 0
            pooled_total = 0

            img_h = int(imgs.shape[-2])
            img_w = int(imgs.shape[-1])
            feat_h = int(context.shape[-2])
            feat_w = int(context.shape[-1])

            mask_device = context.device
            masks_by_cam = torch.zeros(
                (n_cam, len(selected_ids), feat_h, feat_w),
                dtype=torch.bool,
                device=mask_device,
            )
            cam_label_points = [[] for _ in range(n_cam)]
            corners_t_to_lidar = {}
            t_tf_inv = torch.inverse(lidar_t_to_present[t])
            for iid, corners_present in frame_boxes[t].items():
                corners_h = torch.cat(
                    [corners_present.to(torch.float32), torch.ones((8, 1), device=corners_present.device)],
                    dim=1,
                )
                corners_t = (t_tf_inv @ corners_h.t()).t()[:, :3]
                corners_t_to_lidar[iid] = corners_t

            for cam in range(n_cam):
                base_u8 = self._to_display_u8(imgs[b0, t, cam])
                raw_img = Image.fromarray(base_u8.copy(), mode="RGB")
                draw = ImageDraw.Draw(raw_img, mode="RGBA")

                for iid, corners_t in corners_t_to_lidar.items():
                    uv, _depth, valid = self._project_corners_to_aug_uv(
                        corners_lidar=corners_t,
                        rot=rots[b0, t, cam],
                        trans=trans[b0, t, cam],
                        intrins=intrins[b0, t, cam],
                        post_rot=post_rots[b0, t, cam],
                        post_trans=post_trans[b0, t, cam],
                    )
                    if int(valid.sum().item()) < 2:
                        continue

                    color = colors[id_to_col[iid]]
                    for s, e in self._BOX_EDGES:
                        if bool(valid[s].item()) and bool(valid[e].item()):
                            p1 = (float(uv[s, 0].item()), float(uv[s, 1].item()))
                            p2 = (float(uv[e, 0].item()), float(uv[e, 1].item()))
                            draw.line([p1, p2], fill=(color[0], color[1], color[2], 255), width=2)

                    uv_valid = uv[valid]
                    if int(uv_valid.shape[0]) >= 3:
                        hull = self._convex_hull(uv_valid)
                    else:
                        hull = []
                    if len(hull) < 3:
                        if int(uv_valid.shape[0]) >= 2:
                            xmin = float(uv_valid[:, 0].min().item())
                            xmax = float(uv_valid[:, 0].max().item())
                            ymin = float(uv_valid[:, 1].min().item())
                            ymax = float(uv_valid[:, 1].max().item())
                            hull = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                        else:
                            continue

                    draw.polygon(hull, fill=(color[0], color[1], color[2], 60))
                    cx = float(sum(p[0] for p in hull) / max(1, len(hull)))
                    cy = float(sum(p[1] for p in hull) / max(1, len(hull)))
                    cam_label_points[cam].append((cx, cy, int(iid), color))

                    hull_feat = [
                        (
                            (float(x) / max(1.0, float(img_w - 1))) * float(feat_w - 1),
                            (float(y) / max(1.0, float(img_h - 1))) * float(feat_h - 1),
                        )
                        for x, y in hull
                    ]
                    mask = self._polygon_to_mask(hull_feat, feat_h, feat_w)
                    if mask is None:
                        continue
                    masks_by_cam[cam, id_to_col[iid]] |= mask.to(device=mask_device)

                raw_cells.append(raw_img)

            for cam in range(n_cam):
                base_u8 = self._to_display_u8(imgs[b0, t, cam])
                masks_nhw = masks_by_cam[cam].to(torch.float32)
                masks_up = F.interpolate(
                    masks_nhw.unsqueeze(1),
                    size=(img_h, img_w),
                    mode="nearest",
                )[:, 0] > 0.5
                feat_overlay_u8 = self._feature_overlay(base_u8, masks_up, colors)
                feat_img = Image.fromarray(feat_overlay_u8, mode="RGB")
                feat_cells.append(feat_img)

                feat_map = context[b0, t, cam].detach().to(torch.float32)
                for n_idx in range(int(masks_nhw.shape[0])):
                    pooled_total += 1
                    m = masks_nhw[n_idx] > 0.5
                    if not bool(m.any().item()):
                        continue
                    denom = m.to(dtype=feat_map.dtype, device=feat_map.device).sum().clamp(min=1.0)
                    pooled = (feat_map * m.to(feat_map.dtype).unsqueeze(0)).sum(dim=(1, 2)) / denom
                    if torch.isfinite(pooled).all():
                        pooled_valid += 1

            sim_img = None
            if torch.is_tensor(resnet_last) and resnet_last.dim() == 6 and t < int(resnet_last.shape[1]):
                sim_nn, valid_n = self._compute_instance_similarity_from_resnet(
                    resnet_nchw=resnet_last[b0, t],
                    masks_cam_nhw=masks_by_cam,
                )
                sim_img = self._render_similarity_matrix(
                    sim_nn=sim_nn,
                    valid_n=valid_n,
                    n_inst=len(selected_ids),
                    selected_ids=selected_ids,
                )

            target_w = 320
            target_h = max(64, int(round(float(img_h) * float(target_w) / float(max(1, img_w)))))
            gap = 4
            pad_top = 28
            canvas_w = n_cam * target_w + (n_cam - 1) * gap
            sim_resized = None
            extra_h = 0
            if sim_img is not None:
                max_sim_w = max(128, canvas_w - 8)
                if sim_img.width > max_sim_w:
                    sim_h = max(96, int(round(float(sim_img.height) * float(max_sim_w) / float(max(1, sim_img.width)))))
                    sim_resized = sim_img.resize((max_sim_w, sim_h), resample=Image.BILINEAR)
                else:
                    sim_resized = sim_img
                extra_h = int(sim_resized.height) + gap
            canvas_h = pad_top + 2 * target_h + gap + extra_h
            canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (4, 4),
                (
                    f"iter={int(step)} t_local={int(t)} t_global={int(vis_global_frame_indices[t])} "
                    f"inst={len(selected_ids)} cams={n_cam} "
                    f"pooled_valid={pooled_valid}/{pooled_total}"
                ),
                fill=(255, 255, 255),
            )
            for cam in range(n_cam):
                x0 = cam * (target_w + gap)
                raw = raw_cells[cam].resize((target_w, target_h), resample=Image.BILINEAR)
                feat = feat_cells[cam].resize((target_w, target_h), resample=Image.BILINEAR)
                canvas.paste(raw, (x0, pad_top))
                canvas.paste(feat, (x0, pad_top + target_h + gap))
                draw.text((x0 + 2, pad_top + 2), f"cam={cam}", fill=(255, 255, 255))
                draw.text((x0 + 2, pad_top + target_h + gap + 2), "feature-mask overlay", fill=(255, 255, 255))
                sx = float(target_w) / float(max(1, img_w))
                sy = float(target_h) / float(max(1, img_h))
                for cx, cy, iid, color in cam_label_points[cam]:
                    lx = int(x0 + cx * sx)
                    ly = int(pad_top + cy * sy)
                    label = f"id={iid}"
                    x_min = max(x0 + 2, min(x0 + target_w - 40, lx))
                    y_min = max(pad_top + 2, min(pad_top + target_h - 12, ly))
                    # Draw a small black patch for readability, then label text.
                    draw.rectangle([x_min - 1, y_min - 1, x_min + 34, y_min + 10], fill=(0, 0, 0))
                    draw.text((x_min, y_min), label, fill=(color[0], color[1], color[2]))
            if sim_resized is not None:
                y0 = pad_top + 2 * target_h + gap + 2
                x0 = max(0, int((canvas_w - sim_resized.width) // 2))
                canvas.paste(sim_resized, (x0, y0))

            out_path = os.path.join(vis_dir, f"iter_{int(step):06d}_t{int(t):02d}_instance_img.png")
            canvas.save(out_path)
