import torch
import torch.nn.functional as F
import os


class EfficientOCFVisualizationMixin:
    """Visualization methods for EfficientOCF."""

    @staticmethod
    def _normalize_dense_txyz(x):
        if x is None:
            return None
        if isinstance(x, (list, tuple)):
            if len(x) <= 0:
                return None
            if len(x) == 1:
                x = x[0]
            else:
                try:
                    x = torch.stack(list(x), dim=0)
                except Exception:
                    return None
        if (not torch.is_tensor(x)) or (x.numel() <= 0):
            return None
        if x.dim() == 6:
            if int(x.shape[0]) != 1:
                return None
            if int(x.shape[2]) == 1:
                return x[0, :, 0].to(torch.long).contiguous()
            return None
        if x.dim() == 5:
            if int(x.shape[0]) == 1:
                return x[0].to(torch.long).contiguous()
            if int(x.shape[1]) == 1:
                return x[:, 0].to(torch.long).contiguous()
            return None
        if x.dim() == 4:
            return x.to(torch.long).contiguous()
        return None

    @staticmethod
    def _extract_sparse_list_gt_occ_inst(gt_occ_inst):
        if gt_occ_inst is None:
            return None
        x = gt_occ_inst
        if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], (list, tuple)):
            x = x[0]
        if not isinstance(x, (list, tuple)):
            return None
        out = []
        for rows in x:
            arr = rows
            if torch.is_tensor(arr):
                arr = arr.detach().cpu().numpy()
            else:
                arr = getattr(arr, "data", arr)
            arr = arr if isinstance(arr, (list, tuple)) else arr
            import numpy as np
            arr = np.asarray(arr)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                if arr.size <= 0:
                    arr = np.zeros((0, 5), dtype=np.int64)
                else:
                    arr = np.vstack(arr)
            arr = np.asarray(arr, dtype=np.int64)
            if arr.size <= 0:
                arr = np.zeros((0, 5), dtype=np.int64)
            if arr.ndim != 2 or arr.shape[1] < 5:
                return None
            out.append(arr)
        return out

    @staticmethod
    def _extract_meta_tokens(img_metas):
        if img_metas is None:
            return "na", "na"
        m = img_metas
        m = getattr(m, "data", m)
        if isinstance(m, (list, tuple)) and len(m) > 0:
            m0 = m[0]
            if isinstance(m0, (list, tuple)) and len(m0) > 0:
                m = m0[0]
            else:
                m = m0
        if not isinstance(m, dict):
            return "na", "na"
        return str(m.get("scene_token", "na")), str(m.get("lidar_token", "na"))

    @torch.no_grad()
    def maybe_save_gt_alignment_bev_vis(
        self,
        segmentation=None,
        gt_occ=None,
        gt_occ_inst=None,
        img_metas=None,
        step: int = 0,
    ) -> None:
        vis_every = int(getattr(self, "debug_gt_alignment_vis_every", 0))
        if vis_every <= 0:
            return
        if (int(step) % vis_every) != 0:
            return
        if not self._is_main_process():
            return

        occ_txyz = self._normalize_dense_txyz(gt_occ)
        inst_sparse = self._extract_sparse_list_gt_occ_inst(gt_occ_inst)
        if (occ_txyz is None) or (inst_sparse is None):
            return

        t_all = min(int(occ_txyz.shape[0]), len(inst_sparse))
        if t_all <= 0:
            return
        t_show = min(t_all, int(getattr(self, "debug_gt_alignment_vis_max_frames", 7)))
        occ_txyz = occ_txyz[:t_show]
        inst_sparse = inst_sparse[:t_show]

        try:
            import numpy as np
            from PIL import Image, ImageDraw
        except Exception:
            return

        x_dim = int(occ_txyz.shape[1])
        y_dim = int(occ_txyz.shape[2])

        gmo_ids = tuple(int(v) for v in getattr(self, "gmo_ids", (2, 3, 4, 5, 6, 7, 9, 10)))

        gap = 4
        text_h = 18
        row_h = y_dim
        row_w = x_dim
        canvas_w = t_show * row_w + (t_show - 1) * gap
        canvas_h = text_h + (row_h * 3) + (gap * 2)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for t in range(t_show):
            x0 = t * (row_w + gap)

            occ = occ_txyz[t].to(torch.long)
            occ_valid = (occ != 255) & (occ != 0)
            occ_fg = torch.zeros_like(occ_valid)
            for cid in gmo_ids:
                occ_fg = occ_fg | (occ == int(cid))
            occ_mask = occ_valid & occ_fg
            occ_bev = torch.any(occ_mask, dim=2).to(torch.bool).cpu().numpy().T
            occ_rgb = np.zeros((row_h, row_w, 3), dtype=np.uint8)
            occ_rgb[occ_bev] = np.array([240, 240, 240], dtype=np.uint8)

            inst_rows = inst_sparse[t]
            inst_bev = np.zeros((row_h, row_w), dtype=np.bool_)
            if inst_rows.shape[0] > 0:
                valid = (
                    (inst_rows[:, 0] >= 0) & (inst_rows[:, 0] < x_dim)
                    & (inst_rows[:, 1] >= 0) & (inst_rows[:, 1] < y_dim)
                    & (inst_rows[:, 4] > 0)
                )
                rr = inst_rows[valid]
                if rr.shape[0] > 0:
                    inst_bev[rr[:, 1], rr[:, 0]] = True
            inst_rgb = np.zeros((row_h, row_w, 3), dtype=np.uint8)
            inst_rgb[inst_bev] = np.array([240, 240, 240], dtype=np.uint8)

            overlap_rgb = np.zeros((row_h, row_w, 3), dtype=np.uint8)
            overlap = inst_bev & occ_bev
            only_inst = inst_bev & (~occ_bev)
            only_occ = occ_bev & (~inst_bev)
            overlap_rgb[overlap] = np.array([255, 220, 70], dtype=np.uint8)
            overlap_rgb[only_inst] = np.array([40, 180, 40], dtype=np.uint8)
            overlap_rgb[only_occ] = np.array([220, 60, 60], dtype=np.uint8)

            y1 = text_h
            y2 = y1 + row_h + gap
            y3 = y2 + row_h + gap
            canvas[y1:y1 + row_h, x0:x0 + row_w] = inst_rgb
            canvas[y2:y2 + row_h, x0:x0 + row_w] = occ_rgb
            canvas[y3:y3 + row_h, x0:x0 + row_w] = overlap_rgb

        img = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(img)
        draw.text(
            (2, 1),
            "row1: gt_occ_inst  row2: gt_occ(gmo)  row3: overlap(yellow)/inst_only(green)/occ_only(red)",
            fill=(255, 255, 255),
        )
        for t in range(t_show):
            x0 = t * (row_w + gap)
            inst_n = int(np.count_nonzero(canvas[text_h:text_h + row_h, x0:x0 + row_w, 0]))
            occ_y = text_h + row_h + gap
            occ_n = int(np.count_nonzero(canvas[occ_y:occ_y + row_h, x0:x0 + row_w, 0]))
            inst_y = text_h + (row_h + gap) * 2
            overlap_n = int(np.count_nonzero(canvas[inst_y:inst_y + row_h, x0:x0 + row_w, 0]))
            tag = f"t={t}"
            if t == int(getattr(self, "time_receptive_field", 3) - 1):
                tag += "*"
            draw.text((x0 + 2, text_h + 2), f"{tag} i={inst_n}", fill=(255, 255, 255))
            draw.text((x0 + 2, occ_y + 2), f"o={occ_n}", fill=(255, 255, 255))
            draw.text((x0 + 2, inst_y + 2), f"ov={overlap_n}", fill=(255, 255, 255))
            # Thin white border per visualization panel for readability on dark background.
            draw.rectangle((x0, y1, x0 + row_w - 1, y1 + row_h - 1), outline=(255, 255, 255), width=1)
            draw.rectangle((x0, y2, x0 + row_w - 1, y2 + row_h - 1), outline=(255, 255, 255), width=1)
            draw.rectangle((x0, y3, x0 + row_w - 1, y3 + row_h - 1), outline=(255, 255, 255), width=1)

        scene_token, lidar_token = self._extract_meta_tokens(img_metas)
        vis_dir = str(getattr(self, "debug_gt_alignment_vis_dir", "./work_dirs/gt_alignment_vis"))
        os.makedirs(vis_dir, exist_ok=True)
        out_name = f"iter_{int(step):06d}_{scene_token}_{lidar_token}.png"
        img.save(os.path.join(vis_dir, out_name))

    @staticmethod
    def _bev_mask_from_instance_txyz(frame_xyz):
        if (not torch.is_tensor(frame_xyz)) or frame_xyz.dim() != 3:
            return None
        return torch.any((frame_xyz.to(torch.long) > 0) & (frame_xyz.to(torch.long) != 255), dim=2)

    @staticmethod
    def _bev_mask_for_ids(frame_xyz, instance_ids):
        if (
            (not torch.is_tensor(frame_xyz))
            or frame_xyz.dim() != 3
            or (not torch.is_tensor(instance_ids))
            or int(instance_ids.numel()) <= 0
        ):
            return None
        out = torch.zeros(frame_xyz.shape[:2], device=frame_xyz.device, dtype=torch.bool)
        frame = frame_xyz.to(torch.long)
        for iid in instance_ids.to(device=frame.device, dtype=torch.long).reshape(-1).tolist():
            out = out | torch.any(frame == int(iid), dim=2)
        return out

    @torch.no_grad()
    def maybe_save_gmo_quality_alignment_vis(
        self,
        fine_instance_occ3d_txyz=None,
        bbox_instance_occ3d_txyz=None,
        inst_match_result=None,
        img_metas=None,
        step: int = 0,
        present_idx: int = 0,
    ) -> None:
        vis_every = int(getattr(self, "debug_gmo_quality_alignment_vis_every", 0))
        if vis_every <= 0:
            return
        if (int(step) % vis_every) != 0:
            return
        if not self._is_main_process():
            return

        fine_txyz = self._normalize_dense_txyz(fine_instance_occ3d_txyz)
        bbox_txyz = self._normalize_dense_txyz(bbox_instance_occ3d_txyz)
        if fine_txyz is None or bbox_txyz is None:
            return
        t_all = min(int(fine_txyz.shape[0]), int(bbox_txyz.shape[0]))
        if t_all <= 0:
            return
        fine_txyz = fine_txyz[:t_all]
        bbox_txyz = bbox_txyz[:t_all]
        pidx = max(0, min(int(present_idx), t_all - 1))

        pass_ids = fine_txyz.new_empty((0,), dtype=torch.long)
        if isinstance(inst_match_result, dict):
            matched_query_idx = inst_match_result.get("matched_query_idx", None)
            matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
            gt_ids_n = inst_match_result.get("gt_ids_n", None)
            gt_valid_tn = inst_match_result.get("gt_valid_tn", None)
            gt_centers_tn3 = inst_match_result.get("gt_centers_tn3", None)
            if (
                torch.is_tensor(matched_query_idx)
                and torch.is_tensor(matched_inst_idx)
                and torch.is_tensor(gt_ids_n)
                and int(matched_query_idx.numel()) > 0
                and int(matched_query_idx.numel()) == int(matched_inst_idx.numel())
                and int(gt_ids_n.numel()) > 0
            ):
                mq = matched_query_idx.to(device=fine_txyz.device, dtype=torch.long).reshape(-1)
                mi = matched_inst_idx.to(device=fine_txyz.device, dtype=torch.long).reshape(-1)
                gt_ids = gt_ids_n.to(device=fine_txyz.device, dtype=torch.long).reshape(-1)
                keep = (mq >= 0) & (mi >= 0) & (mi < int(gt_ids.numel()))
                if bool(keep.any().item()):
                    mi = mi[keep]
                    pair_gt_ids = gt_ids.index_select(0, mi)
                    quality_keep, _, _, ratios = self._build_matched_gmo_quality_mask(
                        pair_gt_ids=pair_gt_ids,
                        matched_inst_idx=mi,
                        gt_occ_txyz=fine_txyz,
                        bbox_instance_occ3d_txyz=bbox_txyz,
                        gt_centers_tn3=gt_centers_tn3,
                        gt_valid_tn=gt_valid_tn,
                        present_idx=pidx,
                        roi_radius_m=float(getattr(self, "query_gmo_quality_roi_radius_m", 30.0)),
                        min_occupancy_ratio=float(getattr(self, "query_gmo_quality_min_occupancy_ratio", 0.4)),
                    )
                    if bool(quality_keep.any().item()):
                        pass_ids = torch.unique(pair_gt_ids[quality_keep], sorted=True)
        try:
            import numpy as np
            from PIL import Image, ImageDraw
        except Exception:
            return

        scene_token, lidar_token = self._extract_meta_tokens(img_metas)
        vis_dir = str(getattr(self, "debug_gmo_quality_alignment_vis_dir", "./work_dirs/gmo_quality_alignment_vis"))
        out_dir = os.path.join(vis_dir, f"iter_{int(step):06d}_{scene_token}_{lidar_token}")
        os.makedirs(out_dir, exist_ok=True)

        x_dim = int(min(fine_txyz.shape[1], bbox_txyz.shape[1]))
        y_dim = int(min(fine_txyz.shape[2], bbox_txyz.shape[2]))
        gap = 6
        text_h = 44
        panel_w = x_dim
        panel_h = y_dim
        canvas_w = panel_w * 3 + gap * 2
        canvas_h = text_h + panel_h
        pass_ids_cpu = pass_ids.detach().cpu() if torch.is_tensor(pass_ids) else torch.empty((0,), dtype=torch.long)
        pass_preview = ",".join(str(int(v)) for v in pass_ids_cpu[:12].tolist())
        if int(pass_ids_cpu.numel()) > 12:
            pass_preview += ",..."

        for t in range(t_all):
            fine = fine_txyz[t, :x_dim, :y_dim].to(torch.long)
            bbox = bbox_txyz[t, :x_dim, :y_dim].to(torch.long)
            fine_bev_t = self._bev_mask_from_instance_txyz(fine)
            bbox_bev_t = self._bev_mask_from_instance_txyz(bbox)
            if fine_bev_t is None or bbox_bev_t is None:
                continue
            pass_fine_t = self._bev_mask_for_ids(fine, pass_ids)
            pass_bbox_t = self._bev_mask_for_ids(bbox, pass_ids)

            fine_bev = fine_bev_t.detach().cpu().numpy().T
            bbox_bev = bbox_bev_t.detach().cpu().numpy().T
            pass_fine = pass_fine_t.detach().cpu().numpy().T if pass_fine_t is not None else np.zeros_like(fine_bev)
            pass_bbox = pass_bbox_t.detach().cpu().numpy().T if pass_bbox_t is not None else np.zeros_like(bbox_bev)

            bbox_rgb = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
            fine_rgb = np.zeros_like(bbox_rgb)
            diff_rgb = np.zeros_like(bbox_rgb)
            bbox_rgb[bbox_bev] = np.array([70, 150, 255], dtype=np.uint8)
            fine_rgb[fine_bev] = np.array([255, 110, 70], dtype=np.uint8)
            bbox_rgb[pass_bbox] = np.array([180, 220, 255], dtype=np.uint8)
            fine_rgb[pass_fine] = np.array([255, 210, 170], dtype=np.uint8)

            overlap = bbox_bev & fine_bev
            bbox_only = bbox_bev & (~fine_bev)
            fine_only = fine_bev & (~bbox_bev)
            diff_rgb[overlap] = np.array([255, 220, 70], dtype=np.uint8)
            diff_rgb[bbox_only] = np.array([70, 150, 255], dtype=np.uint8)
            diff_rgb[fine_only] = np.array([255, 70, 70], dtype=np.uint8)
            pass_any = pass_bbox | pass_fine
            diff_rgb[pass_any & overlap] = np.array([255, 255, 255], dtype=np.uint8)

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            y0 = text_h
            canvas[y0:y0 + panel_h, 0:panel_w] = bbox_rgb
            x1 = panel_w + gap
            canvas[y0:y0 + panel_h, x1:x1 + panel_w] = fine_rgb
            x2 = (panel_w + gap) * 2
            canvas[y0:y0 + panel_h, x2:x2 + panel_w] = diff_rgb

            img = Image.fromarray(canvas, mode="RGB")
            draw = ImageDraw.Draw(img)
            tag = f"iter={int(step)} frame={t}/{t_all - 1}"
            if t == pidx:
                tag += " present/filter"
            bbox_n = int(np.count_nonzero(bbox_bev))
            fine_n = int(np.count_nonzero(fine_bev))
            overlap_n = int(np.count_nonzero(overlap))
            ratio = float(fine_n) / float(max(bbox_n, 1))
            draw.text((4, 2), f"{tag} scene={scene_token} lidar={lidar_token}", fill=(255, 255, 255))
            draw.text(
                (4, 20),
                f"bbox={bbox_n} fine={fine_n} overlap={overlap_n} fine/bbox={ratio:.3f} pass_obj={int(pass_ids_cpu.numel())} ids={pass_preview}",
                fill=(255, 255, 255),
            )
            draw.text((4, text_h + 4), "bbox GT", fill=(255, 255, 255))
            draw.text((x1 + 4, text_h + 4), "fine voxel GT", fill=(255, 255, 255))
            draw.text((x2 + 4, text_h + 4), "overlap: yellow, bbox-only: blue, fine-only: red, pass-overlap: white", fill=(255, 255, 255))
            draw.rectangle((0, y0, panel_w - 1, y0 + panel_h - 1), outline=(255, 255, 255), width=1)
            draw.rectangle((x1, y0, x1 + panel_w - 1, y0 + panel_h - 1), outline=(255, 255, 255), width=1)
            draw.rectangle((x2, y0, x2 + panel_w - 1, y0 + panel_h - 1), outline=(255, 255, 255), width=1)
            out_name = f"frame_t{int(t):02d}.png"
            img.save(os.path.join(out_dir, out_name))

    @staticmethod
    def _depth_bin_center_from_cfg(bin_idx: torch.Tensor, num_bins: int, depth_min: float, depth_max: float):
        if (num_bins <= 0) or (not (depth_max > depth_min)):
            return None
        bin_size = (float(depth_max) - float(depth_min)) / float(num_bins)
        return float(depth_min) + (bin_idx.to(torch.float32) + 0.5) * float(bin_size)

    @staticmethod
    def _instance_color(instance_id: int):
        iid = int(instance_id)
        return (
            int((37 * iid + 17) % 255),
            int((97 * iid + 29) % 255),
            int((53 * iid + 71) % 255),
        )

    @staticmethod
    def _to_display_u8_local(img_chw: torch.Tensor):
        x = img_chw.detach().to(torch.float32).cpu()
        if x.dim() != 3:
            return None
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
    def _save_query_inst_depth_lift_status(
        vis_dir: str,
        step: int,
        scene_token: str,
        lidar_token: str,
        message: str,
    ) -> None:
        try:
            os.makedirs(vis_dir, exist_ok=True)
            out_name = (
                f"iter_{int(step):06d}_{str(scene_token)}_{str(lidar_token)}"
                f"_status.txt"
            )
            out_path = os.path.join(vis_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(str(message).strip() + "\n")
        except Exception:
            return

    @staticmethod
    def _save_query_attn_softargmax_status(
        vis_dir: str,
        step: int,
        scene_token: str,
        lidar_token: str,
        message: str,
    ) -> None:
        try:
            os.makedirs(vis_dir, exist_ok=True)
            out_name = (
                f"iter_{int(step):06d}_{str(scene_token)}_{str(lidar_token)}"
                f"_attn_softargmax_status.txt"
            )
            out_path = os.path.join(vis_dir, out_name)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(str(message).strip() + "\n")
        except Exception:
            return

    @torch.no_grad()
    def maybe_save_query_attn_softargmax_vis(
        self,
        img_inputs_seq=None,
        query_attn_weights_tqnhw: torch.Tensor = None,
        query_attn_soft_lift_pack: dict = None,
        query_match_inputs: dict = None,
        inst_match_result: dict = None,
        gt_segmentation_instance3d_txyz_fine=None,
        gt_segmentation_instance3d_txyz_bbox=None,
        gt_source_start_global: int = None,
        future_egomotion: torch.Tensor = None,
        img_metas=None,
        step: int = 0,
    ) -> None:
        vis_every = int(getattr(self, "debug_query_attn_softargmax_vis_every", 0))
        if vis_every <= 0:
            return
        if (int(step) % vis_every) != 0:
            return
        if not self._is_main_process():
            return

        vis_dir = str(
            getattr(
                self,
                "debug_query_attn_softargmax_vis_dir",
                "./work_dirs/query_attn_softargmax_vis",
            )
        )
        os.makedirs(vis_dir, exist_ok=True)
        scene_token, lidar_token = self._extract_meta_tokens(img_metas)

        def _bail(reason: str):
            self._save_query_attn_softargmax_status(
                vis_dir=vis_dir,
                step=int(step),
                scene_token=scene_token,
                lidar_token=lidar_token,
                message=reason,
            )
            return

        if (
            (not torch.is_tensor(query_attn_weights_tqnhw))
            or query_attn_weights_tqnhw.dim() != 5
            or (not isinstance(query_attn_soft_lift_pack, dict))
            or (not isinstance(query_match_inputs, dict))
            or (not isinstance(inst_match_result, dict))
            or (not torch.is_tensor(future_egomotion))
        ):
            return _bail("skip: attention weights, soft-lift pack, match inputs, or match result missing")

        if (
            (not isinstance(img_inputs_seq, (list, tuple)))
            or len(img_inputs_seq) <= 0
            or (not torch.is_tensor(img_inputs_seq[0]))
            or img_inputs_seq[0].dim() != 6
            or int(img_inputs_seq[0].shape[0]) <= 0
        ):
            return _bail("skip: img_inputs_seq[0] must be [B,T,N,C,H,W]")
        imgs_seq = img_inputs_seq[0]

        center_img_tqn2 = query_attn_soft_lift_pack.get("center_img_tqn2", None)
        center_feat_tqn2 = query_attn_soft_lift_pack.get("center_feat_tqn2", None)
        selected_cam_idx_tq = query_attn_soft_lift_pack.get("selected_cam_idx_tq", None)
        selected_cam_valid_tq = query_attn_soft_lift_pack.get("selected_cam_valid_tq", None)
        attn_src_idx_t = query_attn_soft_lift_pack.get("attn_src_idx_t", None)
        output_global_idx_t = query_attn_soft_lift_pack.get("query_output_global_frame_indices", None)
        present_global_idx_meta = query_attn_soft_lift_pack.get("query_present_global_idx", None)
        present_global_idx = int(getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1)))
        if torch.is_tensor(present_global_idx_meta) and present_global_idx_meta.numel() > 0:
            present_global_idx = int(present_global_idx_meta.reshape(-1)[0].item())
        if (
            (not torch.is_tensor(center_img_tqn2))
            or center_img_tqn2.dim() != 4
            or int(center_img_tqn2.shape[-1]) != 2
            or (not torch.is_tensor(center_feat_tqn2))
            or center_feat_tqn2.dim() != 4
            or int(center_feat_tqn2.shape[-1]) != 2
            or (not torch.is_tensor(attn_src_idx_t))
            or attn_src_idx_t.numel() <= 0
        ):
            return _bail("skip: soft-lift pack lacks center_img/center_feat/attn_src_idx")
        if (not torch.is_tensor(output_global_idx_t)) or output_global_idx_t.numel() <= 0:
            frame_start_fallback = int(query_match_inputs.get("frame_start_idx", 0))
            output_global_idx_t = attn_src_idx_t.to(torch.long) + int(frame_start_fallback)
        else:
            output_global_idx_t = output_global_idx_t.to(torch.long)

        matched_query_idx = inst_match_result.get("matched_query_idx", None)
        matched_inst_idx = inst_match_result.get("matched_inst_idx", None)
        gt_ids_n = inst_match_result.get("gt_ids_n", None)
        if (
            (not torch.is_tensor(matched_query_idx))
            or (not torch.is_tensor(matched_inst_idx))
            or (not torch.is_tensor(gt_ids_n))
            or matched_query_idx.numel() <= 0
            or matched_query_idx.numel() != matched_inst_idx.numel()
            or gt_ids_n.numel() <= 0
        ):
            return _bail("skip: no Hungarian matched query/GT pairs")

        fine_txyz = self._normalize_dense_txyz(gt_segmentation_instance3d_txyz_fine)
        bbox_txyz = self._normalize_dense_txyz(gt_segmentation_instance3d_txyz_bbox)
        if fine_txyz is None and bbox_txyz is None:
            return _bail("skip: fine/bbox instance occupancy tensor missing")

        try:
            import numpy as np
            from PIL import Image, ImageDraw
        except Exception:
            return _bail("skip: PIL/numpy import failed")

        t_attn, q_total, n_cam_attn, feat_h, feat_w = [int(v) for v in query_attn_weights_tqnhw.shape]
        t_center = int(center_img_tqn2.shape[0])
        q_center = int(center_img_tqn2.shape[1])
        n_cam_center = int(center_img_tqn2.shape[2])
        t_count = int(min(t_center, int(attn_src_idx_t.numel()), int(output_global_idx_t.numel())))
        n_cam = int(min(n_cam_attn, n_cam_center, int(query_match_inputs.get("n_cam", n_cam_attn))))
        if t_count <= 0 or q_total <= 0 or q_center <= 0 or n_cam <= 0:
            return _bail("skip: invalid attention/center dimensions")
        has_selected_cam = (
            torch.is_tensor(selected_cam_idx_tq)
            and torch.is_tensor(selected_cam_valid_tq)
            and selected_cam_idx_tq.dim() == 2
            and selected_cam_valid_tq.dim() == 2
            and int(selected_cam_idx_tq.shape[0]) >= t_count
            and int(selected_cam_valid_tq.shape[0]) >= t_count
            and int(selected_cam_idx_tq.shape[1]) >= min(q_total, q_center)
            and int(selected_cam_valid_tq.shape[1]) >= min(q_total, q_center)
        )
        if has_selected_cam:
            selected_cam_idx_tq = selected_cam_idx_tq[:t_count, :min(q_total, q_center)].to(
                device=query_attn_weights_tqnhw.device,
                dtype=torch.long,
            )
            selected_cam_valid_tq = selected_cam_valid_tq[:t_count, :min(q_total, q_center)].to(
                device=query_attn_weights_tqnhw.device,
                dtype=torch.bool,
            )

        mq = matched_query_idx.to(device=query_attn_weights_tqnhw.device, dtype=torch.long).reshape(-1)
        mi = matched_inst_idx.to(device=query_attn_weights_tqnhw.device, dtype=torch.long).reshape(-1)
        gt_ids = gt_ids_n.to(device=query_attn_weights_tqnhw.device, dtype=torch.long).reshape(-1)
        keep = (
            (mq >= 0)
            & (mq < min(q_total, q_center))
            & (mi >= 0)
            & (mi < int(gt_ids.numel()))
        )
        if not bool(keep.any().item()):
            return _bail("skip: matched query/GT indices are out of bounds")
        mq = mq[keep]
        mi = mi[keep]
        gt_ids_matched = gt_ids.index_select(0, mi)

        max_queries = max(1, int(getattr(self, "debug_query_attn_softargmax_vis_max_queries", 16)))
        candidate_limit = min(int(mq.numel()), max(max_queries, max_queries * 4))
        attn_src = attn_src_idx_t[:t_count].to(device=query_attn_weights_tqnhw.device, dtype=torch.long)
        output_global_idx_t = output_global_idx_t[:t_count].to(device=query_attn_weights_tqnhw.device, dtype=torch.long)
        attn_src = attn_src.clamp(min=0, max=max(0, t_attn - 1))
        attn_for_rank = query_attn_weights_tqnhw.index_select(0, attn_src).index_select(1, mq)[:, :, :n_cam]
        rank_score = attn_for_rank.to(torch.float32).amax(dim=(0, 2, 3, 4))
        order = torch.argsort(rank_score, descending=True)[:candidate_limit]
        mq = mq.index_select(0, order)
        gt_ids_matched = gt_ids_matched.index_select(0, order)

        gt_t_src = fine_txyz if fine_txyz is not None else bbox_txyz
        gt_t_total = int(gt_t_src.shape[0])
        if gt_source_start_global is None:
            seq_total = int(imgs_seq.shape[1])
            if gt_t_total >= seq_total:
                gt_source_start_global = 0
            elif gt_t_total >= int(self.n_future_frames_plus):
                gt_source_start_global = int(
                    self.time_receptive_field + self.n_future_frames - self.n_future_frames_plus
                )
            else:
                gt_source_start_global = int(query_match_inputs.get("frame_start_idx", 0))
        else:
            gt_source_start_global = int(gt_source_start_global)
        gt_t_idx = output_global_idx_t - int(gt_source_start_global)
        gt_keep = (gt_t_idx >= 0) & (gt_t_idx < int(gt_t_src.shape[0]))
        if not bool(gt_keep.any().item()):
            return _bail(
                "skip: no query output frame maps into GT source window "
                f"(gt_source_start_global={int(gt_source_start_global)}, "
                f"query_global={[int(v) for v in output_global_idx_t.detach().cpu().tolist()]})"
            )
        if not bool(gt_keep.all().item()):
            keep_idx = torch.nonzero(gt_keep, as_tuple=False).squeeze(1)
            attn_src = attn_src.index_select(0, keep_idx)
            output_global_idx_t = output_global_idx_t.index_select(0, keep_idx)
            gt_t_idx = gt_t_idx.index_select(0, keep_idx)
            t_count = int(attn_src.numel())

        proj_pack = self._project_gt_instances_to_cam_masks(
            segmentation_instance3d_txyz=fine_txyz,
            future_egomotion=future_egomotion,
            gt_inst_ids_n=gt_ids_matched,
            query_match_inputs=query_match_inputs,
            attn_t_idx_t=attn_src,
            seg_t_idx_t=gt_t_idx,
            fallback_segmentation_instance3d_txyz=bbox_txyz,
        )
        if not isinstance(proj_pack, dict):
            return _bail("skip: GT CAM projection failed")
        gt_mask_tknhw = proj_pack.get("gt_inst_cam_mask_tnnhw", None)
        gt_valid_tkn = proj_pack.get("gt_inst_cam_valid_tnn", None)
        if (
            (not torch.is_tensor(gt_mask_tknhw))
            or gt_mask_tknhw.dim() != 5
            or (not torch.is_tensor(gt_valid_tkn))
            or gt_valid_tkn.dim() != 3
        ):
            return _bail("skip: projected GT CAM masks invalid")

        t_count = int(min(t_count, gt_mask_tknhw.shape[0]))
        k_count = int(min(mq.numel(), gt_mask_tknhw.shape[1]))
        n_cam = int(min(n_cam, gt_mask_tknhw.shape[2]))
        if t_count <= 0 or k_count <= 0 or n_cam <= 0:
            return _bail("skip: no projected GT mask entries")
        mq = mq[:k_count]
        gt_ids_matched = gt_ids_matched[:k_count]
        gt_mask_tknhw = gt_mask_tknhw[:t_count, :k_count, :n_cam].to(torch.bool)
        gt_valid_tkn = gt_valid_tkn[:t_count, :k_count, :n_cam].to(torch.bool)
        if not bool(gt_valid_tkn.any().item()):
            return _bail("skip: projected GT CAM masks have no valid matched entries")

        max_frames = max(
            int(getattr(self, "time_receptive_field", 1)),
            int(getattr(self, "debug_query_attn_softargmax_vis_max_frames", 2)),
        )
        max_cams = max(1, int(getattr(self, "debug_query_attn_softargmax_vis_max_cams", 3)))
        frame_score = gt_valid_tkn.to(torch.float32).sum(dim=(1, 2))
        frame_order = torch.argsort(frame_score, descending=True).tolist()
        frame_keep = [int(t) for t in frame_order if float(frame_score[t].item()) > 0.0][:max_frames]
        if len(frame_keep) <= 0:
            return _bail("skip: no frame has valid matched projection")

        frame_start = int(query_match_inputs.get("frame_start_idx", 0))
        cfg_img_h = max(1, int(query_match_inputs.get("img_h", int(imgs_seq.shape[-2]))))
        cfg_img_w = max(1, int(query_match_inputs.get("img_w", int(imgs_seq.shape[-1]))))
        tau = max(1e-6, float(getattr(self, "query_attn_softargmax_tau", 1.0)))

        def _resize_panel(panel_img):
            max_w = 640
            max_h = 420
            w, h = panel_img.size
            scale = min(1.0, float(max_w) / float(max(1, w)), float(max_h) / float(max(1, h)))
            if scale < 1.0:
                panel_img = panel_img.resize(
                    (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                    Image.BILINEAR,
                )
            return panel_img

        def _make_panel(t_idx: int, k_idx: int, cam_idx: int):
            src_t = int(attn_src[t_idx].item())
            img_t = int(frame_start + src_t)
            if img_t < 0 or img_t >= int(imgs_seq.shape[1]) or cam_idx >= int(imgs_seq.shape[2]):
                return None
            base_np = self._to_display_u8_local(imgs_seq[0, img_t, cam_idx])
            if base_np is None:
                return None
            img_h, img_w = int(base_np.shape[0]), int(base_np.shape[1])

            q_idx = int(mq[k_idx].item())
            attn_hw = query_attn_weights_tqnhw[src_t, q_idx, cam_idx].to(torch.float32)
            prob_pack = self._normalize_attention_prob_spatial_tqnhw(
                attn_hw.view(1, 1, 1, feat_h, feat_w),
                tau=tau,
            )
            if not torch.is_tensor(prob_pack):
                return None
            prob_hw = prob_pack[0, 0, 0]
            prob_img = F.interpolate(
                prob_hw[None, None],
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            prob_np = prob_img.detach().cpu().numpy()
            prob_np = prob_np / max(float(prob_np.max()), 1e-12)

            mask_img = F.interpolate(
                gt_mask_tknhw[t_idx, k_idx, cam_idx].to(torch.float32)[None, None],
                size=(img_h, img_w),
                mode="nearest",
            )[0, 0] > 0.5
            mask_np = mask_img.detach().cpu().numpy().astype(np.bool_)

            panel = Image.fromarray(base_np, mode="RGB").convert("RGBA")
            heat = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            heat[..., 0] = np.clip(255.0 * prob_np, 0.0, 255.0).astype(np.uint8)
            heat[..., 1] = np.clip(150.0 * prob_np, 0.0, 255.0).astype(np.uint8)
            heat[..., 3] = np.clip(150.0 * np.sqrt(prob_np), 0.0, 150.0).astype(np.uint8)
            panel = Image.alpha_composite(panel, Image.fromarray(heat, mode="RGBA"))

            gt_overlay = np.zeros((img_h, img_w, 4), dtype=np.uint8)
            gt_overlay[mask_np, 1] = 255
            gt_overlay[mask_np, 2] = 220
            gt_overlay[mask_np, 3] = 95
            panel = Image.alpha_composite(panel, Image.fromarray(gt_overlay, mode="RGBA"))

            draw = ImageDraw.Draw(panel)
            center = center_img_tqn2[t_idx, q_idx, cam_idx].detach().to(torch.float32).cpu()
            cx = float(center[0].item()) * float(max(1, img_w - 1)) / float(max(1, cfg_img_w - 1))
            cy = float(center[1].item()) * float(max(1, img_h - 1)) / float(max(1, cfg_img_h - 1))
            radius = max(4, int(round(min(img_h, img_w) / 90.0)))
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                outline=(255, 255, 0, 255),
                width=max(2, radius // 2),
            )
            draw.line((cx - radius * 2, cy, cx + radius * 2, cy), fill=(255, 255, 0, 255), width=2)
            draw.line((cx, cy - radius * 2, cx, cy + radius * 2), fill=(255, 255, 0, 255), width=2)

            max_flat = int(torch.argmax(prob_hw.reshape(-1)).item())
            max_y = max_flat // max(1, feat_w)
            max_x = max_flat % max(1, feat_w)
            mx = (float(max_x) / float(max(1, feat_w - 1))) * float(max(1, img_w - 1))
            my = (float(max_y) / float(max(1, feat_h - 1))) * float(max(1, img_h - 1))
            draw.line((mx - radius, my - radius, mx + radius, my + radius), fill=(255, 255, 255, 230), width=2)
            draw.line((mx - radius, my + radius, mx + radius, my - radius), fill=(255, 255, 255, 230), width=2)

            gt_area = int(gt_mask_tknhw[t_idx, k_idx, cam_idx].to(torch.long).sum().item())
            cam_mass = float(attn_hw.clamp_min(0.0).sum().item())
            pmax = float(prob_hw.max().item())
            text = (
                f"q_t={t_idx} g={int(output_global_idx_t[t_idx].item())} "
                f"gt_t={int(gt_t_idx[t_idx].item())} src={src_t} cam={cam_idx} "
                f"q={q_idx} gt={int(gt_ids_matched[k_idx].item())} "
                f"center=({cx:.1f},{cy:.1f}) mask={gt_area} "
                f"mass={cam_mass:.3g} pmax={pmax:.3g}"
            )
            draw.rectangle((0, 0, img_w, 18), fill=(0, 0, 0, 170))
            draw.text((3, 2), text, fill=(255, 255, 255, 255))
            return _resize_panel(panel.convert("RGB"))

        saved = 0
        for t_idx in frame_keep:
            if has_selected_cam:
                selected_cam_k = selected_cam_idx_tq[t_idx].index_select(0, mq)
                selected_valid_k = selected_cam_valid_tq[t_idx].index_select(0, mq)
                selected_valid_k = selected_valid_k & (
                    selected_cam_k >= 0
                ) & (selected_cam_k < n_cam)
                cam_score = torch.zeros((n_cam,), device=gt_valid_tkn.device, dtype=torch.float32)
                if bool(selected_valid_k.any().item()):
                    k_sel = torch.nonzero(selected_valid_k, as_tuple=False).squeeze(1)
                    cam_sel = selected_cam_k.index_select(0, k_sel)
                    gt_ok = gt_valid_tkn[t_idx, k_sel, cam_sel]
                    if bool(gt_ok.any().item()):
                        k_ok = k_sel[gt_ok]
                        cam_ok = cam_sel[gt_ok]
                        cam_score.scatter_add_(
                            0,
                            cam_ok,
                            torch.ones_like(cam_ok, dtype=torch.float32),
                        )
                cam_order = torch.argsort(cam_score, descending=True).tolist()
                cam_keep = [int(c) for c in cam_order if float(cam_score[c].item()) > 0.0][:max_cams]
                if len(cam_keep) <= 0:
                    cam_score = gt_valid_tkn[t_idx].to(torch.float32).sum(dim=0)
                    cam_order = torch.argsort(cam_score, descending=True).tolist()
                    cam_keep = [int(c) for c in cam_order if float(cam_score[c].item()) > 0.0][:max_cams]
            else:
                cam_score = gt_valid_tkn[t_idx].to(torch.float32).sum(dim=0)
                cam_order = torch.argsort(cam_score, descending=True).tolist()
                cam_keep = [int(c) for c in cam_order if float(cam_score[c].item()) > 0.0][:max_cams]
            for cam_idx in cam_keep:
                valid_k_mask = gt_valid_tkn[t_idx, :, cam_idx]
                if has_selected_cam:
                    selected_cam_k = selected_cam_idx_tq[t_idx].index_select(0, mq)
                    selected_valid_k = selected_cam_valid_tq[t_idx].index_select(0, mq)
                    valid_k_mask = valid_k_mask & selected_valid_k & (selected_cam_k == int(cam_idx))
                valid_k = torch.nonzero(valid_k_mask, as_tuple=False).squeeze(1)
                if int(valid_k.numel()) <= 0:
                    continue
                area_k = gt_mask_tknhw[t_idx, valid_k, cam_idx].to(torch.float32).sum(dim=(1, 2))
                attn_peak_k = []
                src_t = int(attn_src[t_idx].item())
                for k_idx in valid_k.tolist():
                    q_idx = int(mq[int(k_idx)].item())
                    attn_peak_k.append(float(query_attn_weights_tqnhw[src_t, q_idx, cam_idx].to(torch.float32).max().item()))
                attn_peak = torch.as_tensor(attn_peak_k, device=area_k.device, dtype=torch.float32)
                order_k = torch.argsort(area_k + 1e-3 * attn_peak, descending=True)
                valid_k = valid_k.index_select(0, order_k[:max_queries])

                panels = []
                for k_idx in valid_k.tolist():
                    panel = _make_panel(int(t_idx), int(k_idx), int(cam_idx))
                    if panel is not None:
                        panels.append(panel)
                if len(panels) <= 0:
                    continue

                cols = min(4, len(panels))
                rows = int((len(panels) + cols - 1) // cols)
                gap = 4
                cell_w = max(p.size[0] for p in panels)
                cell_h = max(p.size[1] for p in panels)
                canvas = Image.new(
                    "RGB",
                    (cols * cell_w + (cols - 1) * gap, rows * cell_h + (rows - 1) * gap),
                    (10, 10, 10),
                )
                for idx, panel in enumerate(panels):
                    x0 = (idx % cols) * (cell_w + gap)
                    y0 = (idx // cols) * (cell_h + gap)
                    canvas.paste(panel, (x0, y0))

                out_name = (
                    f"iter_{int(step):06d}_{scene_token}_{lidar_token}"
                    f"_t{int(t_idx):02d}_g{int(output_global_idx_t[t_idx].item()):02d}"
                    f"_gt{int(gt_t_idx[t_idx].item()):02d}"
                    f"_cam{int(cam_idx):02d}_attn_softargmax.png"
                )
                canvas.save(os.path.join(vis_dir, out_name))
                saved += 1

        if saved <= 0:
            return _bail("skip: no panels were saved after projection/ranking")
        self._save_query_attn_softargmax_status(
            vis_dir=vis_dir,
            step=int(step),
            scene_token=scene_token,
            lidar_token=lidar_token,
            message=(
                f"ok: saved={int(saved)} "
                f"query_present_only={int(bool(getattr(self, 'query_present_only', False)))} "
                f"global_timeline=[0,1,2,3,4,5,6]=[t-2,t-1,t,t+1,t+2,t+3,t+4] "
                f"plus_timeline=[0,1,2,3,4,5]=[t-1,t,t+1,t+2,t+3,t+4] "
                f"present_global_idx={int(present_global_idx)} "
                f"gt_source_start_global={int(gt_source_start_global)} "
                f"query_local_to_global={[int(v) for v in output_global_idx_t.detach().cpu().tolist()]} "
                f"attn_src_idx_t={[int(v) for v in attn_src.detach().cpu().tolist()]} "
                f"gt_source_idx_t={[int(v) for v in gt_t_idx.detach().cpu().tolist()]}"
            ),
        )

    @torch.no_grad()
    def maybe_save_query_inst_depth_lift_vis(
        self,
        img_inputs_seq=None,
        query_match_inputs: dict = None,
        query_inst_depth_target_pack: dict = None,
        gt_segmentation_instance3d_txyz_fine=None,
        gt_segmentation_instance3d_txyz_bbox=None,
        future_egomotion: torch.Tensor = None,
        img_metas=None,
        step: int = 0,
    ) -> None:
        vis_every = int(getattr(self, "debug_query_inst_depth_lift_vis_every", 0))
        if vis_every <= 0:
            return
        if (int(step) % vis_every) != 0:
            return
        if not self._is_main_process():
            return
        vis_dir = str(getattr(self, "debug_query_inst_depth_lift_vis_dir", "./work_dirs/query_inst_depth_lift_vis"))
        os.makedirs(vis_dir, exist_ok=True)
        scene_token, lidar_token = self._extract_meta_tokens(img_metas)

        def _bail(reason: str):
            self._save_query_inst_depth_lift_status(
                vis_dir=vis_dir,
                step=int(step),
                scene_token=scene_token,
                lidar_token=lidar_token,
                message=reason,
            )
            return

        if (not isinstance(query_match_inputs, dict)) or (not isinstance(query_inst_depth_target_pack, dict)):
            return _bail("skip: query_match_inputs or query_inst_depth_target_pack is missing")

        depth_bin_tcn = query_inst_depth_target_pack.get("gt_inst_depth_bin_tcn", None)
        depth_valid_tcn = query_inst_depth_target_pack.get("gt_inst_depth_valid_tcn", None)
        inst_ids_n = query_inst_depth_target_pack.get("gt_inst_ids_n", None)
        attn_t_idx_t = query_inst_depth_target_pack.get("attn_t_idx_t", None)
        output_global_idx_t = query_inst_depth_target_pack.get("query_output_global_frame_indices", None)
        present_global_idx_meta = query_inst_depth_target_pack.get("query_present_global_idx", None)
        if (
            (not torch.is_tensor(depth_bin_tcn))
            or (not torch.is_tensor(depth_valid_tcn))
            or depth_bin_tcn.dim() != 3
            or depth_valid_tcn.dim() != 3
            or tuple(depth_bin_tcn.shape) != tuple(depth_valid_tcn.shape)
        ):
            return _bail("skip: depth target tensors are invalid")

        rots_tn33 = query_match_inputs.get("rots_tn33", None)
        trans_tn3 = query_match_inputs.get("trans_tn3", None)
        intrins_tn33 = query_match_inputs.get("intrins_tn33", None)
        post_rots_tn33 = query_match_inputs.get("post_rots_tn33", None)
        post_trans_tn3 = query_match_inputs.get("post_trans_tn3", None)
        img_h = int(query_match_inputs.get("img_h", 0))
        img_w = int(query_match_inputs.get("img_w", 0))
        feat_h = int(query_match_inputs.get("feat_h", 0))
        feat_w = int(query_match_inputs.get("feat_w", 0))
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
            or feat_h <= 1
            or feat_w <= 1
        ):
            return _bail("skip: camera tensors/query_match_inputs metadata invalid")

        fine_txyz = self._normalize_dense_txyz(gt_segmentation_instance3d_txyz_fine)
        bbox_txyz = self._normalize_dense_txyz(gt_segmentation_instance3d_txyz_bbox)
        if fine_txyz is None:
            return _bail("skip: fine occupancy tensor is missing")
        if bbox_txyz is None:
            bbox_txyz = fine_txyz

        query_present_only = bool(getattr(self, "query_present_only", False))
        present_global_idx = int(getattr(self, "query_present_global_idx", int(self.time_receptive_field - 1)))
        if torch.is_tensor(present_global_idx_meta) and present_global_idx_meta.numel() > 0:
            present_global_idx = int(present_global_idx_meta.reshape(-1)[0].item())

        t_hist = int(
            min(
                depth_bin_tcn.shape[0],
                depth_valid_tcn.shape[0],
                rots_tn33.shape[0],
                trans_tn3.shape[0],
                intrins_tn33.shape[0],
                post_rots_tn33.shape[0],
                post_trans_tn3.shape[0],
            )
        )
        if t_hist <= 0:
            return _bail("skip: t_hist <= 0")
        n_cam = int(min(depth_bin_tcn.shape[1], rots_tn33.shape[1]))
        n_inst = int(depth_bin_tcn.shape[2])
        if n_cam <= 0 or n_inst <= 0:
            return _bail(f"skip: n_cam={n_cam}, n_inst={n_inst}")

        depth_bin_tcn = depth_bin_tcn[:t_hist, :n_cam].to(torch.long)
        depth_valid_tcn = depth_valid_tcn[:t_hist, :n_cam].to(torch.bool)
        src_t_idx_t = torch.arange(t_hist, device=depth_bin_tcn.device, dtype=torch.long)
        if torch.is_tensor(attn_t_idx_t) and int(attn_t_idx_t.numel()) >= t_hist:
            src_t_idx_t = attn_t_idx_t[:t_hist].to(device=depth_bin_tcn.device, dtype=torch.long)
        src_t_max = int(
            min(
                rots_tn33.shape[0],
                trans_tn3.shape[0],
                intrins_tn33.shape[0],
                post_rots_tn33.shape[0],
                post_trans_tn3.shape[0],
                fine_txyz.shape[0],
                bbox_txyz.shape[0],
            )
            - 1
        )
        if src_t_max < 0:
            return _bail("skip: invalid source-frame upper bound")
        src_t_idx_t = src_t_idx_t.clamp(min=0, max=src_t_max)
        if (not torch.is_tensor(output_global_idx_t)) or int(output_global_idx_t.numel()) < t_hist:
            output_global_idx_t = src_t_idx_t.to(torch.long) + int(frame_start)
        else:
            output_global_idx_t = output_global_idx_t[:t_hist].to(device=depth_bin_tcn.device, dtype=torch.long)
        if torch.is_tensor(inst_ids_n) and inst_ids_n.numel() == n_inst:
            inst_ids_n = inst_ids_n.to(torch.long)
        else:
            inst_ids_n = torch.arange(n_inst, dtype=torch.long, device=depth_bin_tcn.device)

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
        num_bins = int(getattr(self, "query_inst_depth_num_bins", 64))
        if (num_bins <= 0) or (not (depth_max > depth_min)):
            return _bail(
                f"skip: invalid depth range/bins (num_bins={num_bins}, min={depth_min}, max={depth_max})"
            )

        proj_pack = self._project_gt_instances_to_cam_masks(
            segmentation_instance3d_txyz=fine_txyz,
            future_egomotion=future_egomotion,
            gt_inst_ids_n=inst_ids_n,
            query_match_inputs=query_match_inputs,
            attn_t_idx_t=src_t_idx_t,
            fallback_segmentation_instance3d_txyz=bbox_txyz,
        )
        if not isinstance(proj_pack, dict):
            return _bail("skip: GT projection pack is None")
        mask_tnnhw = proj_pack.get("gt_inst_cam_mask_tnnhw", None)
        mask_valid_tnn = proj_pack.get("gt_inst_cam_valid_tnn", None)
        if (
            (not torch.is_tensor(mask_tnnhw))
            or (not torch.is_tensor(mask_valid_tnn))
            or mask_tnnhw.dim() != 5
            or mask_valid_tnn.dim() != 3
        ):
            return _bail("skip: projected GT mask tensors invalid")
        if int(mask_tnnhw.shape[0]) < t_hist:
            return _bail(
                f"skip: projected GT mask time too short ({int(mask_tnnhw.shape[0])} < {t_hist})"
            )
        mask_tnnhw = mask_tnnhw[:t_hist, :n_inst, :n_cam].to(torch.bool)
        mask_valid_tnn = mask_valid_tnn[:t_hist, :n_inst, :n_cam].to(torch.bool)

        valid_triplet_tcn = depth_valid_tcn & mask_valid_tnn.permute(0, 2, 1)
        # If fine projection and fallback both miss due id/layout mismatch, retry with bbox-only source.
        if (not bool(valid_triplet_tcn.any().item())) and torch.is_tensor(bbox_txyz):
            proj_pack_bbox = self._project_gt_instances_to_cam_masks(
                segmentation_instance3d_txyz=bbox_txyz,
                future_egomotion=future_egomotion,
                gt_inst_ids_n=inst_ids_n,
                query_match_inputs=query_match_inputs,
                attn_t_idx_t=src_t_idx_t,
                fallback_segmentation_instance3d_txyz=None,
            )
            if isinstance(proj_pack_bbox, dict):
                mask_bbox = proj_pack_bbox.get("gt_inst_cam_mask_tnnhw", None)
                valid_bbox = proj_pack_bbox.get("gt_inst_cam_valid_tnn", None)
                if (
                    torch.is_tensor(mask_bbox)
                    and torch.is_tensor(valid_bbox)
                    and mask_bbox.dim() == 5
                    and valid_bbox.dim() == 3
                    and int(mask_bbox.shape[0]) >= t_hist
                ):
                    mask_tnnhw = mask_bbox[:t_hist, :n_inst, :n_cam].to(torch.bool)
                    mask_valid_tnn = valid_bbox[:t_hist, :n_inst, :n_cam].to(torch.bool)
                    valid_triplet_tcn = depth_valid_tcn & mask_valid_tnn.permute(0, 2, 1)
        if not bool(valid_triplet_tcn.any().item()):
            if not bool(getattr(self, "_dbg_printed_query_inst_depth_lift_empty", False)):
                print(
                    "[EfficientOCF][depth-lift-vis] skip: no valid (t,cam,inst) after projection "
                    f"(step={int(step)}, depth_valid={float(depth_valid_tcn.to(torch.float32).sum().item()):.0f})",
                    flush=True,
                )
                self._dbg_printed_query_inst_depth_lift_empty = True
            return _bail(
                "skip: no valid (t,cam,inst) after projection "
                f"(depth_valid_count={float(depth_valid_tcn.to(torch.float32).sum().item()):.0f})"
            )

        # Build present-frame axis-aligned GT bbox corners from bbox supervision occupancy.
        pc = torch.as_tensor(self.point_cloud_range, dtype=torch.float32, device=depth_bin_tcn.device)
        x_dim, y_dim, z_dim = [int(v) for v in bbox_txyz.shape[1:]]
        voxel = torch.tensor(
            [
                (pc[3] - pc[0]) / float(max(1, x_dim)),
                (pc[4] - pc[1]) / float(max(1, y_dim)),
                (pc[5] - pc[2]) / float(max(1, z_dim)),
            ],
            dtype=torch.float32,
            device=depth_bin_tcn.device,
        )
        pc_min = pc[:3]
        bbox_corners_tn83 = torch.zeros((t_hist, n_inst, 8, 3), device=depth_bin_tcn.device, dtype=torch.float32)
        bbox_center_tn3 = torch.zeros((t_hist, n_inst, 3), device=depth_bin_tcn.device, dtype=torch.float32)
        bbox_valid_tn = torch.zeros((t_hist, n_inst), device=depth_bin_tcn.device, dtype=torch.bool)
        for t in range(t_hist):
            src_t = int(src_t_idx_t[t].item())
            occ_t = bbox_txyz[src_t].to(device=depth_bin_tcn.device, dtype=torch.long)
            for n in range(n_inst):
                iid = int(inst_ids_n[n].item())
                pts = torch.nonzero(occ_t == iid, as_tuple=False)
                if int(pts.numel()) <= 0:
                    continue
                mins = pts.min(dim=0).values.to(torch.float32)
                maxs = pts.max(dim=0).values.to(torch.float32) + 1.0
                wmin = pc_min + mins * voxel
                wmax = pc_min + maxs * voxel
                corners = torch.stack(
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
                bbox_corners_tn83[t, n] = corners
                bbox_center_tn3[t, n] = 0.5 * (wmin + wmax)
                bbox_valid_tn[t, n] = True

        # Build transforms present-lidar -> lidar_t for each history frame.
        if not torch.is_tensor(future_egomotion):
            return _bail("skip: future_egomotion missing")
        ego = future_egomotion
        if ego.dim() == 4:
            ego = ego[0]
        ego = ego.to(torch.float32)
        frame_indices_hist = list(range(frame_start, max(frame_start, frame_end)))
        if len(frame_indices_hist) < rots_tn33.shape[0]:
            frame_indices_hist = list(
                range(int(self.time_receptive_field - int(rots_tn33.shape[0])), int(self.time_receptive_field))
            )
        frame_indices_hist = frame_indices_hist[: int(rots_tn33.shape[0])]
        frame_indices = [int(frame_indices_hist[int(src_t.item())]) for src_t in src_t_idx_t]
        lidar_t_to_present = self._build_lidar_frame_to_present(
            ego,
            frame_indices=frame_indices,
            present_global_idx=present_global_idx,
        )
        if len(lidar_t_to_present) != t_hist:
            return _bail(
                f"skip: lidar_t_to_present length mismatch ({len(lidar_t_to_present)} != {t_hist})"
            )

        try:
            import numpy as np
            from PIL import Image, ImageDraw
        except Exception:
            return _bail("skip: PIL/numpy import failed")

        pc_xmin = float(pc[0].item())
        pc_ymin = float(pc[1].item())
        pc_xmax = float(pc[3].item())
        pc_ymax = float(pc[4].item())
        map_w = max(256, int(getattr(self, "debug_query_inst_depth_lift_vis_bev_w", 768)))
        map_h = max(256, int(getattr(self, "debug_query_inst_depth_lift_vis_bev_h", 768)))
        dx = max(1e-6, pc_xmax - pc_xmin)
        dy = max(1e-6, pc_ymax - pc_ymin)

        def _xy_to_px(x_m: float, y_m: float):
            ux = (float(x_m) - pc_xmin) / dx
            uy = (float(y_m) - pc_ymin) / dy
            px = int(round(ux * float(map_w - 1)))
            py = int(round((1.0 - uy) * float(map_h - 1)))
            return px, py

        def _inside(px: int, py: int):
            return (0 <= int(px) < map_w) and (0 <= int(py) < map_h)

        # Rank frames/cams with most valid triplets.
        max_frames = int(getattr(self, "debug_query_inst_depth_lift_vis_max_frames", 3))
        max_cams = int(getattr(self, "debug_query_inst_depth_lift_vis_max_cams", 2))
        max_inst = int(getattr(self, "debug_query_inst_depth_lift_vis_max_instances", 16))
        valid_count_t = valid_triplet_tcn.to(torch.float32).sum(dim=(1, 2))
        frame_order = torch.argsort(valid_count_t, descending=True).tolist()
        frame_keep = [int(t) for t in frame_order if float(valid_count_t[t].item()) > 0.0][:max_frames]
        if query_present_only and len(frame_keep) > 0:
            present_keep = [
                int(t) for t in frame_keep if int(output_global_idx_t[t].item()) == int(present_global_idx)
            ]
            if len(present_keep) > 0:
                frame_keep = present_keep[:1]
            else:
                frame_keep = frame_keep[:1]
        if len(frame_keep) <= 0:
            return _bail("skip: frame_keep is empty despite valid_triplet")

        # Camera-wise visualization.
        saved_count = 0
        for t in frame_keep:
            cam_count_t = valid_triplet_tcn[t].to(torch.float32).sum(dim=1)
            cam_order = torch.argsort(cam_count_t, descending=True).tolist()
            cam_keep = [int(c) for c in cam_order if float(cam_count_t[c].item()) > 0.0][:max_cams]
            if len(cam_keep) <= 0:
                continue
            for cam in cam_keep:
                canvas_np = np.zeros((map_h, map_w, 3), dtype=np.uint8)
                canvas_np[:, :] = np.array([12, 12, 12], dtype=np.uint8)
                canvas = Image.fromarray(canvas_np, mode="RGB")
                draw = ImageDraw.Draw(canvas)

                # Select instances for this t/c by mask area.
                valid_inst = torch.nonzero(valid_triplet_tcn[t, cam], as_tuple=False).squeeze(1)
                if int(valid_inst.numel()) <= 0:
                    continue
                areas = []
                for n in valid_inst.tolist():
                    m = mask_tnnhw[t, n, cam]
                    areas.append(float(m.to(torch.float32).sum().item()))
                order = sorted(range(len(areas)), key=lambda i: -areas[i])
                valid_inst = valid_inst[torch.as_tensor(order[:max_inst], device=valid_inst.device, dtype=torch.long)]

                drawn = 0
                for n in valid_inst.tolist():
                    mask_feat = mask_tnnhw[t, n, cam]
                    yy, xx = torch.nonzero(mask_feat, as_tuple=True)
                    if int(xx.numel()) <= 0:
                        continue
                    cx_feat = float(xx.to(torch.float32).mean().item())
                    cy_feat = float(yy.to(torch.float32).mean().item())
                    cx_img = (cx_feat / max(1.0, float(feat_w - 1))) * float(img_w - 1)
                    cy_img = (cy_feat / max(1.0, float(feat_h - 1))) * float(img_h - 1)

                    bin_idx = depth_bin_tcn[t, cam, n]
                    depth_mid = self._depth_bin_center_from_cfg(
                        bin_idx=bin_idx,
                        num_bins=num_bins,
                        depth_min=depth_min,
                        depth_max=depth_max,
                    )
                    if depth_mid is None:
                        continue

                    src_t = int(src_t_idx_t[t].item())
                    post_rot_22 = post_rots_tn33[src_t, cam].to(torch.float32)[:2, :2]
                    post_trans_2 = post_trans_tn3[src_t, cam].to(torch.float32)[:2]
                    uv_aug = torch.tensor([cx_img, cy_img], device=depth_bin_tcn.device, dtype=torch.float32)
                    uv_raw = torch.linalg.solve(post_rot_22, (uv_aug - post_trans_2).view(2, 1)).view(2)
                    pix_h = torch.stack(
                        [
                            uv_raw[0],
                            uv_raw[1],
                            torch.ones((), device=depth_bin_tcn.device, dtype=torch.float32),
                        ],
                        dim=0,
                    )
                    cam_ray = torch.linalg.solve(
                        intrins_tn33[src_t, cam].to(torch.float32),
                        pix_h.view(3, 1),
                    ).view(3)
                    cam_xyz = cam_ray * float(depth_mid)
                    lift_lidar_t = (
                        rots_tn33[src_t, cam].to(torch.float32) @ cam_xyz.view(3, 1)
                    ).view(3) + trans_tn3[src_t, cam].to(torch.float32)

                    color = self._instance_color(int(inst_ids_n[n].item()))

                    # 3D error in present-lidar coordinates.
                    lift_h = torch.cat(
                        [lift_lidar_t, torch.ones((1,), device=lift_lidar_t.device, dtype=torch.float32)],
                        dim=0,
                    ).view(4, 1)
                    lift_present = (lidar_t_to_present[t].to(torch.float32) @ lift_h).view(4)[:3]
                    lx = float(lift_present[0].item())
                    ly = float(lift_present[1].item())
                    lpx, lpy = _xy_to_px(lx, ly)

                    # Draw GT bbox footprint + GT center + lifted center in BEV.
                    if bool(bbox_valid_tn[t, n].item()):
                        gt_center_present = bbox_center_tn3[t, n]
                        err3d = torch.norm(lift_present - gt_center_present, p=2).item()
                        gx = float(gt_center_present[0].item())
                        gy = float(gt_center_present[1].item())
                        gpx, gpy = _xy_to_px(gx, gy)

                        foot_xy = bbox_corners_tn83[t, n, :4, :2].to(torch.float32)
                        poly = []
                        for k in range(4):
                            px, py = _xy_to_px(float(foot_xy[k, 0].item()), float(foot_xy[k, 1].item()))
                            poly.append((px, py))
                        if len(poly) == 4:
                            for k in range(4):
                                p0 = poly[k]
                                p1 = poly[(k + 1) % 4]
                                draw.line((p0[0], p0[1], p1[0], p1[1]), fill=(0, 255, 0), width=2)

                        if _inside(gpx, gpy):
                            draw.ellipse((gpx - 3, gpy - 3, gpx + 3, gpy + 3), fill=(0, 255, 0))
                        if _inside(lpx, lpy):
                            draw.line((lpx - 4, lpy, lpx + 4, lpy), fill=(255, 0, 0), width=2)
                            draw.line((lpx, lpy - 4, lpx, lpy + 4), fill=(255, 0, 0), width=2)
                        if _inside(gpx, gpy) and _inside(lpx, lpy):
                            draw.line((gpx, gpy, lpx, lpy), fill=(255, 220, 70), width=1)
                    else:
                        err3d = float("nan")
                        if _inside(lpx, lpy):
                            draw.line((lpx - 4, lpy, lpx + 4, lpy), fill=(255, 0, 0), width=2)
                            draw.line((lpx, lpy - 4, lpx, lpy + 4), fill=(255, 0, 0), width=2)

                    label = (
                        f"id={int(inst_ids_n[n].item())} "
                        f"bin={int(bin_idx.item())} "
                        f"d={float(depth_mid):.2f}m "
                        f"err3d={float(err3d):.2f}m"
                    )
                    tx = int(max(4, min(map_w - 260, 6 + (drawn % 2) * 260)))
                    ty = int(max(18, min(map_h - 14, 24 + (12 * (drawn % 28)))))
                    draw.text((tx, ty), label, fill=(color[0], color[1], color[2]))
                    drawn += 1

                draw.text(
                    (4, 4),
                    (
                        f"BEV q_t={t} g={int(output_global_idx_t[t].item())} src={int(src_t_idx_t[t].item())} cam={cam} "
                        f"valid_inst={int(valid_inst.numel())} "
                        f"depth_bins={num_bins} range=[{depth_min:.1f},{depth_max:.1f})"
                    ),
                    fill=(255, 255, 255),
                )
                out_name = (
                    f"iter_{int(step):06d}_{scene_token}_{lidar_token}"
                    f"_t{int(t):02d}_g{int(output_global_idx_t[t].item()):02d}"
                    f"_cam{int(cam):02d}_bev.png"
                )
                canvas.save(os.path.join(vis_dir, out_name))
                saved_count += 1
        if saved_count <= 0:
            return _bail("skip: visualization loop completed but no image was saved")
        self._save_query_inst_depth_lift_status(
            vis_dir=vis_dir,
            step=int(step),
            scene_token=scene_token,
            lidar_token=lidar_token,
            message=(
                f"ok: saved={int(saved_count)} "
                f"query_present_only={int(query_present_only)} "
                f"timeline=[0,1,2,3,4,5,6]=[t-2,t-1,t,t+1,t+2,t+3,t+4] "
                f"present_global_idx={int(present_global_idx)} "
                f"query_local_to_global={[int(v) for v in output_global_idx_t.detach().cpu().tolist()]} "
                f"attn_src_idx_t={[int(v) for v in src_t_idx_t.detach().cpu().tolist()]}"
            ),
        )

    def _build_query_visualization_bundle(
        self,
        centers_world_tq3: torch.Tensor,
        query_cls_scores_qc: torch.Tensor,
        inst_match_result: dict,
        gt_segmentation_instance3d_txyz: torch.Tensor,
        gt_instance_centers_full_tn3: torch.Tensor = None,
        gt_instance_valid_full_tn: torch.Tensor = None,
        gt_instance_ids_full_n: torch.Tensor = None,
        gaussian_sigmas_world_tq3: torch.Tensor = None,
        mixture_centers_world_tqg3: torch.Tensor = None,
        mixture_sigmas_world_tqg3: torch.Tensor = None,
        mixture_yaw_tqg: torch.Tensor = None,
        mixture_weights_tqg: torch.Tensor = None,
        traj_mode_idx_q: torch.Tensor = None,
        query_attn_cam_score_pack: dict = None,
    ):
        if (not torch.is_tensor(centers_world_tq3)) or centers_world_tq3.dim() != 3:
            return None
        if (not torch.is_tensor(query_cls_scores_qc)) or query_cls_scores_qc.dim() != 2:
            return None
        q_count, c_count = [int(v) for v in query_cls_scores_qc.shape]
        if q_count <= 0 or c_count <= 1:
            return None
        if int(centers_world_tq3.shape[1]) != q_count:
            return None
        if (not torch.is_tensor(gt_segmentation_instance3d_txyz)) or gt_segmentation_instance3d_txyz.dim() != 4:
            return None
        has_surrogate_sigma = (
            torch.is_tensor(gaussian_sigmas_world_tq3)
            and tuple(gaussian_sigmas_world_tq3.shape) == tuple(centers_world_tq3.shape)
        )
        has_mixture = (
            torch.is_tensor(mixture_centers_world_tqg3)
            and torch.is_tensor(mixture_sigmas_world_tqg3)
            and torch.is_tensor(mixture_yaw_tqg)
            and torch.is_tensor(mixture_weights_tqg)
            and mixture_centers_world_tqg3.dim() == 4
            and tuple(mixture_centers_world_tqg3.shape) == tuple(mixture_sigmas_world_tqg3.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_yaw_tqg.shape)
            and tuple(mixture_centers_world_tqg3.shape[:3]) == tuple(mixture_weights_tqg.shape)
            and int(mixture_centers_world_tqg3.shape[1]) == q_count
        )
        if (not has_surrogate_sigma) and (not has_mixture):
            return None

        score_frame_count = min(
            int(getattr(self, "time_receptive_field", 1)),
            int(centers_world_tq3.shape[0]),
            int(gt_segmentation_instance3d_txyz.shape[0]),
        )
        if score_frame_count <= 0:
            return None
        score_centers_tq3 = centers_world_tq3[:score_frame_count].contiguous()
        score_gt_occ_txyz = gt_segmentation_instance3d_txyz[:score_frame_count].contiguous()
        score_sigmas_tq3 = (
            gaussian_sigmas_world_tq3[:score_frame_count].contiguous()
            if has_surrogate_sigma
            else centers_world_tq3.new_zeros(score_centers_tq3.shape)
        )
        score_mix_centers_tqg3 = (
            mixture_centers_world_tqg3[:score_frame_count].contiguous()
            if has_mixture
            else None
        )
        score_mix_sigmas_tqg3 = (
            mixture_sigmas_world_tqg3[:score_frame_count].contiguous()
            if has_mixture
            else None
        )
        score_mix_yaw_tqg = (
            mixture_yaw_tqg[:score_frame_count].contiguous()
            if has_mixture
            else None
        )
        score_mix_weights_tqg = (
            mixture_weights_tqg[:score_frame_count].contiguous()
            if has_mixture
            else None
        )

        cls_scores_qc = query_cls_scores_qc.to(device=centers_world_tq3.device, dtype=torch.float32)
        pred_cls_q = torch.argmax(cls_scores_qc, dim=-1).to(torch.long)
        fg_scores_qc = cls_scores_qc.clone()
        fg_scores_qc[:, int(self.query_bg_class)] = 0.0
        cls_prob_q = fg_scores_qc.max(dim=-1).values.clamp(0.0, 1.0)

        if has_mixture:
            iou_q = self._compute_matched_query_sequence_iou_scores(
                centers_world_tq3=score_centers_tq3,
                sigmas_world_tq3=score_sigmas_tq3,
                mixture_centers_world_tqg3=score_mix_centers_tqg3,
                mixture_sigmas_world_tqg3=score_mix_sigmas_tqg3,
                mixture_yaw_tqg=score_mix_yaw_tqg,
                mixture_weights_tqg=score_mix_weights_tqg,
                gt_instance_occ3d_txyz_pred=score_gt_occ_txyz,
                inst_match_result=inst_match_result,
                objectness_scores_tq=None,
                pair_chunk_size=int(self.query_multi_gaussian_pair_chunk),
            )
        else:
            iou_q = self._compute_matched_query_sequence_iou_scores(
                centers_world_tq3=score_centers_tq3,
                sigmas_world_tq3=score_sigmas_tq3,
                gt_instance_occ3d_txyz_pred=score_gt_occ_txyz,
                inst_match_result=inst_match_result,
                objectness_scores_tq=None,
            )
        if not torch.is_tensor(iou_q) or int(iou_q.numel()) != q_count:
            iou_q = centers_world_tq3.new_zeros((q_count,), dtype=torch.float32)
        iou_q = iou_q.to(dtype=torch.float32).clamp(0.0, 1.0)

        w_iou = float(self.debug_query_score_iou_weight)
        w_cls = float(self.debug_query_score_cls_weight)
        w_cam = float(self.debug_query_score_cam_attn_weight)
        cam_attn_score_q = centers_world_tq3.new_zeros((q_count,), dtype=torch.float32)
        cam_attn_score_valid_q = torch.zeros((q_count,), device=centers_world_tq3.device, dtype=torch.bool)
        if isinstance(query_attn_cam_score_pack, dict):
            cam_q = query_attn_cam_score_pack.get("cam_attn_score_q", None)
            cam_valid_q = query_attn_cam_score_pack.get("cam_attn_score_valid_q", None)
            if torch.is_tensor(cam_q) and int(cam_q.numel()) == q_count:
                cam_attn_score_q = cam_q.to(device=centers_world_tq3.device, dtype=torch.float32).clamp(0.0, 1.0)
                if torch.is_tensor(cam_valid_q) and int(cam_valid_q.numel()) == q_count:
                    cam_attn_score_valid_q = cam_valid_q.to(device=centers_world_tq3.device, dtype=torch.bool)
                else:
                    cam_attn_score_valid_q = torch.isfinite(cam_attn_score_q)
            else:
                cam_attn_score_valid_q = torch.zeros((q_count,), device=centers_world_tq3.device, dtype=torch.bool)
        cam_valid_f_q = cam_attn_score_valid_q.to(torch.float32)
        score_num_q = (w_iou * iou_q) + (w_cls * cls_prob_q) + (w_cam * cam_attn_score_q * cam_valid_f_q)
        score_den_q = (w_iou + w_cls) + (w_cam * cam_valid_f_q)
        score_q = score_num_q / score_den_q.clamp_min(1e-6)
        score_q = score_q.clamp(0.0, 1.0)

        candidate_idx = torch.arange(q_count, device=score_q.device, dtype=torch.long)

        fg_mask_q = pred_cls_q != int(self.query_bg_class)
        selected_candidate_idx = torch.nonzero(fg_mask_q, as_tuple=False).squeeze(1)
        if int(selected_candidate_idx.numel()) > 0:
            candidate_scores = score_q.index_select(0, selected_candidate_idx)
            keep_thr = candidate_scores >= float(self.debug_query_score_threshold)
            selected_candidate_idx = selected_candidate_idx[keep_thr]

        if int(selected_candidate_idx.numel()) > 0 and int(self.debug_query_score_topk) > 0:
            cand_scores = score_q.index_select(0, selected_candidate_idx)
            order = torch.argsort(cand_scores, descending=True)
            selected_candidate_idx = selected_candidate_idx.index_select(0, order)
            nms_radius = max(0.0, float(getattr(self, "debug_query_distance_nms_radius_m", 0.0)))
            if nms_radius > 0.0 and int(selected_candidate_idx.numel()) > 0:
                center_frame_idx = min(
                    max(0, int(getattr(self, "time_receptive_field", 1)) - 1),
                    int(centers_world_tq3.shape[0]) - 1,
                )
                centers_xy_q2 = centers_world_tq3[center_frame_idx, :, :2].to(torch.float32)
                kept = []
                radius_sq = float(nms_radius * nms_radius)
                for qid in selected_candidate_idx.tolist():
                    qid = int(qid)
                    if len(kept) > 0:
                        prev = torch.as_tensor(kept, device=centers_xy_q2.device, dtype=torch.long)
                        dist_sq = ((centers_xy_q2.index_select(0, prev) - centers_xy_q2[qid]) ** 2).sum(dim=-1)
                        if bool((dist_sq <= radius_sq).any().item()):
                            continue
                    kept.append(qid)
                selected_candidate_idx = torch.as_tensor(
                    kept,
                    device=selected_candidate_idx.device,
                    dtype=torch.long,
                )
            topk = min(int(self.debug_query_score_topk), int(selected_candidate_idx.numel()))
            selected_idx = selected_candidate_idx[:topk]
        else:
            selected_idx = selected_candidate_idx.new_empty((0,), dtype=torch.long)

        candidate_points = (
            centers_world_tq3.index_select(1, candidate_idx)
            if int(candidate_idx.numel()) > 0 else centers_world_tq3[:, :0, :]
        )
        if has_surrogate_sigma:
            candidate_sigmas = (
                gaussian_sigmas_world_tq3.index_select(1, candidate_idx)
                if int(candidate_idx.numel()) > 0 else gaussian_sigmas_world_tq3[:, :0, :]
            )
        else:
            candidate_sigmas = centers_world_tq3[:, :0, :]
        candidate_mix_centers = (
            mixture_centers_world_tqg3.index_select(1, candidate_idx)
            if (has_mixture and int(candidate_idx.numel()) > 0) else None
        )
        candidate_mix_sigmas = (
            mixture_sigmas_world_tqg3.index_select(1, candidate_idx)
            if (has_mixture and int(candidate_idx.numel()) > 0) else None
        )
        candidate_mix_yaw = (
            mixture_yaw_tqg.index_select(1, candidate_idx)
            if (has_mixture and int(candidate_idx.numel()) > 0) else None
        )
        candidate_mix_weights = (
            mixture_weights_tqg.index_select(1, candidate_idx)
            if (has_mixture and int(candidate_idx.numel()) > 0) else None
        )
        selected_points = (
            centers_world_tq3.index_select(1, selected_idx)
            if int(selected_idx.numel()) > 0 else centers_world_tq3[:, :0, :]
        )
        if has_surrogate_sigma:
            selected_sigmas = (
                gaussian_sigmas_world_tq3.index_select(1, selected_idx)
                if int(selected_idx.numel()) > 0 else gaussian_sigmas_world_tq3[:, :0, :]
            )
        else:
            selected_sigmas = centers_world_tq3[:, :0, :]
        selected_mix_centers = (
            mixture_centers_world_tqg3.index_select(1, selected_idx)
            if (has_mixture and int(selected_idx.numel()) > 0) else None
        )
        selected_mix_sigmas = (
            mixture_sigmas_world_tqg3.index_select(1, selected_idx)
            if (has_mixture and int(selected_idx.numel()) > 0) else None
        )
        selected_mix_yaw = (
            mixture_yaw_tqg.index_select(1, selected_idx)
            if (has_mixture and int(selected_idx.numel()) > 0) else None
        )
        selected_mix_weights = (
            mixture_weights_tqg.index_select(1, selected_idx)
            if (has_mixture and int(selected_idx.numel()) > 0) else None
        )
        matched_idx = selected_idx.new_empty((0,), dtype=torch.long)
        matched_inst_idx = selected_idx.new_empty((0,), dtype=torch.long)
        matched_gt_ids = selected_idx.new_empty((0,), dtype=torch.long)
        if isinstance(inst_match_result, dict):
            match_q_idx = inst_match_result.get("matched_query_idx", None)
            if torch.is_tensor(match_q_idx) and match_q_idx.numel() > 0:
                match_q_idx = match_q_idx.to(device=score_q.device, dtype=torch.long).reshape(-1)
                valid_mask = (match_q_idx >= 0) & (match_q_idx < q_count)
                if bool(valid_mask.any().item()):
                    matched_idx = torch.unique(match_q_idx[valid_mask], sorted=True)
            if int(matched_idx.numel()) > 0:
                match_inst_idx = inst_match_result.get("matched_inst_idx", None)
                gt_ids_n = inst_match_result.get("gt_ids_n", None)
                if (
                    torch.is_tensor(match_q_idx)
                    and torch.is_tensor(match_inst_idx)
                    and torch.is_tensor(gt_ids_n)
                    and int(match_q_idx.numel()) == int(match_inst_idx.numel())
                    and int(gt_ids_n.numel()) > 0
                ):
                    mq = match_q_idx.to(device=score_q.device, dtype=torch.long).reshape(-1)
                    mi = match_inst_idx.to(device=score_q.device, dtype=torch.long).reshape(-1)
                    gt_ids_pool = gt_ids_n.to(device=score_q.device, dtype=torch.long).reshape(-1)
                    keep = (mq >= 0) & (mq < q_count) & (mi >= 0) & (mi < int(gt_ids_pool.numel()))
                    if bool(keep.any().item()):
                        mq = mq[keep]
                        mi = mi[keep]
                        q_to_src = {}
                        for src_idx, qid in enumerate(mq.tolist()):
                            if int(qid) not in q_to_src:
                                q_to_src[int(qid)] = int(src_idx)
                        out_inst = []
                        out_gt_ids = []
                        for qid in matched_idx.tolist():
                            src_idx = q_to_src.get(int(qid), None)
                            if src_idx is None:
                                out_inst.append(-1)
                                out_gt_ids.append(-1)
                            else:
                                inst_idx = int(mi[src_idx].item())
                                out_inst.append(inst_idx)
                                out_gt_ids.append(int(gt_ids_pool[inst_idx].item()))
                        matched_inst_idx = torch.as_tensor(
                            out_inst, device=score_q.device, dtype=torch.long
                        )
                        matched_gt_ids = torch.as_tensor(
                            out_gt_ids, device=score_q.device, dtype=torch.long
                        )
        matched_points = (
            centers_world_tq3.index_select(1, matched_idx)
            if int(matched_idx.numel()) > 0 else centers_world_tq3[:, :0, :]
        )
        if has_surrogate_sigma:
            matched_sigmas = (
                gaussian_sigmas_world_tq3.index_select(1, matched_idx)
                if int(matched_idx.numel()) > 0 else gaussian_sigmas_world_tq3[:, :0, :]
            )
        else:
            matched_sigmas = centers_world_tq3[:, :0, :]
        matched_mix_centers = (
            mixture_centers_world_tqg3.index_select(1, matched_idx)
            if (has_mixture and int(matched_idx.numel()) > 0) else None
        )
        matched_mix_sigmas = (
            mixture_sigmas_world_tqg3.index_select(1, matched_idx)
            if (has_mixture and int(matched_idx.numel()) > 0) else None
        )
        matched_mix_yaw = (
            mixture_yaw_tqg.index_select(1, matched_idx)
            if (has_mixture and int(matched_idx.numel()) > 0) else None
        )
        matched_mix_weights = (
            mixture_weights_tqg.index_select(1, matched_idx)
            if (has_mixture and int(matched_idx.numel()) > 0) else None
        )

        all_gt_traj_tn3 = None
        all_gt_valid_tn = None
        matched_gt_traj_tn3 = None
        matched_gt_valid_tn = None
        if (
            torch.is_tensor(gt_instance_centers_full_tn3)
            and gt_instance_centers_full_tn3.dim() == 3
        ):
            all_gt_traj_tn3 = gt_instance_centers_full_tn3.detach()
            if (
                torch.is_tensor(gt_instance_valid_full_tn)
                and gt_instance_valid_full_tn.dim() == 2
                and int(gt_instance_valid_full_tn.shape[0]) == int(gt_instance_centers_full_tn3.shape[0])
                and int(gt_instance_valid_full_tn.shape[1]) == int(gt_instance_centers_full_tn3.shape[1])
            ):
                all_gt_valid_tn = gt_instance_valid_full_tn.detach()
            if (
                torch.is_tensor(matched_inst_idx)
                and int(matched_inst_idx.numel()) > 0
            ):
                matched_keep = (
                    (matched_inst_idx >= 0)
                    & (matched_inst_idx < int(gt_instance_centers_full_tn3.shape[1]))
                )
                if bool(matched_keep.any().item()):
                    matched_inst_idx_vis = matched_inst_idx[matched_keep].to(
                        device=gt_instance_centers_full_tn3.device,
                        dtype=torch.long,
                    )
                    matched_gt_traj_tn3 = gt_instance_centers_full_tn3.index_select(
                        1, matched_inst_idx_vis
                    ).detach()
                    if torch.is_tensor(all_gt_valid_tn):
                        matched_gt_valid_tn = all_gt_valid_tn.index_select(
                            1, matched_inst_idx_vis
                        ).detach()

        return {
            "candidate_points_tq3": candidate_points.detach(),
            "candidate_sigmas_tq3": candidate_sigmas.detach(),
            "candidate_pred_cls_q": pred_cls_q.index_select(0, candidate_idx).detach(),
            "candidate_cls_prob_q": cls_prob_q.index_select(0, candidate_idx).detach(),
            "candidate_iou_q": iou_q.index_select(0, candidate_idx).detach(),
            "candidate_score_q": score_q.index_select(0, candidate_idx).detach(),
            "selected_points_tq3": selected_points.detach(),
            "selected_sigmas_tq3": selected_sigmas.detach(),
            "selected_pred_cls_q": pred_cls_q.index_select(0, selected_idx).detach(),
            "selected_score_q": score_q.index_select(0, selected_idx).detach(),
            "selected_query_idx_q": selected_idx.detach(),
            "matched_points_tq3": matched_points.detach(),
            "matched_sigmas_tq3": matched_sigmas.detach(),
            "matched_pred_cls_q": pred_cls_q.index_select(0, matched_idx).detach(),
            "matched_score_q": score_q.index_select(0, matched_idx).detach(),
            "matched_query_idx_q": matched_idx.detach(),
            "matched_inst_idx_q": matched_inst_idx.detach(),
            "matched_gt_ids_q": matched_gt_ids.detach(),
            "matched_gt_traj_tn3": matched_gt_traj_tn3,
            "matched_gt_valid_tn": matched_gt_valid_tn,
            "matched_traj_mode_idx_q": (
                traj_mode_idx_q.index_select(0, matched_idx).detach()
                if torch.is_tensor(traj_mode_idx_q) else None
            ),
            "all_gt_traj_tn3": all_gt_traj_tn3,
            "all_gt_valid_tn": all_gt_valid_tn,
            "gt_instance_centers_full_tn3": gt_instance_centers_full_tn3.detach() if torch.is_tensor(gt_instance_centers_full_tn3) else None,
            "gt_instance_valid_full_tn": gt_instance_valid_full_tn.detach() if torch.is_tensor(gt_instance_valid_full_tn) else None,
            "gt_instance_ids_full_n": gt_instance_ids_full_n.detach() if torch.is_tensor(gt_instance_ids_full_n) else None,
            "score_q": score_q.detach(),
            "iou_q": iou_q.detach(),
            "cls_prob_q": cls_prob_q.detach(),
            "cam_attn_score_q": cam_attn_score_q.detach(),
            "cam_attn_score_valid_q": cam_attn_score_valid_q.detach(),
            "score_frame_count": int(score_frame_count),
            "pred_cls_q": pred_cls_q.detach(),
            "gaussian_sigmas_tq3": gaussian_sigmas_world_tq3.detach() if has_surrogate_sigma else None,
            "candidate_mixture_centers_tqg3": candidate_mix_centers.detach() if candidate_mix_centers is not None else None,
            "candidate_mixture_sigmas_tqg3": candidate_mix_sigmas.detach() if candidate_mix_sigmas is not None else None,
            "candidate_mixture_yaw_tqg": candidate_mix_yaw.detach() if candidate_mix_yaw is not None else None,
            "candidate_mixture_weights_tqg": candidate_mix_weights.detach() if candidate_mix_weights is not None else None,
            "selected_mixture_centers_tqg3": selected_mix_centers.detach() if selected_mix_centers is not None else None,
            "selected_mixture_sigmas_tqg3": selected_mix_sigmas.detach() if selected_mix_sigmas is not None else None,
            "selected_mixture_yaw_tqg": selected_mix_yaw.detach() if selected_mix_yaw is not None else None,
            "selected_mixture_weights_tqg": selected_mix_weights.detach() if selected_mix_weights is not None else None,
            "matched_mixture_centers_tqg3": matched_mix_centers.detach() if matched_mix_centers is not None else None,
            "matched_mixture_sigmas_tqg3": matched_mix_sigmas.detach() if matched_mix_sigmas is not None else None,
            "matched_mixture_yaw_tqg": matched_mix_yaw.detach() if matched_mix_yaw is not None else None,
            "matched_mixture_weights_tqg": matched_mix_weights.detach() if matched_mix_weights is not None else None,
            "traj_mode_idx_q": traj_mode_idx_q.detach() if torch.is_tensor(traj_mode_idx_q) else None,
            "top_k": int(self.debug_query_score_topk),
            "score_thr": float(self.debug_query_score_threshold),
            "distance_nms_radius_m": float(getattr(self, "debug_query_distance_nms_radius_m", 0.0)),
            "w_iou": float(w_iou),
            "w_cls": float(w_cls),
            "w_cam": float(w_cam),
        }
