import os
import torch
import numpy as np
import torch.distributed as dist


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


class VisConfig:
    __slots__ = (
        "vis_every", "vis_dir", "point_cloud_range", "spatial_extent3d",
        "class_ids", "class_names", "marker_radius", "conf_thr",
        "gaussian_vis_mode", "gaussian_prob_thr", "gaussian_prob_alpha_scale",
        "gaussian_truncate_sigma",
    )

    def __init__(self, vis_every, vis_dir, point_cloud_range, spatial_extent3d,
                 class_ids, class_names, marker_radius, conf_thr,
                 gaussian_vis_mode, gaussian_prob_thr, gaussian_prob_alpha_scale,
                 gaussian_truncate_sigma):
        self.vis_every = vis_every
        self.vis_dir = vis_dir
        self.point_cloud_range = point_cloud_range
        self.spatial_extent3d = spatial_extent3d
        self.class_ids = class_ids
        self.class_names = class_names
        self.marker_radius = marker_radius
        self.conf_thr = conf_thr
        self.gaussian_vis_mode = gaussian_vis_mode
        self.gaussian_prob_thr = gaussian_prob_thr
        self.gaussian_prob_alpha_scale = gaussian_prob_alpha_scale
        self.gaussian_truncate_sigma = gaussian_truncate_sigma


def draw_marker_splats(canvas, x_idx, y_idx, color, radius, use_max=False):
    if canvas.ndim != 3 or canvas.shape[-1] != 3:
        return
    if x_idx is None or y_idx is None:
        return
    if len(x_idx) <= 0 or len(y_idx) <= 0:
        return
    radius = max(0, int(radius))
    height, width = canvas.shape[:2]
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            xx = np.clip(x_idx + dx, 0, width - 1)
            yy = np.clip(y_idx + dy, 0, height - 1)
            if use_max:
                canvas[yy, xx] = np.maximum(canvas[yy, xx], color)
            else:
                canvas[yy, xx] = color


def draw_cross_marker(canvas, center_x, center_y, color, arm=6, thickness=1,
                      outline_color=(0, 0, 0), outline_thickness=1, use_max=False):
    if canvas.ndim != 3 or canvas.shape[-1] != 3:
        return
    height, width = canvas.shape[:2]
    cx = int(center_x)
    cy = int(center_y)
    if cx < 0 or cx >= width or cy < 0 or cy >= height:
        return

    arm = max(1, int(arm))
    thickness = max(1, int(thickness))
    outline_thickness = max(0, int(outline_thickness))

    def _draw_plus(draw_color, draw_arm, draw_thickness, draw_use_max):
        xs = np.arange(cx - draw_arm, cx + draw_arm + 1, dtype=np.int64)
        ys = np.full_like(xs, fill_value=cy)
        draw_marker_splats(canvas, xs, ys, np.asarray(draw_color, dtype=np.uint8),
                           radius=max(0, int(draw_thickness) - 1), use_max=bool(draw_use_max))
        ys = np.arange(cy - draw_arm, cy + draw_arm + 1, dtype=np.int64)
        xs = np.full_like(ys, fill_value=cx)
        draw_marker_splats(canvas, xs, ys, np.asarray(draw_color, dtype=np.uint8),
                           radius=max(0, int(draw_thickness) - 1), use_max=bool(draw_use_max))

    if outline_thickness > 0 and outline_color is not None:
        _draw_plus(outline_color, draw_arm=arm + outline_thickness,
                   draw_thickness=thickness + (2 * outline_thickness), draw_use_max=False)
    _draw_plus(color, draw_arm=arm, draw_thickness=thickness, draw_use_max=use_max)


def draw_gaussian_bev_footprints(
    canvas, x_idx, y_idx, sigma_x_px, sigma_y_px, color,
    yaw_rad=None, component_weight=None, valid_mask=None,
    vis_mode="ellipse", truncate_sigma=3.0, fill_alpha=0.18, outline_alpha=0.60,
    outline_thickness_px=1, prob_threshold=0.5, prob_alpha_scale=4.0, use_max=True,
):
    if canvas.ndim != 3 or canvas.shape[-1] != 3:
        return
    if x_idx is None or y_idx is None or sigma_x_px is None or sigma_y_px is None:
        return

    vis_mode = str(vis_mode).lower()
    if vis_mode not in ("ellipse", "prob", "threshold"):
        return

    x_idx = np.asarray(x_idx)
    y_idx = np.asarray(y_idx)
    sigma_x_px = np.asarray(sigma_x_px, dtype=np.float32)
    sigma_y_px = np.asarray(sigma_y_px, dtype=np.float32)
    if x_idx.shape != y_idx.shape or x_idx.shape != sigma_x_px.shape or x_idx.shape != sigma_y_px.shape:
        return
    if x_idx.ndim == 1:
        x_idx = x_idx[:, None]
        y_idx = y_idx[:, None]
        sigma_x_px = sigma_x_px[:, None]
        sigma_y_px = sigma_y_px[:, None]
    elif x_idx.ndim != 2:
        return

    num_queries, num_components = x_idx.shape
    if num_queries <= 0 or num_components <= 0:
        return

    color_arr = np.asarray(color, dtype=np.uint8)
    if color_arr.ndim == 1:
        color_arr = np.repeat(color_arr[None, :], repeats=num_queries, axis=0)
    elif color_arr.ndim != 2 or color_arr.shape[0] != num_queries or color_arr.shape[1] != 3:
        return

    if yaw_rad is None:
        yaw_arr = np.zeros((num_queries, num_components), dtype=np.float32)
    else:
        yaw_arr = np.asarray(yaw_rad, dtype=np.float32)
        if yaw_arr.shape != x_idx.shape:
            return

    if component_weight is None:
        weight_arr = np.ones((num_queries, num_components), dtype=np.float32)
    else:
        weight_arr = np.asarray(component_weight, dtype=np.float32)
        if weight_arr.shape != x_idx.shape:
            return

    if valid_mask is None:
        valid_arr = np.ones((num_queries, num_components), dtype=np.bool_)
    else:
        valid_arr = np.asarray(valid_mask, dtype=np.bool_)
        if valid_arr.shape != x_idx.shape:
            return

    height, width = canvas.shape[:2]
    trunc = max(1.0, float(truncate_sigma))
    thickness = max(1, int(outline_thickness_px))
    prob_threshold = float(np.clip(prob_threshold, 0.0, 1.0))
    prob_alpha_scale = max(0.0, float(prob_alpha_scale))

    def _blend_patch(dst_patch, mask, rgb_color, alpha_scale):
        if not np.any(mask):
            return
        alpha = np.asarray(alpha_scale, dtype=np.float32)
        if alpha.ndim == 0:
            alpha = np.full(mask.shape, fill_value=float(alpha), dtype=np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
        if not np.any(alpha > 0.0):
            return
        src = np.clip(alpha[..., None] * rgb_color[None, None, :], 0.0, 255.0).astype(np.uint8)
        if use_max:
            dst_patch[mask] = np.maximum(dst_patch[mask], src[mask])
        else:
            dst_patch[mask] = src[mask]

    def _binary_outline(mask):
        if mask.shape[0] <= 2 or mask.shape[1] <= 2:
            return mask
        inner = np.zeros_like(mask, dtype=np.bool_)
        inner[1:-1, 1:-1] = (
            mask[1:-1, 1:-1] & mask[:-2, 1:-1] & mask[2:, 1:-1]
            & mask[1:-1, :-2] & mask[1:-1, 2:]
        )
        return mask & (~inner)

    for q_idx in range(num_queries):
        q_valid = valid_arr[q_idx]
        if not np.any(q_valid):
            continue
        color_i = color_arr[q_idx].astype(np.float32)

        if vis_mode == "ellipse":
            for g_idx in np.nonzero(q_valid)[0].tolist():
                sx = float(abs(sigma_x_px[q_idx, g_idx]))
                sy = float(abs(sigma_y_px[q_idx, g_idx]))
                if (not np.isfinite(sx)) or (not np.isfinite(sy)):
                    continue

                yaw = float(yaw_arr[q_idx, g_idx])
                if not np.isfinite(yaw):
                    yaw = 0.0
                cos_y = float(np.cos(yaw))
                sin_y = float(np.sin(yaw))
                rx = max(1, int(np.ceil(trunc * np.sqrt((sx * cos_y) ** 2 + (sy * sin_y) ** 2))))
                ry = max(1, int(np.ceil(trunc * np.sqrt((sx * sin_y) ** 2 + (sy * cos_y) ** 2))))
                cx = int(x_idx[q_idx, g_idx])
                cy = int(y_idx[q_idx, g_idx])

                x0 = max(0, cx - rx)
                x1 = min(width - 1, cx + rx)
                y0 = max(0, cy - ry)
                y1 = min(height - 1, cy + ry)
                if x1 < x0 or y1 < y0:
                    continue

                xs = np.arange(x0, x1 + 1, dtype=np.float32)
                ys = np.arange(y0, y1 + 1, dtype=np.float32)
                xx, yy = np.meshgrid(xs, ys)
                dx = xx - float(cx)
                dy = yy - float(cy)
                dx_rot = (cos_y * dx) + (sin_y * dy)
                dy_rot = (-sin_y * dx) + (cos_y * dy)
                norm = (dx_rot / max(sx * trunc, 1e-6)) ** 2 + (dy_rot / max(sy * trunc, 1e-6)) ** 2
                fill_mask = norm <= 1.0
                if not np.any(fill_mask):
                    continue

                inner_sx = max(((sx * trunc) - float(thickness)) / trunc, 1e-6)
                inner_sy = max(((sy * trunc) - float(thickness)) / trunc, 1e-6)
                inner_norm = (dx_rot / max(inner_sx * trunc, 1e-6)) ** 2 + (dy_rot / max(inner_sy * trunc, 1e-6)) ** 2
                outline_mask = fill_mask & (inner_norm >= 1.0)

                patch = canvas[y0:y1 + 1, x0:x1 + 1]
                if fill_alpha > 0.0:
                    _blend_patch(patch, fill_mask, color_i, float(fill_alpha))
                if outline_alpha > 0.0 and np.any(outline_mask):
                    _blend_patch(patch, outline_mask, color_i, float(outline_alpha))
            continue

        rx_all = []
        ry_all = []
        cx_all = []
        cy_all = []
        for g_idx in np.nonzero(q_valid)[0].tolist():
            sx = float(abs(sigma_x_px[q_idx, g_idx]))
            sy = float(abs(sigma_y_px[q_idx, g_idx]))
            if (not np.isfinite(sx)) or (not np.isfinite(sy)):
                continue
            yaw = float(yaw_arr[q_idx, g_idx])
            if not np.isfinite(yaw):
                yaw = 0.0
            cos_y = float(np.cos(yaw))
            sin_y = float(np.sin(yaw))
            rx_all.append(max(1, int(np.ceil(trunc * np.sqrt((sx * cos_y) ** 2 + (sy * sin_y) ** 2)))))
            ry_all.append(max(1, int(np.ceil(trunc * np.sqrt((sx * sin_y) ** 2 + (sy * cos_y) ** 2)))))
            cx_all.append(int(x_idx[q_idx, g_idx]))
            cy_all.append(int(y_idx[q_idx, g_idx]))
        if len(rx_all) <= 0:
            continue

        x0 = max(0, min(cx - rx for cx, rx in zip(cx_all, rx_all)))
        x1 = min(width - 1, max(cx + rx for cx, rx in zip(cx_all, rx_all)))
        y0 = max(0, min(cy - ry for cy, ry in zip(cy_all, ry_all)))
        y1 = min(height - 1, max(cy + ry for cy, ry in zip(cy_all, ry_all)))
        if x1 < x0 or y1 < y0:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float32)
        ys = np.arange(y0, y1 + 1, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        total_prob = np.zeros_like(xx, dtype=np.float32)

        for g_idx in np.nonzero(q_valid)[0].tolist():
            sx = float(abs(sigma_x_px[q_idx, g_idx]))
            sy = float(abs(sigma_y_px[q_idx, g_idx]))
            if (not np.isfinite(sx)) or (not np.isfinite(sy)):
                continue
            cx = float(x_idx[q_idx, g_idx])
            cy = float(y_idx[q_idx, g_idx])
            yaw = float(yaw_arr[q_idx, g_idx])
            if not np.isfinite(yaw):
                yaw = 0.0
            comp_w = float(weight_arr[q_idx, g_idx])
            if not np.isfinite(comp_w):
                comp_w = 1.0
            comp_w = max(0.0, comp_w)
            if comp_w <= 0.0:
                continue

            cos_y = float(np.cos(yaw))
            sin_y = float(np.sin(yaw))
            dx = xx - cx
            dy = yy - cy
            dx_rot = (cos_y * dx) + (sin_y * dy)
            dy_rot = (-sin_y * dx) + (cos_y * dy)
            mahal_sq = (dx_rot / max(sx, 1e-6)) ** 2 + (dy_rot / max(sy, 1e-6)) ** 2
            support_mask = mahal_sq <= (trunc * trunc)
            if not np.any(support_mask):
                continue
            comp_prob = np.exp(-0.5 * mahal_sq).astype(np.float32) * comp_w
            total_prob[support_mask] += comp_prob[support_mask]

        if not np.any(total_prob > 0.0):
            continue

        patch = canvas[y0:y1 + 1, x0:x1 + 1]
        total_prob = np.clip(total_prob, 0.0, 1.0)
        if vis_mode == "prob":
            prob_mask = total_prob > 0.0
            alpha_map = np.clip(total_prob * prob_alpha_scale * max(float(fill_alpha), 0.0), 0.0, 1.0)
            _blend_patch(patch, prob_mask, color_i, alpha_map)
            if outline_alpha > 0.0 and prob_threshold > 0.0:
                thr_mask = total_prob >= prob_threshold
                outline_mask = _binary_outline(thr_mask)
                if np.any(outline_mask):
                    _blend_patch(patch, outline_mask, color_i, float(outline_alpha))
        else:
            fill_mask = total_prob >= prob_threshold
            if not np.any(fill_mask):
                continue
            if fill_alpha > 0.0:
                _blend_patch(patch, fill_mask, color_i, float(fill_alpha))
            if outline_alpha > 0.0:
                outline_mask = _binary_outline(fill_mask)
                if np.any(outline_mask):
                    _blend_patch(patch, outline_mask, color_i, float(outline_alpha))


def get_query_vis_palette(class_ids, class_names):
    base_palette = np.asarray(
        [
            [110, 110, 110],
            [255, 170, 40],
            [80, 180, 255],
            [255, 80, 80],
            [255, 220, 80],
            [180, 100, 255],
            [80, 255, 200],
            [140, 255, 100],
            [255, 120, 220],
            [255, 255, 120],
            [120, 220, 255],
            [255, 160, 120],
        ],
        dtype=np.uint8,
    )
    color_map = {}
    for idx, raw_id in enumerate(class_ids):
        color_map[int(raw_id)] = base_palette[idx % len(base_palette)]
    return color_map


def draw_query_vis_legend(draw, x_start, y_start, canvas_w, class_ids, class_names):
    color_map = get_query_vis_palette(class_ids, class_names)
    items = []
    for raw_id, name in zip(class_ids, class_names):
        raw_id = int(raw_id)
        label = "background query" if raw_id == 0 else str(name)
        items.append((raw_id, label))
    if len(items) <= 0:
        return 0

    x = int(x_start)
    y = int(y_start)
    line_h = 16
    swatch = 10
    gap = 8
    for raw_id, name in items:
        est_w = swatch + 4 + max(42, int(7 * len(name))) + gap
        if x + est_w > int(canvas_w) and x > int(x_start):
            x = int(x_start)
            y += line_h + 6
        color = tuple(int(v) for v in color_map.get(int(raw_id), np.asarray([255, 255, 255], dtype=np.uint8)))
        draw.rectangle([x, y + 3, x + swatch, y + 3 + swatch], fill=color)
        draw.text((x + swatch + 4, y), name, fill=(255, 255, 255))
        x += est_w
    return (y - int(y_start)) + line_h


def draw_generic_vis_legend(draw, x_start, y_start, canvas_w, conf_thr,
                             matched_color=(255, 80, 255), ego_color=(255, 255, 0),
                             gaussian_label="Gaussian footprint (BEV xy)"):
    items = [
        ((240, 240, 240), "GT occupied BEV"),
        ((0, 0, 0), "empty / non-query GT"),
        ((40, 180, 40), "GT overlay"),
        (ego_color, "ego center (+ marker)"),
        ((140, 140, 140), gaussian_label),
        ((30, 255, 255), f"query conf>={conf_thr:.2f}"),
        ((255, 60, 60), f"query conf<{conf_thr:.2f}"),
        (matched_color, "Hungarian-matched query"),
    ]
    x = int(x_start)
    y = int(y_start)
    line_h = 16
    swatch = 10
    gap = 10
    for color, name in items:
        est_w = swatch + 4 + max(72, int(7 * len(name))) + gap
        if x + est_w > int(canvas_w) and x > int(x_start):
            x = int(x_start)
            y += line_h + 6
        draw.rectangle([x, y + 3, x + swatch, y + 3 + swatch], fill=color)
        draw.text((x + swatch + 4, y), name, fill=(255, 255, 255))
        x += est_w
    return (y - int(y_start)) + line_h


def normalize_gt_occ_semantic_for_vis(gt_occ_semantic):
    if gt_occ_semantic is None:
        return None

    seg = gt_occ_semantic
    if isinstance(seg, (list, tuple)):
        if len(seg) <= 0:
            return None
        if len(seg) == 1:
            seg = seg[0]
        else:
            try:
                seg = torch.stack(seg, dim=0)
            except Exception:
                return None

    if not torch.is_tensor(seg):
        return None

    if seg.dim() == 6:
        if seg.shape[0] != 1:
            return None
        if seg.shape[2] in (1, 2):
            seg = seg[0, :, 0]
        else:
            return None
    elif seg.dim() == 5:
        if seg.shape[1] in (1, 2):
            seg = seg[:, 0]
        elif seg.shape[0] == 1:
            seg = seg[0]
        else:
            return None
    elif seg.dim() != 4:
        return None

    return seg.to(torch.long).contiguous()


def normalize_gt_occ_inst_for_vis(gt_occ_inst):
    if gt_occ_inst is None:
        return None, None

    seg = gt_occ_inst
    if isinstance(seg, (list, tuple)):
        if len(seg) <= 0:
            return None, None
        if len(seg) == 1:
            seg = seg[0]
        else:
            try:
                seg = torch.stack(seg, dim=0)
            except Exception:
                return None, None

    if not torch.is_tensor(seg):
        return None, None

    if seg.dim() == 6:
        if seg.shape[0] != 1:
            return None, None
        if seg.shape[2] == 2:
            seg = seg[0]
        elif seg.shape[2] == 1:
            seg = seg[0, :, 0]
        else:
            return None, None
    elif seg.dim() == 5:
        if seg.shape[1] == 2:
            pass
        elif seg.shape[0] == 1:
            seg = seg[0]
        elif seg.shape[1] == 1:
            seg = seg[:, 0]
        else:
            return None, None
    elif seg.dim() != 4:
        return None, None

    cls_t = None
    inst_t = None
    if seg.dim() == 5 and seg.shape[1] == 2:
        cls_t = seg[:, 0]
        inst_t = seg[:, 1]
    elif seg.dim() == 4:
        inst_t = seg
    else:
        return None, None

    if not torch.is_tensor(inst_t):
        return None, None
    inst_t = inst_t.to(torch.long).contiguous()
    if torch.is_tensor(cls_t):
        cls_t = cls_t.to(torch.long).contiguous()
    return cls_t, inst_t


@torch.no_grad()
def save_prob_grid_vis(
    pred_occ_prob,
    points_world,
    gt_occ_gmo,
    step,
    vis_cfg,
    gt_occ_semantic=None,
    max_frames=6,
    voxel_center_offset=0.5,
    prob_threshold=0.5,
    pred_layout="auto",
    pred_occ_prob_pos_obj=None,
    pred_occ_prob_all_obj=None,
    points_world_all=None,
    point_conf_all=None,
    point_class_ids_all=None,
    point_weights_all=None,
    query_vis_bundle=None,
):
    if vis_cfg.vis_every <= 0:
        return
    if (int(step) % vis_cfg.vis_every) != 0:
        return
    if not is_main_process():
        return
    if pred_occ_prob is None or points_world is None:
        return
    if vis_cfg.point_cloud_range is None or vis_cfg.spatial_extent3d is None:
        return

    try:
        from PIL import Image, ImageDraw
    except Exception:
        return

    if points_world.dim() == 4:
        pts = points_world
    elif points_world.dim() == 3:
        pts = points_world.unsqueeze(2)
    elif points_world.dim() == 5:
        pts = points_world[0]
    else:
        return

    if pred_occ_prob.dim() == 5 and pred_occ_prob.shape[1] == 1:
        pred_base = pred_occ_prob[:, 0]
    elif pred_occ_prob.dim() == 4:
        pred_base = pred_occ_prob
    else:
        return
    pred = pred_base

    if gt_occ_gmo.dim() != 4:
        return

    T_gt, Z, X, Y = gt_occ_gmo.shape
    T_pred = pts.shape[0]
    T_prob = pred.shape[0]
    if pred.shape[1] != Z:
        return

    shape_is_zyx = (pred.shape[2] == Y and pred.shape[3] == X)
    shape_is_zxy = (pred.shape[2] == X and pred.shape[3] == Y)
    if not (shape_is_zyx or shape_is_zxy):
        return

    layout = str(pred_layout).lower()
    if layout == "auto":
        if shape_is_zxy and not shape_is_zyx:
            layout = "zxy"
        elif shape_is_zyx and not shape_is_zxy:
            layout = "zyx"
        else:
            layout = "zxy"
    elif layout not in ("zyx", "zxy"):
        return

    T = int(min(T_gt, T_pred, T_prob, max_frames))
    if T <= 0:
        return

    pts = pts[:T].to(torch.float32)
    gt = gt_occ_gmo[:T]

    pc_min = pts.new_tensor(vis_cfg.point_cloud_range[:3])
    extent = pts.new_tensor(vis_cfg.spatial_extent3d)
    grid_size = pts.new_tensor([float(X), float(Y), float(Z)])
    voxel_size = extent / grid_size

    off = float(voxel_center_offset)
    ix = (pts[..., 0] - pc_min[0]) / voxel_size[0] - off
    iy = (pts[..., 1] - pc_min[1]) / voxel_size[1] - off
    iz = (pts[..., 2] - pc_min[2]) / voxel_size[2] - off
    valid = (ix >= 0.0) & (ix <= float(X - 1)) & (iy >= 0.0) & (iy <= float(Y - 1)) & (iz >= 0.0) & (iz <= float(Z - 1))

    def _normalize_points(points_tq3):
        if not torch.is_tensor(points_tq3):
            return None
        if points_tq3.dim() == 4:
            p = points_tq3
        elif points_tq3.dim() == 3:
            p = points_tq3.unsqueeze(2)
        elif points_tq3.dim() == 5:
            p = points_tq3[0]
        else:
            return None
        if int(p.shape[0]) <= 0:
            return None
        return p[:T].to(torch.float32)

    def _expand_attr_for_points(attr, points_t):
        if not torch.is_tensor(attr) or points_t is None:
            return None
        if attr.dim() == 1:
            x = attr[None, :].expand(T, -1)
        else:
            x = attr[:T]
        if x.dim() == 2 and points_t.dim() == 4 and points_t.shape[2] != 1:
            x = x[:, :, None].expand(-1, -1, points_t.shape[2])
        elif x.dim() == 2 and points_t.dim() == 4 and points_t.shape[2] == 1:
            x = x[:, :, None]
        elif x.dim() != points_t.dim() - 1:
            return None
        return x

    def _normalize_sigmas(sigmas_tq3, points_t):
        if (not torch.is_tensor(sigmas_tq3)) or points_t is None:
            return None
        if sigmas_tq3.dim() == 4:
            sig = sigmas_tq3
        elif sigmas_tq3.dim() == 3 and points_t.dim() == 4:
            sig = sigmas_tq3.unsqueeze(2)
        elif sigmas_tq3.dim() == 5:
            sig = sigmas_tq3[0]
        else:
            return None
        sig = sig[:T].to(torch.float32)
        if tuple(sig.shape) != tuple(points_t.shape):
            return None
        return sig

    def _normalize_scalar_attr(attr_tqg, points_t):
        if (not torch.is_tensor(attr_tqg)) or points_t is None:
            return None
        if attr_tqg.dim() == 3:
            attr = attr_tqg
        elif attr_tqg.dim() == 2 and points_t.dim() == 4 and int(points_t.shape[2]) == 1:
            attr = attr_tqg.unsqueeze(2)
        elif attr_tqg.dim() == 4 and attr_tqg.shape[0] == 1:
            attr = attr_tqg[0]
        else:
            return None
        attr = attr[:T].to(torch.float32)
        if tuple(attr.shape) != tuple(points_t.shape[:-1]):
            return None
        return attr

    def _project_points(points_t, scores_t=None, class_ids_t=None):
        if points_t is None:
            return None, None, None, None, None
        ix_t = (points_t[..., 0] - pc_min[0]) / voxel_size[0] - off
        iy_t = (points_t[..., 1] - pc_min[1]) / voxel_size[1] - off
        iz_t = (points_t[..., 2] - pc_min[2]) / voxel_size[2] - off
        valid_t = (
            (ix_t >= 0.0) & (ix_t <= float(X - 1)) &
            (iy_t >= 0.0) & (iy_t <= float(Y - 1)) &
            (iz_t >= 0.0) & (iz_t <= float(Z - 1))
        )
        ix_i = torch.round(ix_t).to(torch.long).cpu().numpy()
        iy_i = torch.round(iy_t).to(torch.long).cpu().numpy()
        valid_np = valid_t.cpu().numpy()
        scores_np = None
        class_np = None
        if torch.is_tensor(scores_t):
            scores_np = scores_t.to(torch.float32).clamp(0.0, 1.0).cpu().numpy()
        if torch.is_tensor(class_ids_t):
            class_np = class_ids_t.to(torch.long).cpu().numpy()
        return ix_i, iy_i, valid_np, scores_np, class_np

    def _project_gaussians(points_t, sigmas_t, yaw_t=None, weight_t=None):
        if points_t is None or sigmas_t is None:
            return None, None, None, None, None, None, None
        ix_t = (points_t[..., 0] - pc_min[0]) / voxel_size[0] - off
        iy_t = (points_t[..., 1] - pc_min[1]) / voxel_size[1] - off
        sigma_x_t = (sigmas_t[..., 0] / voxel_size[0]).to(torch.float32)
        sigma_y_t = (sigmas_t[..., 1] / voxel_size[1]).to(torch.float32)
        valid_t = (
            (ix_t >= 0.0) & (ix_t <= float(X - 1)) &
            (iy_t >= 0.0) & (iy_t <= float(Y - 1)) &
            torch.isfinite(sigma_x_t) & torch.isfinite(sigma_y_t) &
            (sigma_x_t > 0.0) & (sigma_y_t > 0.0)
        )
        ix_i = torch.round(ix_t).to(torch.long).cpu().numpy()
        iy_i = torch.round(iy_t).to(torch.long).cpu().numpy()
        sx_np = sigma_x_t.cpu().numpy()
        sy_np = sigma_y_t.cpu().numpy()
        yaw_np = None
        weight_np = None
        if torch.is_tensor(yaw_t):
            yaw_np = yaw_t[:T].to(torch.float32).cpu().numpy()
        if torch.is_tensor(weight_t):
            weight_np = weight_t[:T].to(torch.float32).cpu().numpy()
        valid_np = valid_t.cpu().numpy()
        return ix_i, iy_i, sx_np, sy_np, yaw_np, weight_np, valid_np

    use_score_bundle = isinstance(query_vis_bundle, dict)
    bundle_top_k = 0
    bundle_score_thr = vis_cfg.conf_thr
    bundle_w_cls = 0.5
    bundle_w_cam = 0.0

    if use_score_bundle:
        pts_candidate = _normalize_points(query_vis_bundle.get("candidate_points_tq3", None))
        pts_selected = _normalize_points(query_vis_bundle.get("selected_points_tq3", None))
        pts_matched = _normalize_points(query_vis_bundle.get("matched_points_tq3", None))
        gpts_candidate = _normalize_points(query_vis_bundle.get("candidate_mixture_centers_tqg3", None))
        gpts_selected = _normalize_points(query_vis_bundle.get("selected_mixture_centers_tqg3", None))
        gpts_matched = _normalize_points(query_vis_bundle.get("matched_mixture_centers_tqg3", None))
        sig_candidate = _normalize_sigmas(query_vis_bundle.get("candidate_mixture_sigmas_tqg3", None), gpts_candidate)
        sig_selected = _normalize_sigmas(query_vis_bundle.get("selected_mixture_sigmas_tqg3", None), gpts_selected)
        sig_matched = _normalize_sigmas(query_vis_bundle.get("matched_mixture_sigmas_tqg3", None), gpts_matched)
        yaw_candidate = _normalize_scalar_attr(query_vis_bundle.get("candidate_mixture_yaw_tqg", None), gpts_candidate)
        yaw_selected = _normalize_scalar_attr(query_vis_bundle.get("selected_mixture_yaw_tqg", None), gpts_selected)
        yaw_matched = _normalize_scalar_attr(query_vis_bundle.get("matched_mixture_yaw_tqg", None), gpts_matched)
        w_candidate = _normalize_scalar_attr(query_vis_bundle.get("candidate_mixture_weights_tqg", None), gpts_candidate)
        w_selected = _normalize_scalar_attr(query_vis_bundle.get("selected_mixture_weights_tqg", None), gpts_selected)
        w_matched = _normalize_scalar_attr(query_vis_bundle.get("matched_mixture_weights_tqg", None), gpts_matched)
        if gpts_candidate is None:
            gpts_candidate = pts_candidate
            sig_candidate = _normalize_sigmas(query_vis_bundle.get("candidate_sigmas_tq3", None), gpts_candidate)
            yaw_candidate = None
            w_candidate = None
        if gpts_selected is None:
            gpts_selected = pts_selected
            sig_selected = _normalize_sigmas(query_vis_bundle.get("selected_sigmas_tq3", None), gpts_selected)
            yaw_selected = None
            w_selected = None
        if gpts_matched is None:
            gpts_matched = pts_matched
            sig_matched = _normalize_sigmas(query_vis_bundle.get("matched_sigmas_tq3", None), gpts_matched)
            yaw_matched = None
            w_matched = None
        sc_candidate = _expand_attr_for_points(query_vis_bundle.get("candidate_score_q", None), pts_candidate)
        sc_selected = _expand_attr_for_points(query_vis_bundle.get("selected_score_q", None), pts_selected)
        sc_matched = _expand_attr_for_points(query_vis_bundle.get("matched_score_q", None), pts_matched)
        cls_candidate = _expand_attr_for_points(query_vis_bundle.get("candidate_pred_cls_q", None), pts_candidate)
        cls_selected = _expand_attr_for_points(query_vis_bundle.get("selected_pred_cls_q", None), pts_selected)
        cls_matched = _expand_attr_for_points(query_vis_bundle.get("matched_pred_cls_q", None), pts_matched)
        bundle_top_k = int(query_vis_bundle.get("top_k", 0))
        bundle_score_thr = float(query_vis_bundle.get("score_thr", bundle_score_thr))
        bundle_w_cls = float(query_vis_bundle.get("w_cls", bundle_w_cls))
        bundle_w_cam = float(query_vis_bundle.get("w_cam", bundle_w_cam))
    else:
        pts_candidate = _normalize_points(points_world_all if points_world_all is not None else pts)
        if point_conf_all is None:
            point_conf_all = point_weights_all
        sc_candidate = _expand_attr_for_points(point_conf_all, pts_candidate)
        cls_candidate = _expand_attr_for_points(point_class_ids_all, pts_candidate)
        pts_selected = pts_candidate
        sc_selected = sc_candidate
        cls_selected = cls_candidate
        pts_matched = None
        gpts_candidate = gpts_selected = gpts_matched = None
        sig_candidate = sig_selected = sig_matched = None
        yaw_candidate = yaw_selected = yaw_matched = None
        w_candidate = w_selected = w_matched = None
        sc_matched = cls_matched = None

    cand_x_i, cand_y_i, cand_valid_np, cand_score_np, cand_cls_np = _project_points(
        pts_candidate, sc_candidate, cls_candidate)
    sel_x_i, sel_y_i, sel_valid_np, sel_score_np, sel_cls_np = _project_points(
        pts_selected, sc_selected, cls_selected)
    mat_x_i, mat_y_i, mat_valid_np, mat_score_np, mat_cls_np = _project_points(
        pts_matched, sc_matched, cls_matched)
    cand_gx_i, cand_gy_i, cand_sx_np, cand_sy_np, cand_yaw_np, cand_w_np, cand_gvalid_np = _project_gaussians(
        gpts_candidate, sig_candidate, yaw_candidate, w_candidate)
    sel_gx_i, sel_gy_i, sel_sx_np, sel_sy_np, sel_yaw_np, sel_w_np, sel_gvalid_np = _project_gaussians(
        gpts_selected, sig_selected, yaw_selected, w_selected)
    mat_gx_i, mat_gy_i, mat_sx_np, mat_sy_np, mat_yaw_np, mat_w_np, mat_gvalid_np = _project_gaussians(
        gpts_matched, sig_matched, yaw_matched, w_matched)
    bundle_has_gaussian = any(x is not None for x in (cand_sx_np, sel_sx_np, mat_sx_np))

    gt_np = gt.cpu().numpy()
    gt_sem_np = None
    if torch.is_tensor(gt_occ_semantic):
        gt_sem = gt_occ_semantic
        if gt_sem.dim() == 6 and gt_sem.shape[0] == 1 and gt_sem.shape[2] == 1:
            gt_sem = gt_sem[0, :, 0]
        elif gt_sem.dim() == 5 and gt_sem.shape[1] == 1:
            gt_sem = gt_sem[:, 0]
        elif gt_sem.dim() == 4:
            pass
        else:
            gt_sem = None
        if gt_sem is not None:
            gt_sem = gt_sem[:T].to(torch.long)
            if gt_sem.shape[-1] == Z and gt_sem.shape[1] == X and gt_sem.shape[2] == Y:
                gt_sem_np = gt_sem.cpu().numpy()
            elif gt_sem.shape[1] == Z and gt_sem.shape[2] == X and gt_sem.shape[3] == Y:
                gt_sem_np = gt_sem.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    row_gt = []
    row_all = []
    row_hi = []
    row_hi_cls = []
    row_matched = []
    row_gt_cls = []
    stats_valid = []
    stats_hi = []
    stats_lo = []
    stats_matched = []

    conf_thr = vis_cfg.conf_thr
    marker_radius = vis_cfg.marker_radius
    gt_color = np.array([40, 180, 40], dtype=np.uint8)
    hi_color = np.array([30, 255, 255], dtype=np.uint8)
    lo_color = np.array([255, 60, 60], dtype=np.uint8)
    matched_color = np.array([255, 80, 255], dtype=np.uint8)
    ego_color = np.array([255, 255, 0], dtype=np.uint8)
    ego_outline_color = np.array([0, 0, 0], dtype=np.uint8)
    ego_cross_arm = max(2, marker_radius + 3)
    ego_cross_thickness = 1
    vx = float(voxel_size[0].item())
    vy = float(voxel_size[1].item())
    ego_ix_f = (0.0 - float(pc_min[0].item())) / max(vx, 1e-6) - off
    ego_iy_f = (0.0 - float(pc_min[1].item())) / max(vy, 1e-6) - off
    ego_visible = (
        (ego_ix_f >= 0.0) and (ego_ix_f <= float(X - 1))
        and (ego_iy_f >= 0.0) and (ego_iy_f <= float(Y - 1))
    )
    ego_ix = int(np.round(ego_ix_f)) if ego_visible else -1
    ego_iy = int(np.round(ego_iy_f)) if ego_visible else -1
    class_palette = get_query_vis_palette(vis_cfg.class_ids, vis_cfg.class_names)
    gaussian_vis_mode = vis_cfg.gaussian_vis_mode
    gaussian_prob_threshold = vis_cfg.gaussian_prob_thr
    gaussian_prob_alpha_scale = vis_cfg.gaussian_prob_alpha_scale

    for t in range(T):
        gt_bev = (gt_np[t] == 1).any(axis=0).T.astype(np.bool_)
        gt_rgb = np.zeros((Y, X, 3), dtype=np.uint8)
        gt_rgb[gt_bev] = np.array([240, 240, 240], dtype=np.uint8)
        gt_cls_rgb = np.zeros((Y, X, 3), dtype=np.uint8)
        if gt_sem_np is not None:
            gt_sem_t = gt_sem_np[t]
            class_count_xyk = []
            class_raw_ids = []
            for raw_id in vis_cfg.class_ids:
                raw_id = int(raw_id)
                if raw_id <= 0:
                    continue
                class_raw_ids.append(raw_id)
                class_count_xyk.append((gt_sem_t == raw_id).sum(axis=-1))
            if len(class_count_xyk) > 0:
                class_count_xyk = np.stack(class_count_xyk, axis=-1)
                dom_idx_xy = np.argmax(class_count_xyk, axis=-1)
                dom_count_xy = np.max(class_count_xyk, axis=-1)
                gt_cls_xy = np.zeros((X, Y, 3), dtype=np.uint8)
                for k, raw_id in enumerate(class_raw_ids):
                    mask_xy = (dom_count_xy > 0) & (dom_idx_xy == k)
                    if np.any(mask_xy):
                        gt_cls_xy[mask_xy] = class_palette.get(
                            int(raw_id), np.asarray([255, 255, 255], dtype=np.uint8))
                gt_cls_rgb = np.transpose(gt_cls_xy, (1, 0, 2)).copy()

        all_ov = np.zeros((Y, X, 3), dtype=np.uint8)
        all_ov[gt_bev] = gt_color
        hi_ov = np.zeros((Y, X, 3), dtype=np.uint8)
        hi_ov[gt_bev] = gt_color
        hi_cls_ov = np.zeros((Y, X, 3), dtype=np.uint8)
        hi_cls_ov[gt_bev] = gt_color
        matched_ov = np.zeros((Y, X, 3), dtype=np.uint8)
        matched_ov[gt_bev] = gt_color

        valid_count = 0
        hi_count = 0
        lo_count = 0
        matched_count = 0
        trunc_sigma = vis_cfg.gaussian_truncate_sigma

        if use_score_bundle:
            if (cand_gx_i is not None) and (cand_gvalid_np is not None):
                if cand_gvalid_np[t].any():
                    draw_gaussian_bev_footprints(
                        all_ov, cand_gx_i[t], cand_gy_i[t], cand_sx_np[t], cand_sy_np[t], lo_color,
                        yaw_rad=None if cand_yaw_np is None else cand_yaw_np[t],
                        component_weight=None if cand_w_np is None else cand_w_np[t],
                        valid_mask=cand_gvalid_np[t], vis_mode=gaussian_vis_mode,
                        truncate_sigma=trunc_sigma, fill_alpha=0.10, outline_alpha=0.30,
                        outline_thickness_px=1, prob_threshold=gaussian_prob_threshold,
                        prob_alpha_scale=gaussian_prob_alpha_scale, use_max=True,
                    )
            if (sel_gx_i is not None) and (sel_gvalid_np is not None):
                if sel_gvalid_np[t].any():
                    draw_gaussian_bev_footprints(
                        all_ov, sel_gx_i[t], sel_gy_i[t], sel_sx_np[t], sel_sy_np[t], hi_color,
                        yaw_rad=None if sel_yaw_np is None else sel_yaw_np[t],
                        component_weight=None if sel_w_np is None else sel_w_np[t],
                        valid_mask=sel_gvalid_np[t], vis_mode=gaussian_vis_mode,
                        truncate_sigma=trunc_sigma, fill_alpha=0.14, outline_alpha=0.55,
                        outline_thickness_px=1, prob_threshold=gaussian_prob_threshold,
                        prob_alpha_scale=gaussian_prob_alpha_scale, use_max=True,
                    )
                    draw_gaussian_bev_footprints(
                        hi_ov, sel_gx_i[t], sel_gy_i[t], sel_sx_np[t], sel_sy_np[t], hi_color,
                        yaw_rad=None if sel_yaw_np is None else sel_yaw_np[t],
                        component_weight=None if sel_w_np is None else sel_w_np[t],
                        valid_mask=sel_gvalid_np[t], vis_mode=gaussian_vis_mode,
                        truncate_sigma=trunc_sigma, fill_alpha=0.14, outline_alpha=0.55,
                        outline_thickness_px=1, prob_threshold=gaussian_prob_threshold,
                        prob_alpha_scale=gaussian_prob_alpha_scale, use_max=True,
                    )
                    if sel_cls_np is not None:
                        raw_cls_hi = sel_cls_np[t].reshape(-1)
                        hi_gauss_colors = np.stack(
                            [class_palette.get(int(cls_id), hi_color) for cls_id in raw_cls_hi.tolist()],
                            axis=0,
                        ).astype(np.uint8)
                    else:
                        hi_gauss_colors = np.repeat(hi_color[None, :], repeats=int(sel_gx_i[t].shape[0]), axis=0)
                    draw_gaussian_bev_footprints(
                        hi_cls_ov, sel_gx_i[t], sel_gy_i[t], sel_sx_np[t], sel_sy_np[t], hi_gauss_colors,
                        yaw_rad=None if sel_yaw_np is None else sel_yaw_np[t],
                        component_weight=None if sel_w_np is None else sel_w_np[t],
                        valid_mask=sel_gvalid_np[t], vis_mode=gaussian_vis_mode,
                        truncate_sigma=trunc_sigma, fill_alpha=0.16, outline_alpha=0.60,
                        outline_thickness_px=1, prob_threshold=gaussian_prob_threshold,
                        prob_alpha_scale=gaussian_prob_alpha_scale, use_max=True,
                    )
            if (mat_gx_i is not None) and (mat_gvalid_np is not None):
                if mat_gvalid_np[t].any():
                    draw_gaussian_bev_footprints(
                        matched_ov, mat_gx_i[t], mat_gy_i[t], mat_sx_np[t], mat_sy_np[t], matched_color,
                        yaw_rad=None if mat_yaw_np is None else mat_yaw_np[t],
                        component_weight=None if mat_w_np is None else mat_w_np[t],
                        valid_mask=mat_gvalid_np[t], vis_mode=gaussian_vis_mode,
                        truncate_sigma=trunc_sigma, fill_alpha=0.16, outline_alpha=0.70,
                        outline_thickness_px=1, prob_threshold=gaussian_prob_threshold,
                        prob_alpha_scale=gaussian_prob_alpha_scale, use_max=True,
                    )
            if (cand_x_i is not None) and (cand_valid_np is not None):
                xt_c = cand_x_i[t].reshape(-1)
                yt_c = cand_y_i[t].reshape(-1)
                vt_c = cand_valid_np[t].reshape(-1)
                valid_count = int(vt_c.sum())
                if vt_c.any():
                    draw_marker_splats(all_ov, xt_c[vt_c], yt_c[vt_c], lo_color,
                                       radius=max(1, marker_radius), use_max=True)
            if (sel_x_i is not None) and (sel_valid_np is not None):
                xt_s = sel_x_i[t].reshape(-1)
                yt_s = sel_y_i[t].reshape(-1)
                vt_s = sel_valid_np[t].reshape(-1)
                hi_count = int(vt_s.sum())
                lo_count = max(0, valid_count - hi_count)
                if vt_s.any():
                    draw_marker_splats(all_ov, xt_s[vt_s], yt_s[vt_s], hi_color,
                                       radius=max(1, marker_radius + 1), use_max=True)
                    draw_marker_splats(hi_ov, xt_s[vt_s], yt_s[vt_s], hi_color,
                                       radius=max(1, marker_radius + 1), use_max=True)
                    if sel_cls_np is not None:
                        raw_cls_hi = sel_cls_np[t].reshape(-1)[vt_s]
                        hi_colors = np.stack(
                            [class_palette.get(int(cls_id), hi_color) for cls_id in raw_cls_hi.tolist()],
                            axis=0,
                        ).astype(np.uint8)
                    else:
                        hi_colors = np.repeat(hi_color[None, :], repeats=int(hi_count), axis=0)
                    draw_marker_splats(hi_cls_ov, xt_s[vt_s], yt_s[vt_s], hi_colors,
                                       radius=max(1, marker_radius + 1), use_max=True)
            if (mat_x_i is not None) and (mat_valid_np is not None):
                xt_m = mat_x_i[t].reshape(-1)
                yt_m = mat_y_i[t].reshape(-1)
                vt_m = mat_valid_np[t].reshape(-1)
                matched_count = int(vt_m.sum())
                if vt_m.any():
                    draw_marker_splats(matched_ov, xt_m[vt_m], yt_m[vt_m], matched_color,
                                       radius=max(1, marker_radius + 1), use_max=True)
        else:
            if (cand_x_i is not None) and (cand_valid_np is not None):
                xt_all = cand_x_i[t].reshape(-1)
                yt_all = cand_y_i[t].reshape(-1)
                vt_all = cand_valid_np[t].reshape(-1)
                if cand_score_np is not None:
                    st_all = np.clip(cand_score_np[t].reshape(-1).astype(np.float32), 0.0, 1.0)
                else:
                    st_all = np.ones_like(xt_all, dtype=np.float32)
                hi = vt_all & (st_all >= conf_thr)
                lo = vt_all & (~hi)
                valid_count = int(vt_all.sum())
                hi_count = int(hi.sum())
                lo_count = int(lo.sum())
                if lo.any():
                    draw_marker_splats(all_ov, xt_all[lo], yt_all[lo], lo_color,
                                       radius=max(1, marker_radius), use_max=True)
                if hi.any():
                    draw_marker_splats(all_ov, xt_all[hi], yt_all[hi], hi_color,
                                       radius=max(1, marker_radius + 1), use_max=True)
                    draw_marker_splats(hi_ov, xt_all[hi], yt_all[hi], hi_color,
                                       radius=max(1, marker_radius + 1), use_max=True)
                    if cand_cls_np is not None:
                        raw_cls_hi = cand_cls_np[t].reshape(-1)[hi]
                        hi_colors = np.stack(
                            [class_palette.get(int(cls_id), hi_color) for cls_id in raw_cls_hi.tolist()],
                            axis=0,
                        ).astype(np.uint8)
                    else:
                        hi_colors = np.repeat(hi_color[None, :], repeats=int(hi_count), axis=0)
                    draw_marker_splats(hi_cls_ov, xt_all[hi], yt_all[hi], hi_colors,
                                       radius=max(1, marker_radius + 1), use_max=True)
            matched_count = 0

        if ego_visible:
            for row_canvas in (gt_rgb, all_ov, hi_ov, hi_cls_ov, matched_ov, gt_cls_rgb):
                draw_cross_marker(row_canvas, center_x=ego_ix, center_y=ego_iy, color=ego_color,
                                   arm=ego_cross_arm, thickness=ego_cross_thickness,
                                   outline_color=ego_outline_color, outline_thickness=1, use_max=False)

        row_gt.append(gt_rgb)
        row_all.append(all_ov)
        row_hi.append(hi_ov)
        row_hi_cls.append(hi_cls_ov)
        row_matched.append(matched_ov)
        row_gt_cls.append(gt_cls_rgb)
        stats_valid.append(valid_count)
        stats_hi.append(hi_count)
        stats_lo.append(lo_count)
        stats_matched.append(matched_count)

    gap = 4
    text_h = 18
    row_h = Y
    canvas_w = T * X + (T - 1) * gap
    num_rows = 6
    legend_h = 120
    canvas_h = text_h + num_rows * row_h + (num_rows - 1) * gap + legend_h
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for t in range(T):
        x0 = t * (X + gap)
        y0 = text_h
        y1 = y0 + row_h + gap
        y2 = y1 + row_h + gap
        y3 = y2 + row_h + gap
        y4 = y3 + row_h + gap
        y5 = y4 + row_h + gap
        canvas[y0:y0 + row_h, x0:x0 + X] = row_gt[t]
        canvas[y1:y1 + row_h, x0:x0 + X] = row_all[t]
        canvas[y2:y2 + row_h, x0:x0 + X] = row_hi[t]
        canvas[y3:y3 + row_h, x0:x0 + X] = row_hi_cls[t]
        canvas[y4:y4 + row_h, x0:x0 + X] = row_matched[t]
        canvas[y5:y5 + row_h, x0:x0 + X] = row_gt_cls[t]

    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)

    if gaussian_vis_mode == "prob":
        gaussian_desc = f"rotated Gaussian prob map (thr={gaussian_prob_threshold:.2f})"
    elif gaussian_vis_mode == "threshold":
        gaussian_desc = f"rotated Gaussian prob>={gaussian_prob_threshold:.2f} fill"
    else:
        gaussian_desc = "rotated Gaussian ellipse footprint"

    if use_score_bundle:
        score_q = query_vis_bundle.get("score_q", None)
        base_score_q = query_vis_bundle.get("base_score_q", None)
        objectness_q = query_vis_bundle.get("objectness_q", None)
        cls_prob_q = query_vis_bundle.get("cls_prob_q", None)
        cam_attn_score_q = query_vis_bundle.get("cam_attn_score_q", None)
        cam_attn_score_valid_q = query_vis_bundle.get("cam_attn_score_valid_q", None)
        score_mean = float(score_q.float().mean().item()) if torch.is_tensor(score_q) and score_q.numel() > 0 else 0.0
        base_mean = float(base_score_q.float().mean().item()) if torch.is_tensor(base_score_q) and base_score_q.numel() > 0 else 0.0
        obj_mean = float(objectness_q.float().mean().item()) if torch.is_tensor(objectness_q) and objectness_q.numel() > 0 else 0.0
        cls_mean = float(cls_prob_q.float().mean().item()) if torch.is_tensor(cls_prob_q) and cls_prob_q.numel() > 0 else 0.0
        cam_mean = 0.0
        if torch.is_tensor(cam_attn_score_q) and cam_attn_score_q.numel() > 0:
            if torch.is_tensor(cam_attn_score_valid_q) and cam_attn_score_valid_q.numel() == cam_attn_score_q.numel():
                valid_mask = cam_attn_score_valid_q.to(dtype=torch.bool)
                if bool(valid_mask.any().item()):
                    cam_mean = float(cam_attn_score_q[valid_mask].float().mean().item())
            else:
                cam_mean = float(cam_attn_score_q.float().mean().item())
        header = (
            f"row1: gt_occ_inst occupancy BEV  "
            f"row2: all queries, high-score highlighted  "
            f"row3: selected(score>=thr + topk)  "
            f"row4: selected(class-colored)  "
            f"row5: Hungarian-matched query centers only  "
            f"row6: GT class BEV(gt_occ_inst cls) | topk={bundle_top_k} thr={bundle_score_thr:.2f} "
            f"w_cls={bundle_w_cls:.2f} w_cam={bundle_w_cam:.2f} "
            f"mean(obj/base/cls/cam/score)=({obj_mean:.3f}/{base_mean:.3f}/{cls_mean:.3f}/{cam_mean:.3f}/{score_mean:.3f})"
        )
        if bundle_has_gaussian:
            header += f" | Gaussian vis: {gaussian_desc}"
    else:
        header = (
            f"row1: gt_occ_inst occupancy BEV  "
            f"row2: all query centers (green=gt, cyan=conf>={conf_thr:.2f}, red=conf<{conf_thr:.2f})  "
            f"row3: query centers with conf>={conf_thr:.2f} only  "
            f"row4: query centers with conf>={conf_thr:.2f} only, class-colored  "
            f"row5: Hungarian-matched query centers only  "
            f"row6: GT class BEV(gt_occ_inst cls), class-colored"
        )
    if ego_visible:
        header += f" | ego(+)=xy(0,0)->pix({ego_ix},{ego_iy})"
    else:
        header += " | ego(+)=xy(0,0) out-of-range"
    draw.text((2, 1), header, fill=(255, 255, 255))

    for t in range(T):
        x0 = t * (X + gap)
        y0 = text_h
        y1 = y0 + row_h + gap
        y2 = y1 + row_h + gap
        y3 = y2 + row_h + gap
        y4 = y3 + row_h + gap
        y5 = y4 + row_h + gap
        draw.text((x0 + 2, y0 + 2), f"t={t}", fill=(255, 255, 255))
        if use_score_bundle:
            draw.text((x0 + 2, y1 + 2), f"cand={stats_valid[t]} sel={stats_hi[t]} drop={stats_lo[t]}", fill=(255, 255, 255))
        else:
            draw.text((x0 + 2, y1 + 2), f"all={stats_valid[t]} hi={stats_hi[t]} lo={stats_lo[t]}", fill=(255, 255, 255))
        draw.text((x0 + 2, y2 + 2), f"hi={stats_hi[t]}", fill=(255, 255, 255))
        draw.text((x0 + 2, y3 + 2), f"hi(class)={stats_hi[t]}", fill=(255, 255, 255))
        draw.text((x0 + 2, y4 + 2), f"matched={stats_matched[t]}", fill=(255, 255, 255))
        draw.text((x0 + 2, y5 + 2), "GT class", fill=(255, 255, 255))

        for yy in (y0, y1, y2, y3, y4, y5):
            draw.rectangle(
                [x0, yy, x0 + X - 1, yy + row_h - 1],
                outline=(255, 255, 255),
                width=1,
            )

    legend_y = text_h + num_rows * row_h + (num_rows - 1) * gap + 4
    draw.text((2, legend_y), "Legend:", fill=(255, 255, 255))
    generic_h = draw_generic_vis_legend(
        draw, x_start=60, y_start=legend_y, canvas_w=canvas_w, conf_thr=conf_thr,
        matched_color=tuple(int(v) for v in matched_color.tolist()),
        ego_color=tuple(int(v) for v in ego_color.tolist()),
        gaussian_label=gaussian_desc,
    )
    draw_query_vis_legend(
        draw, x_start=60, y_start=legend_y + generic_h + 6, canvas_w=canvas_w,
        class_ids=vis_cfg.class_ids, class_names=vis_cfg.class_names,
    )

    os.makedirs(vis_cfg.vis_dir, exist_ok=True)
    img.save(os.path.join(vis_cfg.vis_dir, f"iter_{int(step):06d}_prob.png"))
