import math

import torch

from projects.occ_plugin.occupancy.dense_heads.query_head import QueryHead
from projects.occ_plugin.occupancy.detectors.utils_gt_prep import EfficientOCFGTPrepMixin
from projects.occ_plugin.occupancy.detectors.utils_instance_img_debug import EfficientOCFInstanceImgDebugMixin
from projects.occ_plugin.occupancy.detectors.utils_loss import EfficientOCFLossMixin
from projects.occ_plugin.occupancy.detectors.utils_matcher import EfficientOCFMatcherMixin


class _DummyDetector(
    EfficientOCFMatcherMixin,
    EfficientOCFLossMixin,
    EfficientOCFInstanceImgDebugMixin,
    EfficientOCFGTPrepMixin,
):
    def __init__(self):
        self.time_receptive_field = 3
        self.n_future_frames = 4
        self.n_future_frames_plus = 5
        self.query_present_only = True
        self.query_present_global_idx = 2
        self.query_center_match_cost_weight = 1.0
        self.query_bev_iou_match_cost_weight = 0.0
        self.query_traj_residual_max_m = (15.0, 15.0)
        self.spatial_extent3d = (80.0, 80.0, 6.4)
        self.point_cloud_range = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4)
        self.query_attn_softargmax_tau = 1.0
        self.query_inst_depth_num_bins = 8
        self.query_inst_depth_range_mode = "custom"
        self.query_inst_depth_min = 1.0
        self.query_inst_depth_max = 9.0
        self.query_attn_cam_frame_mode = "overlap_only"


def _make_future_egomotion(seq_len=7, dx_step=1.0):
    ego = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(seq_len, 1, 1)
    for idx in range(seq_len):
        ego[idx, 0, 3] = dx_step
    return ego


def _make_query_match_inputs(t_hist=3, img_hw=32, feat_hw=8):
    rots = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(t_hist, 1, 1, 1)
    trans = torch.zeros((t_hist, 1, 3), dtype=torch.float32)
    intrins = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(t_hist, 1, 1, 1)
    intrins[:, 0, 0, 0] = 1.0
    intrins[:, 0, 1, 1] = 1.0
    post_rots = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(t_hist, 1, 1, 1)
    post_trans = torch.zeros((t_hist, 1, 3), dtype=torch.float32)
    context = torch.zeros((t_hist, 1, 4, feat_hw, feat_hw), dtype=torch.float32)
    return {
        "context_seq_tnchw": context,
        "rots_tn33": rots,
        "trans_tn3": trans,
        "intrins_tn33": intrins,
        "post_rots_tn33": post_rots,
        "post_trans_tn3": post_trans,
        "img_h": img_hw,
        "img_w": img_hw,
        "feat_h": feat_hw,
        "feat_w": feat_hw,
        "frame_start_idx": 0,
        "frame_end_idx_exclusive": t_hist,
        "n_cam": 1,
    }


def _test_full7_match_and_center_cost():
    det = _DummyDetector()
    centers_full = torch.zeros((7, 2, 3), dtype=torch.float32)
    gt_centers_full = torch.zeros((7, 2, 3), dtype=torch.float32)
    gt_valid_full = torch.ones((7, 2), dtype=torch.bool)
    gt_ids = torch.tensor([10, 20], dtype=torch.long)

    hist_offsets = [0.5, 1.0, 1.5]
    future_offsets = [50.0, 60.0, 70.0, 80.0]
    for t_idx, off in enumerate(hist_offsets + future_offsets):
        centers_full[t_idx, 0, 0] = off
        centers_full[t_idx, 1, 0] = 10.0 + off
        gt_centers_full[t_idx, 0, 0] = off
        gt_centers_full[t_idx, 1, 0] = 10.0 + off
    centers_full[3:, 0, 0] += 1000.0
    centers_full[3:, 1, 0] -= 1000.0

    match = det._match_queries_to_gt_instances(
        centers_world=centers_full,
        gt_inst_center_world_tn3=gt_centers_full,
        gt_inst_center_valid_tn=gt_valid_full,
        gt_inst_ids_n=gt_ids,
        query_temporal_cost_frame_indices=[0, 1, 2],
    )
    assert torch.equal(match["matched_query_idx"], torch.tensor([0, 1]))
    assert tuple(match["gt_centers_tn3"].shape) == (7, 2, 3)
    center_cost_qn = match["cost_center_qn"]
    assert center_cost_qn is not None
    assert float(center_cost_qn[0, 0].item()) < 1e-6
    assert float(center_cost_qn[1, 1].item()) < 1e-6
    assert float(center_cost_qn[0, 1].item()) > 0.05


def _test_full7_gt_packing():
    det = _DummyDetector()
    hist_centers = torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3)
    hist_valid = torch.ones((3, 2), dtype=torch.bool)
    hist_ids = torch.tensor([20, 10], dtype=torch.long)
    traj_centers = (100.0 + torch.arange(5 * 2 * 3, dtype=torch.float32)).reshape(5, 2, 3)
    traj_valid = torch.ones((5, 2), dtype=torch.bool)
    traj_ids = torch.tensor([10, 20], dtype=torch.long)
    full_centers, full_valid, full_ids = det._build_full_query_instance_centers_from_history_and_traj(
        history_centers_tn3=hist_centers,
        history_valid_tn=hist_valid,
        history_ids_n=hist_ids,
        traj_centers_tn3=traj_centers,
        traj_valid_tn=traj_valid,
        traj_ids_n=traj_ids,
    )
    assert tuple(full_centers.shape) == (7, 2, 3)
    assert tuple(full_valid.shape) == (7, 2)
    assert torch.equal(full_ids, torch.tensor([10, 20]))
    assert torch.allclose(full_centers[0, 0], hist_centers[0, 1])
    assert torch.allclose(full_centers[2, 0], traj_centers[0, 0])


def _test_depth_and_attn_targets_use_three_history_frames():
    det = _DummyDetector()
    future_egomotion = _make_future_egomotion()
    query_match_inputs = _make_query_match_inputs()
    gt_centers = torch.tensor(
        [
            [[2.0, 0.0, 4.0]],
            [[3.0, 0.0, 4.0]],
            [[4.0, 0.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    gt_valid = torch.ones((3, 1), dtype=torch.bool)
    gt_ids = torch.tensor([7], dtype=torch.long)
    depth_pack = det.build_query_inst_depth_targets(
        gt_inst_center_world_tn3=gt_centers,
        gt_inst_center_valid_tn=gt_valid,
        gt_inst_ids_n=gt_ids,
        future_egomotion=future_egomotion,
        query_match_inputs=query_match_inputs,
    )
    assert depth_pack is not None
    assert tuple(depth_pack["gt_inst_depth_bin_tcn"].shape[:1]) == (3,)
    assert tuple(depth_pack["query_output_global_frame_indices"].tolist()) == (0, 1, 2)

    seg = torch.zeros((3, 4, 4, 4), dtype=torch.long)
    seg[:, 2, 2, 2] = 7
    attn_pack = det.build_query_attn_bbox_targets(
        segmentation_instance3d_txyz=seg,
        future_egomotion=future_egomotion,
        gt_inst_ids_n=gt_ids,
        query_match_inputs=query_match_inputs,
    )
    assert attn_pack is not None
    assert tuple(attn_pack["gt_inst_mask_tnhw"].shape[:2]) == (3, 1)


def _test_projection_roundtrip():
    det = _DummyDetector()
    future_egomotion = _make_future_egomotion()
    query_match_inputs = _make_query_match_inputs()
    point_present = torch.tensor([[4.0, 0.5, 4.0]], dtype=torch.float32)
    frame_indices = [0, 1, 2]
    lidar_t_to_present = det._build_lidar_frame_to_present(
        future_egomotion,
        frame_indices=frame_indices,
        present_global_idx=2,
    )
    for t_idx in range(3):
        tf_inv = torch.inverse(lidar_t_to_present[t_idx]).to(torch.float32)
        point_h = torch.cat([point_present, torch.ones((1, 1), dtype=torch.float32)], dim=1)
        point_lidar_t = (tf_inv @ point_h.t()).t()[:, :3]
        uv, depth, valid = det._project_points_to_aug_uv(
            points_lidar=point_lidar_t,
            rot=query_match_inputs["rots_tn33"][t_idx, 0],
            trans=query_match_inputs["trans_tn3"][t_idx, 0],
            intrins=query_match_inputs["intrins_tn33"][t_idx, 0],
            post_rot=query_match_inputs["post_rots_tn33"][t_idx, 0],
            post_trans=query_match_inputs["post_trans_tn3"][t_idx, 0],
        )
        assert bool(valid[0].item())
        uv1 = torch.cat([uv, torch.ones((1, 1), dtype=torch.float32)], dim=1)
        cam_ray = (torch.inverse(query_match_inputs["intrins_tn33"][t_idx, 0]) @ uv1.t()).t()
        cam_xyz = cam_ray * depth.view(1, 1)
        lifted_lidar_t = (
            query_match_inputs["rots_tn33"][t_idx, 0] @ cam_xyz.t()
        ).t() + query_match_inputs["trans_tn3"][t_idx, 0].view(1, 3)
        lifted_h = torch.cat([lifted_lidar_t, torch.ones((1, 1), dtype=torch.float32)], dim=1)
        lifted_present = (lidar_t_to_present[t_idx] @ lifted_h.t()).t()[:, :3]
        assert torch.allclose(lifted_present, point_present, atol=1e-4)


def _test_query_head_present_index():
    net = QueryHead(
        embed_dim=16,
        num_past_frames=3,
        num_future_frames=5,
        query_present_only=True,
        query_traj_num_steps=4,
        num_query_classes=3,
        point_cloud_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        spatial_extent3d=(80.0, 80.0, 6.4),
        query_depth_num_bins=8,
    )
    query_inst = torch.randn(3, 5, 16)
    out = net(query_inst, return_query_feats=True, compute_direct_center=False)
    assert int(out["query_present_local_idx"]) == 2
    assert tuple(out["query_feat_tqd"].shape) == (3, 5, 16)
    assert tuple(out["traj_offsets_fq2"].shape) == (4, 5, 2)


def main():
    _test_full7_match_and_center_cost()
    _test_full7_gt_packing()
    _test_depth_and_attn_targets_use_three_history_frames()
    _test_projection_roundtrip()
    _test_query_head_present_index()
    print("smoke_past_frame_matching: OK")


if __name__ == "__main__":
    main()
