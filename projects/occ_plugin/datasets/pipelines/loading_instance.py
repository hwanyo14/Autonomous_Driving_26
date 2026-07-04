# Developed by Jingyi Xu based on the codebase of Cam4DOcc and PowerBEV 
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF

import numpy as np
from mmdet.datasets.builder import PIPELINES
import os
import torch
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
import time
import copy

@PIPELINES.register_module()
class LoadInstanceWithFlow(object):
    def __init__(self, ocf_dataset_path, grid_size=[512, 512, 40], pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], background=0, use_flow=True, use_separate_classes=False, use_lyft=False, validate_cache=False, write_cache=True, load_segmentation_bev=True, load_segmentation_instance3d=False, segmentation_instance3d_path=None, segmentation_instance3d_key='segmentation_instance_saved_list2', load_segmentation_cls_instance3d=False, segmentation_cls_dataset_path=None, segmentation_cls_key='segmentation_saved_list2', validate_segmentation_cls_instance3d_alignment=False, load_gt_occ_inst=False, gt_occ_inst_dataset_path=None, gt_occ_inst_key='segmentation_instance_saved_list2', exclude_occ_class_ids=()):
        '''
        Loading sequential occupancy labels and instance flows for training and testing
        '''
        self.ocf_dataset_path = ocf_dataset_path
        self.pc_range = pc_range
        self.resolution = [(self.pc_range[3+i] - self.pc_range[i])/grid_size[i] for i in range(len(self.pc_range[:3]))]
        self.start_position = [self.pc_range[i] + self.resolution[i] / 2.0 for i in range(len(self.pc_range[:3]))]
        self.dimension = grid_size
        self.pc_range = np.array(self.pc_range)
        self.resolution = np.array(self.resolution)
        self.start_position = np.array(self.start_position)
        self.dimension = np.array(self.dimension)
        self.background = background
        self.use_flow = use_flow
        self.use_separate_classes = use_separate_classes
        self.use_lyft = use_lyft
        self.validate_cache = bool(validate_cache)
        self.write_cache = bool(write_cache)
        self.load_segmentation_bev = bool(load_segmentation_bev)
        self.load_segmentation_instance3d = bool(load_segmentation_instance3d)
        self.segmentation_instance3d_path = segmentation_instance3d_path
        self.segmentation_instance3d_key = str(segmentation_instance3d_key)
        self.load_segmentation_cls_instance3d = bool(load_segmentation_cls_instance3d)
        self.segmentation_cls_dataset_path = segmentation_cls_dataset_path
        self.segmentation_cls_key = str(segmentation_cls_key)
        self.strict_cls_instance3d_alignment = bool(validate_segmentation_cls_instance3d_alignment)
        self.load_gt_occ_inst = bool(load_gt_occ_inst)
        self.gt_occ_inst_dataset_path = gt_occ_inst_dataset_path
        self.gt_occ_inst_key = str(gt_occ_inst_key)
        self.exclude_occ_class_ids = tuple(int(v) for v in exclude_occ_class_ids)
        if self.load_segmentation_cls_instance3d and (not self.load_segmentation_instance3d):
            raise ValueError(
                "load_segmentation_cls_instance3d=True requires "
                "load_segmentation_instance3d=True."
            )

    def resolve_segmentation_instance3d_dir(self, prefix):
        candidates = []
        if self.segmentation_instance3d_path is not None:
            candidates.append(self.segmentation_instance3d_path)

        candidates.append(os.path.join(self.ocf_dataset_path, prefix, "segmentation_instance3d"))
        candidates.append(os.path.join("/mnt/hdd8/datasets/efficientocf", prefix, "segmentation_instance3d"))

        for c in candidates:
            if c is not None and os.path.isdir(c):
                return c
        return None

    def resolve_segmentation_cls_dir(self, prefix):
        if self.segmentation_cls_dataset_path is None:
            return None

        base = os.path.normpath(self.segmentation_cls_dataset_path)
        candidates = []
        if os.path.basename(base) == "segmentation":
            candidates.append(self.segmentation_cls_dataset_path)
        candidates.append(os.path.join(self.segmentation_cls_dataset_path, prefix, "segmentation"))
        candidates.append(os.path.join(self.segmentation_cls_dataset_path, "segmentation"))

        for c in candidates:
            if c is not None and os.path.isdir(c):
                return c
        return None

    def resolve_gt_occ_inst_dir(self, prefix):
        if self.gt_occ_inst_dataset_path is None:
            return None

        base = os.path.normpath(self.gt_occ_inst_dataset_path)
        candidates = []
        if os.path.basename(base) == "segmentation_instance3d":
            candidates.append(self.gt_occ_inst_dataset_path)
        candidates.append(os.path.join(self.gt_occ_inst_dataset_path, prefix, "segmentation_instance3d"))
        candidates.append(os.path.join(self.gt_occ_inst_dataset_path, "segmentation_instance3d"))

        for c in candidates:
            if c is not None and os.path.isdir(c):
                return c
        return None

    def _normalize_sparse_rows(self, rows, min_cols, label):
        arr = np.asarray(rows)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            if arr.size == 0:
                arr = np.zeros((0, min_cols), dtype=np.int64)
            else:
                arr = np.vstack(arr)
        arr = np.asarray(arr, dtype=np.int64)
        if arr.size == 0:
            return np.zeros((0, min_cols), dtype=np.int64)
        if arr.ndim != 2 or arr.shape[1] < min_cols:
            raise ValueError(f"{label} row shape is invalid: {arr.shape}")
        return arr

    def _sort_rows_by_xyz(self, rows):
        if rows.shape[0] <= 1:
            return rows
        order = np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))
        return rows[order]

    def _filter_sparse_rows_by_class(self, rows, class_col):
        arr = np.asarray(rows, dtype=np.int64)
        exclude_occ_class_ids = tuple(int(v) for v in getattr(self, "exclude_occ_class_ids", ()))
        if arr.size == 0 or len(exclude_occ_class_ids) <= 0:
            return arr
        if arr.ndim != 2 or arr.shape[1] <= int(class_col):
            return arr
        keep = ~np.isin(arr[:, int(class_col)], np.asarray(exclude_occ_class_ids, dtype=np.int64))
        return arr[keep]

    def _coords_to_struct(self, xyz):
        xyz = np.asarray(xyz, dtype=np.int64)
        if xyz.size == 0:
            return np.zeros((0,), dtype=[('x', np.int64), ('y', np.int64), ('z', np.int64)])
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"coords must be [N,3], got {xyz.shape}")
        return np.ascontiguousarray(xyz).view(
            dtype=[('x', np.int64), ('y', np.int64), ('z', np.int64)]
        ).reshape(-1)

    def _validate_sparse_rows_basic(self, rows, label, sample_key, frame_idx):
        if rows.shape[0] <= 0:
            return

        xyz = rows[:, :3]
        dim_xyz = self.dimension[:3].reshape(1, 3)
        oob_mask = (xyz < 0).any(axis=1) | (xyz >= dim_xyz).any(axis=1)
        if bool(np.any(oob_mask)):
            preview = xyz[oob_mask][:5].tolist()
            raise ValueError(
                f"{label} has out-of-bounds voxel coords for sample={sample_key}, "
                f"frame={frame_idx}: preview={preview}, grid_size={self.dimension[:3].tolist()}"
            )

        uniq_xyz, counts = np.unique(xyz, axis=0, return_counts=True)
        dup_xyz = uniq_xyz[counts > 1]
        if dup_xyz.shape[0] > 0:
            preview = dup_xyz[:5].tolist()
            raise ValueError(
                f"{label} has duplicate voxel coords for sample={sample_key}, "
                f"frame={frame_idx}: preview={preview}"
            )

    def _validate_segmentation_cls_instance3d_alignment(
        self,
        segmentation_instance3d_sparse_list,
        segmentation_cls_sparse_list,
        sample_key,
    ):
        if (not self.strict_cls_instance3d_alignment):
            return
        if segmentation_instance3d_sparse_list is None:
            raise ValueError(
                "segmentation_instance3d sparse list is required for strict "
                "segmentation_cls_instance3d alignment validation."
            )
        if segmentation_cls_sparse_list is None:
            raise ValueError(
                "segmentation_cls sparse list is required for strict "
                "segmentation_cls_instance3d alignment validation."
            )
        if len(segmentation_instance3d_sparse_list) != len(segmentation_cls_sparse_list):
            raise ValueError(
                "segmentation_instance3d and segmentation_cls sequence lengths differ for "
                f"sample={sample_key}: {len(segmentation_instance3d_sparse_list)} vs "
                f"{len(segmentation_cls_sparse_list)}"
            )

        for frame_idx, (inst_rows_raw, cls_rows_raw) in enumerate(
            zip(segmentation_instance3d_sparse_list, segmentation_cls_sparse_list)
        ):
            inst_rows = self._normalize_sparse_rows(
                inst_rows_raw,
                min_cols=4,
                label="segmentation_instance3d",
            )
            cls_rows = self._normalize_sparse_rows(
                cls_rows_raw,
                min_cols=4,
                label="segmentation_cls",
            )

            self._validate_sparse_rows_basic(
                inst_rows,
                label="segmentation_instance3d",
                sample_key=sample_key,
                frame_idx=frame_idx,
            )
            self._validate_sparse_rows_basic(
                cls_rows,
                label="segmentation_cls",
                sample_key=sample_key,
                frame_idx=frame_idx,
            )

            inst_xyz = self._sort_rows_by_xyz(inst_rows)[:, :3]
            cls_xyz = self._sort_rows_by_xyz(cls_rows)[:, :3]
            # Exact per-frame coordinate equality is too strict for the current
            # training path because query cls supervision aggregates labels over
            # the temporal horizon and uses only voxels that actually carry a
            # semantic class. Here we validate cache well-formedness only; the
            # model path performs the final instance-level supervision check.

    def _validate_gt_occ_inst_sparse_list(self, sparse_list, sample_key, expected_seq_len):
        if sparse_list is None:
            raise ValueError(f"gt_occ_inst sparse list is missing for sample={sample_key}")
        if len(sparse_list) != int(expected_seq_len):
            raise ValueError(
                f"gt_occ_inst sequence length mismatch for sample={sample_key}: "
                f"{len(sparse_list)} vs {int(expected_seq_len)}"
            )
        for frame_idx, rows_raw in enumerate(sparse_list):
            rows = self._normalize_sparse_rows(
                rows_raw,
                min_cols=5,
                label="gt_occ_inst",
            )
            self._validate_sparse_rows_basic(
                rows,
                label="gt_occ_inst",
                sample_key=sample_key,
                frame_idx=frame_idx,
            )

    def get_poly_region(self, instance_annotation, present_egopose, present_ego2lidar):
        """
        Obtain the bounding box polygon of the instance
        """
        present_ego_translation, present_ego_rotation = present_egopose
        present_ego2lidar_translation, present_ego2lidar_rotation = present_ego2lidar

        box = Box(
            instance_annotation['translation'], instance_annotation['size'], Quaternion(instance_annotation['rotation'])
        )
        box.translate(present_ego_translation)
        box.rotate(present_ego_rotation)

        box.translate(present_ego2lidar_translation)
        box.rotate(present_ego2lidar_rotation)
        pts=box.corners().T

        X_min_box = pts.min(axis=0)[0]
        X_max_box = pts.max(axis=0)[0]
        Y_min_box = pts.min(axis=0)[1]
        Y_max_box = pts.max(axis=0)[1]
        Z_min_box = pts.min(axis=0)[2]
        Z_max_box = pts.max(axis=0)[2]

        if self.pc_range[0] <= X_min_box and X_max_box <= self.pc_range[3] \
                and self.pc_range[1] <= Y_min_box and Y_max_box <= self.pc_range[4] \
                and self.pc_range[2] <= Z_min_box and Z_max_box <= self.pc_range[5]:
            pts = np.round((pts - self.start_position[:3] + self.resolution[:3] / 2.0) / self.resolution[:3]).astype(np.int32)

            return pts
        else:
            return None

    def fill_occupancy(self, occ_instance, occ_segmentation, occ_attribute_label, instance_fill_info):
        x_grid = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.float)
        x_grid = x_grid.view(self.dimension[0], 1, 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        y_grid = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.float)
        y_grid = y_grid.view(1, self.dimension[1], 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        z_grid = torch.linspace(0, self.dimension[2]-1, self.dimension[2], dtype=torch.float)
        z_grid = z_grid.view(1, 1, self.dimension[2]).expand(self.dimension[0], self.dimension[1], self.dimension[2])
        mesh_grid_3d = torch.stack((x_grid, y_grid, z_grid), -1)
        mesh_grid_3d = mesh_grid_3d.view(-1, 3)

        occ_instance = torch.from_numpy(occ_instance).view(-1, 1)
        occ_segmentation = torch.from_numpy(occ_segmentation).view(-1, 1)
        occ_attribute_label = torch.from_numpy(occ_attribute_label).view(-1, 1)
        occ_height = copy.deepcopy(occ_instance)

        for instance_info in instance_fill_info:
            poly_region_pts = instance_info['poly_region']
            semantic_id = instance_info['semantic_id']
            instance_id = instance_info['instance_id']
            attribute_label=instance_info['attribute_label']

            X_min_box = poly_region_pts.min(axis=0)[0]
            X_max_box = poly_region_pts.max(axis=0)[0]
            Y_min_box = poly_region_pts.min(axis=0)[1]
            Y_max_box = poly_region_pts.max(axis=0)[1]
            Z_min_box = poly_region_pts.min(axis=0)[2]
            Z_max_box = poly_region_pts.max(axis=0)[2]

            mask_cur_instance = (mesh_grid_3d[:,0] >= X_min_box) & (X_max_box >= mesh_grid_3d[:,0]) \
                                & (mesh_grid_3d[:,1] >= Y_min_box) & (Y_max_box >= mesh_grid_3d[:,1]) \
                                & (mesh_grid_3d[:,2] >= Z_min_box) & (Z_max_box >= mesh_grid_3d[:,2])
            occ_instance[mask_cur_instance] = instance_id
            occ_segmentation[mask_cur_instance] = semantic_id
            occ_attribute_label[mask_cur_instance] = attribute_label
            occ_height[mask_cur_instance] = Z_max_box
        
        occ_instance = occ_instance.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()
        occ_segmentation = occ_segmentation.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()
        occ_attribute_label = occ_attribute_label.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()
        occ_height = occ_height.view(self.dimension[0], self.dimension[1], self.dimension[2]).long()

        if self.use_lyft:
            occ_height = torch.max(occ_height, -1).values

        return occ_instance, occ_segmentation, occ_attribute_label, occ_height

    def get_label(self, input_seq_data):
        """
        Generate labels for semantic segmentation, instance segmentation, z position, attribute from the raw data of nuScenes
        """
        timestep = self.counter
        # Background is ID 0
        segmentation = np.ones((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
        instance = np.ones((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
        attribute_label = np.ones((self.dimension[0], self.dimension[1], self.dimension[2]))  * self.background
        
        instance_dict = input_seq_data['instance_dict']
        egopose_list = input_seq_data['egopose_list']
        ego2lidar_list = input_seq_data['ego2lidar_list']
        time_receptive_field = input_seq_data['time_receptive_field']

        instance_fill_info = []
        
        for instance_token, instance_annotation in instance_dict.items():
            if timestep not in instance_annotation['timestep']:
                continue
            pointer = instance_annotation['timestep'].index(timestep)
            annotation = {
                'translation': instance_annotation['translation'][pointer],
                'rotation': instance_annotation['rotation'][pointer],
                'size': instance_annotation['size'],
            }
            
            poly_region = self.get_poly_region(annotation, egopose_list[time_receptive_field - 1], ego2lidar_list[time_receptive_field - 1]) 

            if isinstance(poly_region, np.ndarray):
                if self.counter >= time_receptive_field and instance_token not in self.visible_instance_set:
                    continue
                self.visible_instance_set.add(instance_token)

                prepare_for_fill = dict(
                    poly_region=poly_region,
                    instance_id=instance_annotation['instance_id'],
                    attribute_label=instance_annotation['attribute_label'][pointer],
                    semantic_id=instance_annotation['semantic_id'],
                )

                instance_fill_info.append(prepare_for_fill)

        instance, segmentation, attribute_label, bbox_height = self.fill_occupancy(instance, segmentation, attribute_label, instance_fill_info)

        segmentation = segmentation.unsqueeze(0)
        instance = instance.unsqueeze(0)
        attribute_label = attribute_label.unsqueeze(0).unsqueeze(0)
        bbox_height = bbox_height.unsqueeze(0)

        return segmentation, instance, attribute_label, bbox_height

    @staticmethod
    def generate_flow(flow, occ_instance_bev_seq, instance, instance_id):
        """
        Generate ground truth for the flow of each instance based on instance segmentation
        """
        seg_len, wx, wy = occ_instance_bev_seq.shape
        ratio = 4
        occ_instance_bev_seq = occ_instance_bev_seq.reshape(seg_len, wx//ratio, ratio, wy//ratio, ratio).permute(0,1,3,2,4).reshape(seg_len, wx//ratio, wy//ratio, ratio**2)
        empty_mask = occ_instance_bev_seq.sum(-1) == 0
        occ_instance_bev_seq = occ_instance_bev_seq.to(torch.int64)
        occ_space = occ_instance_bev_seq[~empty_mask]
        occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1 
        occ_instance_bev_seq[~empty_mask] = occ_space
        occ_instance_bev_seq = torch.mode(occ_instance_bev_seq, dim=-1)[0]
        occ_instance_bev_seq[occ_instance_bev_seq<0] = 0
        occ_instance_bev_seq = occ_instance_bev_seq.long()

        _, wx, wy = occ_instance_bev_seq.shape
        x, y = torch.meshgrid(torch.arange(wx, dtype=torch.float), torch.arange(wy, dtype=torch.float))
        
        grid = torch.stack((x, y), dim=0)

        # Set the first frame
        init_pointer = instance['timestep'][0]
        instance_mask = (occ_instance_bev_seq[init_pointer] == instance_id)

        flow[init_pointer, 0, instance_mask] = grid[0, instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
        flow[init_pointer, 1, instance_mask] = grid[1, instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        for i, timestep in enumerate(instance['timestep']):
            if i == 0:
                continue

            instance_mask = (occ_instance_bev_seq[timestep] == instance_id)
            prev_instance_mask = (occ_instance_bev_seq[timestep-1] == instance_id)
            if instance_mask.sum() == 0 or prev_instance_mask.sum() == 0:
                continue

            flow[timestep, 0, instance_mask] = grid[0, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[0, instance_mask]
            flow[timestep, 1, instance_mask] = grid[1, prev_instance_mask].mean(dim=0, keepdim=True).round() - grid[1, instance_mask]

        return flow

    def get_flow_label(self, input_seq_data, ignore_index=255):
        """
        Generate the global map of the flow ground truth
        """
        occ_instance_bev = input_seq_data['instance_bev']
        instance_dict = input_seq_data['instance_dict']
        instance_map = input_seq_data['instance_map']

        seq_len, wx, wy = occ_instance_bev.shape
        ratio = 4
        flow = ignore_index * torch.ones(seq_len, 2, wx//ratio, wy//ratio)
        
        # ignore flow generation for faster pipelines
        if not self.use_flow:
            return flow

        for token, instance in instance_dict.items():
            flow = self.generate_flow(flow, occ_instance_bev, instance, instance_map[token])
        return flow.float()

    # set ignore index to 0 for vis
    @staticmethod
    def convert_instance_mask_to_center_and_offset_label(input_seq_data, ignore_index=255, sigma=3):
        occ_instance = input_seq_data['instance']
        num_instances=len(input_seq_data['instance_map'])

        seq_len, wx, wy, wz = occ_instance.shape
        center_label = torch.zeros(seq_len, 1, wx, wy, wz)
        offset_label = ignore_index * torch.ones(seq_len, 3, wx, wy, wz)
        # x is vertical displacement, y is horizontal displacement
        x, y, z = torch.meshgrid(torch.arange(wx, dtype=torch.float), torch.arange(wy, dtype=torch.float), torch.arange(wz, dtype=torch.float))
        
        # Ignore id 0 which is the background
        for instance_id in range(1, num_instances+1):
            for t in range(seq_len):
                instance_mask = (occ_instance[t] == instance_id)

                xc = x[instance_mask].mean().round().long()
                yc = y[instance_mask].mean().round().long()
                zc = z[instance_mask].mean().round().long()

                off_x = xc - x
                off_y = yc - y
                off_z = zc - z

                g = torch.exp(-(off_x ** 2 + off_y ** 2 + off_z ** 2) / sigma ** 2)
                center_label[t, 0] = torch.maximum(center_label[t, 0], g)
                offset_label[t, 0, instance_mask] = off_x[instance_mask]
                offset_label[t, 1, instance_mask] = off_y[instance_mask]
                offset_label[t, 2, instance_mask] = off_z[instance_mask]

        return center_label, offset_label

    def get_segmentation_bev(self, segmentation):
        segmentation = segmentation[0]
        x_voxel,y_voxel,z_voxel = segmentation.shape
        segmentation_bev = segmentation.sum(-1)
        segmentation_bev[segmentation_bev!=0] = 1
        return segmentation_bev

    def get_instance_bev(self, instance):
        instance = instance[0]
        x_voxel,y_voxel,z_voxel = instance.shape
        instance_bev = torch.zeros((x_voxel,y_voxel))

        for x in range(x_voxel):
            for y in range(y_voxel):
                sum_height = instance[x,y,:].view(-1)
                sum_height = torch.sum(sum_height)
                if sum_height != 0:
                    instance_bev[x,y] = self.non_zero_nbr(instance[x,y,:])
        return instance_bev

    def non_zero_nbr(self, instance_cur_xy):
        max_index = 0
        max_value = 0
        for i in range(instance_cur_xy.shape[0]):
            if instance_cur_xy[i] != 0:
                flag = 0
                for j in range(i+1,instance_cur_xy.shape[0]):
                    if instance_cur_xy[i] == instance_cur_xy[j]:
                        flag += 1
                if flag > max_index:
                    max_index = flag
                    max_value = i
        return instance_cur_xy[max_value]
    
    def _npz_load_list(self, npz_path: str, key: str):
        with np.load(npz_path, allow_pickle=True) as f:
            if key in f.files:
                arr = f[key]
            else:
                # 과거 포맷(arr_0) 호환이 필요하면 남겨두세요
                arr = f["arr_0"]

        if isinstance(arr, np.ndarray) and arr.dtype == object:
            return arr.tolist()
        return list(arr)

    def _ensure_numeric_ndarray(self, x, dtype):
        # x가 object 배열이면 내부를 쌓아서 숫자 배열로 만듭니다
        if isinstance(x, np.ndarray) and x.dtype == object:
            x = np.vstack(x)
        return np.asarray(x, dtype=dtype)

    def atomic_savez(self, path_no_ext, **kwargs):
        final = path_no_ext + ".npz"
        tmp_base = final + f".tmp.{os.getpid()}.{time.time_ns()}"
        tmp_file = tmp_base + ".npz"
        try:
            np.savez(tmp_base, **kwargs)      # 실제 파일은 tmp_base + ".npz"로 생성됨
            os.replace(tmp_file, final)       # 최종 파일로 원자적 교체
        finally:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    def build_instance_center_world_targets(self, sparse_seq_rows):
        """
        Convert per-frame sparse instance rows into dense temporal center targets.

        Args:
            sparse_seq_rows: list[T] of arrays with shape [K, >=4]
                             expected columns: [x, y, z, ..., instance_id]
        Returns:
            centers_world: [T, N, 3] float32 (meters, world range scale)
            valid_mask:    [T, N] bool
            instance_ids:  [N] int64 (global id set over sequence)
        """
        frame_center_maps = []
        all_instance_ids = set()

        for rows in sparse_seq_rows:
            arr = np.asarray(rows)
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = np.vstack(arr) if arr.size > 0 else np.zeros((0, 5), dtype=np.int64)
            arr = np.asarray(arr, dtype=np.int64)

            if arr.size == 0:
                frame_center_maps.append({})
                continue
            if arr.ndim != 2 or arr.shape[1] < 4:
                raise ValueError(f"invalid sparse rows shape: {arr.shape}")

            xyz = arr[:, :3].astype(np.float32, copy=False)
            inst = arr[:, -1].astype(np.int64, copy=False)
            keep = inst > 0
            xyz = xyz[keep]
            inst = inst[keep]

            if inst.size == 0:
                frame_center_maps.append({})
                continue

            cur = {}
            for iid in np.unique(inst):
                pts = xyz[inst == iid]
                if pts.shape[0] == 0:
                    continue
                cur[int(iid)] = ((pts.min(axis=0) + pts.max(axis=0)) * 0.5).astype(np.float32, copy=False)

            frame_center_maps.append(cur)
            all_instance_ids.update(cur.keys())

        instance_ids = np.array(sorted(all_instance_ids), dtype=np.int64)
        t_len = len(frame_center_maps)
        n_inst = int(instance_ids.shape[0])

        centers_voxel = np.zeros((t_len, n_inst, 3), dtype=np.float32)
        valid_mask = np.zeros((t_len, n_inst), dtype=np.bool_)

        if n_inst > 0:
            id_to_col = {int(iid): idx for idx, iid in enumerate(instance_ids.tolist())}
            for t, c_map in enumerate(frame_center_maps):
                for iid, c_xyz in c_map.items():
                    col = id_to_col[int(iid)]
                    centers_voxel[t, col] = c_xyz
                    valid_mask[t, col] = True

        centers_world = centers_voxel * self.resolution[:3].reshape(1, 1, 3) + self.start_position[:3].reshape(1, 1, 3)
        centers_world = centers_world.astype(np.float32, copy=False)
        centers_world[~valid_mask] = 0.0

        return (
            torch.from_numpy(centers_world),
            torch.from_numpy(valid_mask),
            torch.from_numpy(instance_ids),
        )

    # def __call__(self, results):
    #     assert 'attribute_label' not in results.keys()
    #     assert 'segmentation_bev' not in results.keys()
    #     assert 'instance_bev' not in results.keys()
    #     assert 'flow_bev' not in results.keys()

    #     time_receptive_field = results['time_receptive_field']

    #     prefix = "MMO" if self.use_separate_classes else "GMO"

    #     if self.use_lyft:
    #         prefix = prefix + "_lyft"

    #     seg_label_dir = os.path.join(self.ocf_dataset_path, prefix, "segmentation")
    #     if not os.path.exists(seg_label_dir):
    #         os.mkdir(seg_label_dir)
    #     seg_label_path = os.path.join(seg_label_dir, \
    #         results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

    #     seg_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "segmentation_bev")
    #     if not os.path.exists(seg_bev_label_dir):
    #         os.mkdir(seg_bev_label_dir)
    #     seg_bev_label_path = os.path.join(seg_bev_label_dir, \
    #         results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

    #     instance_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "instance_bev")
    #     if not os.path.exists(instance_bev_label_dir):
    #         os.mkdir(instance_bev_label_dir)
    #     instance_bev_label_path = os.path.join(instance_bev_label_dir, \
    #         results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

    #     flow_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "flow_bev")
    #     if not os.path.exists(flow_bev_label_dir):
    #         os.mkdir(flow_bev_label_dir)        
    #     flow_bev_label_path = os.path.join(flow_bev_label_dir, \
    #         results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])


    #     # ========== data regen if the npz files are broken ==========
    #     need_regen = False

    #     check_list = [
    #         (seg_label_path + ".npz", "segmentation_saved_list2"),
    #         (seg_bev_label_path + ".npz", "segmentation_bev_saved_list2"),
    #         (instance_bev_label_path + ".npz", "instance_bev_saved_list2"),
    #         (flow_bev_label_path + ".npz", "flow_bev_saved_list2"),
    #     ]

    #     if self.use_lyft:
    #         check_list.append((pcd_height_label_path + ".npz", "pcd_height_saved_list2"))

    #     for p, key in check_list:
    #         if not os.path.exists(p):
    #             need_regen = True
    #             continue

    #         try:
    #             f = np.load(p, allow_pickle=True)
    #             files = list(f.files)
    #             _ = f[key]  # key 존재 및 로드 가능성 체크
    #             f.close()
    #         except Exception as e:
    #             print(f"[BAD_NPZ] {p} key={key} files={files if 'files' in locals() else 'N/A'} err={repr(e)}", flush=True)
    #             try:
    #                 os.remove(p)
    #             except Exception as _e:
    #                 print(f"[BAD_NPZ_REMOVE_FAIL] {p} err={repr(_e)}", flush=True)
    #             need_regen = True


    #     segmentation_list = []
    #     if (not need_regen) and os.path.exists(seg_label_path+".npz"):
    #         try:
    #             # gt_segmentation_arr = np.load(seg_label_path+".npz",allow_pickle=True)['arr_0']
    #             gt_segmentation_arr = np.load(seg_label_path+".npz",allow_pickle=True)
    #             gt_segmentation_arr = list(gt_segmentation_arr['segmentation_saved_list2'])
    #             for j in range(len(gt_segmentation_arr)):
    #                 segmentation = np.zeros((self.dimension[0], self.dimension[1], self.dimension[2])) * self.background
    #                 gt_segmentation = gt_segmentation_arr[j]
    #                 gt_segmentation = torch.from_numpy(gt_segmentation)
    #                 segmentation[gt_segmentation[:, 0].long(), gt_segmentation[:, 1].long(), gt_segmentation[:, 2].long()] = gt_segmentation[:, -1]
    #                 segmentation = torch.from_numpy(segmentation).unsqueeze(0)
    #                 segmentation_list.append(segmentation)
    #         except Exception as e:
    #             print(f"[BAD_NPZ] {seg_label_path}.npz err={repr(e)}", flush=True)
    #             try:
    #                 os.remove(seg_label_path + ".npz")
    #             except Exception:
    #                 pass
    #             need_regen = True

    #     segmentation_bev_list = []
    #     if (not need_regen) and os.path.exists(seg_bev_label_path+".npz"):
    #         try:
    #             # gt_segmentation_bev_arr = np.load(seg_bev_label_path+".npz",allow_pickle=True)['arr_0']
    #             gt_segmentation_bev_arr = np.load(seg_bev_label_path+".npz",allow_pickle=True)
    #             gt_segmentation_bev_arr = list(gt_segmentation_bev_arr['segmentation_bev_saved_list2'])
    #             for j in range(len(gt_segmentation_bev_arr)):
    #                 segmentation_bev = np.zeros((self.dimension[0], self.dimension[1])) * self.background
    #                 gt_segmentation_bev = gt_segmentation_bev_arr[j]
    #                 gt_segmentation_bev = torch.from_numpy(gt_segmentation_bev)
    #                 segmentation_bev[gt_segmentation_bev[:, 0].long(), gt_segmentation_bev[:, 1].long()] = gt_segmentation_bev[:, -1]
    #                 segmentation_bev = torch.from_numpy(segmentation_bev).unsqueeze(0)
    #                 segmentation_bev_list.append(segmentation_bev)
    #         except Exception as e:
    #             print(f"[BAD_NPZ] {seg_bev_label_path}.npz err={repr(e)}", flush=True)
    #             try:
    #                 os.remove(seg_label_path + ".npz")
    #             except Exception:
    #                 pass
    #             need_regen = True

    #     instance_bev_list = []
    #     if (not need_regen) and os.path.exists(instance_bev_label_path+".npz"):
    #         try:
    #             # gt_instance_bev_arr = np.load(instance_bev_label_path+".npz",allow_pickle=True)['arr_0']
    #             gt_instance_bev_arr = np.load(instance_bev_label_path+".npz",allow_pickle=True)
    #             gt_instance_bev_arr = list(gt_instance_bev_arr['instance_bev_saved_list2'])
    #             for j in range(len(gt_instance_bev_arr)):
    #                 instance_bev = np.ones((self.dimension[0], self.dimension[1])) * self.background
    #                 gt_instance_bev = gt_instance_bev_arr[j]
    #                 gt_instance_bev = torch.from_numpy(gt_instance_bev)
    #                 instance_bev[gt_instance_bev[:, 0].long(), gt_instance_bev[:, 1].long()] = gt_instance_bev[:, -1]
    #                 instance_bev = torch.from_numpy(instance_bev).unsqueeze(0)
    #                 instance_bev_list.append(instance_bev)
    #         except Exception as e:
    #             print(f"[BAD_NPZ] {instance_bev_label_path}.npz err={repr(e)}", flush=True)
    #             try:
    #                 os.remove(seg_label_path + ".npz")
    #             except Exception:
    #                 pass
    #             need_regen = True

    #     flow_bev_list = []
    #     if (not need_regen) and os.path.exists(flow_bev_label_path+".npz"):
    #         try:
    #             # gt_flow_bev_arr = np.load(flow_bev_label_path+".npz",allow_pickle=True)['arr_0']
    #             gt_flow_bev_arr = np.load(flow_bev_label_path+".npz",allow_pickle=True)
    #             gt_flow_bev_arr = list(gt_flow_bev_arr['flow_bev_saved_list2'])
    #             for j in range(len(gt_flow_bev_arr)):
    #                 flow_bev = np.ones((2, self.dimension[0]//4, self.dimension[1]//4)) * 255
    #                 gt_flow_bev = gt_flow_bev_arr[j]
    #                 gt_flow_bev = torch.from_numpy(gt_flow_bev)
    #                 flow_bev[:, gt_flow_bev[:, 0].long(), gt_flow_bev[:, 1].long()] = gt_flow_bev[:, 2:].permute(1, 0)
    #                 flow_bev = torch.from_numpy(flow_bev).unsqueeze(0)
    #                 flow_bev_list.append(flow_bev)
    #         except Exception as e:
    #             print(f"[BAD_NPZ] {flow_bev_label_path}.npz err={repr(e)}", flush=True)
    #             try:
    #                 os.remove(seg_label_path + ".npz")
    #             except Exception:
    #                 pass
    #             need_regen = True

    #     if self.use_lyft == True:
    #         pcd_height_label_dir = os.path.join(self.ocf_dataset_path, prefix, "pcd_height")
    #         if not os.path.exists(pcd_height_label_dir):
    #             os.mkdir(pcd_height_label_dir)        
    #         pcd_height_label_path = os.path.join(pcd_height_label_dir, \
    #             results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])

    #         pcd_height_list = []
    #         if (not need_regen) and os.path.exists(pcd_height_label_path+".npz"):
    #             try:
    #                 # gt_pcd_height_arr = np.load(pcd_height_label_path+".npz",allow_pickle=True)['arr_0']
    #                 gt_pcd_height_arr = np.load(pcd_height_label_path+".npz",allow_pickle=True)
    #                 gt_pcd_height_arr = list(gt_pcd_height_arr['pcd_height_saved_list2'])
    #                 for j in range(len(gt_pcd_height_arr)):
    #                     pcd_height = np.zeros((self.dimension[0], self.dimension[1])) * self.background
    #                     gt_pcd_height = gt_pcd_height_arr[j]
    #                     gt_pcd_height = torch.from_numpy(gt_pcd_height)
    #                     pcd_height[gt_pcd_height[:, 0].long(), gt_pcd_height[:, 1].long()] = gt_pcd_height[:, -1]
    #                     pcd_height = torch.from_numpy(pcd_height).unsqueeze(0)
    #                     pcd_height_list.append(pcd_height)
    #             except Exception as e:
    #                 print(f"[BAD_NPZ] {pcd_height_label_path}.npz err={repr(e)}", flush=True)
    #                 try:
    #                     os.remove(seg_label_path + ".npz")
    #                 except Exception:
    #                     pass
    #                 need_regen = True

    #     if os.path.exists(seg_label_path+".npz") and os.path.exists(seg_bev_label_path+".npz") and os.path.exists(instance_bev_label_path+".npz") and os.path.exists(flow_bev_label_path+".npz"):
    #         results['segmentation'] = torch.cat(segmentation_list, dim=0)
    #         # results['instance'] = torch.cat(instance_list, dim=0)
    #         results['attribute_label'] =  torch.from_numpy(np.zeros((self.dimension[0], self.dimension[1], self.dimension[2]))).unsqueeze(0)
    #         results['segmentation_bev'] = torch.cat(segmentation_bev_list, dim=0)
    #         results['instance_bev'] = torch.cat(instance_bev_list, dim=0)
    #         results['flow_bev'] = torch.cat(flow_bev_list, dim=0).float()

    #         if self.use_lyft == True:
    #             results['height'] = torch.cat(pcd_height_list, dim=0)
    #             for key, value in results.items():
    #                 if key in ['sample_token', 'centerness','flow_bev', 'instance', 'offset', 'time_receptive_field', "indices", \
    #                 'segmentation', 'segmentation_bev', 'instance_bev', 'height', 'attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
    #                     continue
    #                 results[key] = torch.cat(value, dim=0)
    #             return results

    #         for key, value in results.items():
    #             if key in ['sample_token', 'centerness', 'offset', 'flow_bev', 'time_receptive_field', "indices", \
    #                 'segmentation', 'segmentation_bev', 'instance_bev', 'attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
    #                 continue
    #             results[key] = torch.cat(value, dim=0)
    #         return results
        
    #     else:
    #         results['segmentation'] = []
    #         # results['instance'] = []
    #         results['attribute_label'] = []
    #         results['segmentation_bev'] = []
    #         results['instance_bev'] = []

    #         segmentation_saved_list = []
    #         # instance_saved_list = []
    #         segmentation_bev_saved_list = []
    #         instance_bev_saved_list = []

    #         if self.use_lyft:
    #             results['height'] = []
    #             pcd_height_saved_list = []

    #         sequence_length = results['sequence_length']
    #         self.visible_instance_set = set()
    #         for self.counter in range(sequence_length):
    #             segmentation, instance, attribute_label, bbox_height = self.get_label(results)
    #             segmentation_bev = self.get_segmentation_bev(segmentation)
    #             instance_bev = self.get_instance_bev(instance)
                
    #             results['segmentation'].append(segmentation)
    #             # results['instance'].append(instance)
    #             results['attribute_label'].append(attribute_label)
    #             results['segmentation_bev'].append(segmentation_bev.unsqueeze(0))
    #             results['instance_bev'].append(instance_bev.unsqueeze(0))

    #             x_grid = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
    #             x_grid = x_grid.view(self.dimension[0], 1, 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
    #             y_grid = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
    #             y_grid = y_grid.view(1, self.dimension[1], 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
    #             z_grid = torch.linspace(0, self.dimension[2]-1, self.dimension[2], dtype=torch.long)
    #             z_grid = z_grid.view(1, 1, self.dimension[2]).expand(self.dimension[0], self.dimension[1], self.dimension[2])
    #             segmentation_for_save = torch.stack((x_grid, y_grid, z_grid), -1)
    #             segmentation_for_save = segmentation_for_save.view(-1, 3)
    #             segmentation_label = segmentation.squeeze(0).view(-1,1)
    #             segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1)
    #             kept = segmentation_for_save[:,-1]!=0
    #             segmentation_for_save= segmentation_for_save[kept]
    #             segmentation_saved_list.append(segmentation_for_save)

    #             x_grid_bev = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
    #             x_grid_bev = x_grid_bev.view(self.dimension[0], 1).expand(self.dimension[0], self.dimension[1])
    #             y_grid_bev = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
    #             y_grid_bev = y_grid_bev.view(1, self.dimension[1]).expand(self.dimension[0], self.dimension[1])
                
    #             segmentation_bev_for_save = torch.stack((x_grid_bev, y_grid_bev), -1)
    #             segmentation_bev_for_save = segmentation_bev_for_save.view(-1, 2)
    #             segmentation_bev_label = segmentation_bev.unsqueeze(-1).view(-1,1)
    #             segmentation_bev_for_save = torch.cat((segmentation_bev_for_save, segmentation_bev_label), dim=-1)
    #             kept = segmentation_bev_for_save[:,-1]!=0
    #             segmentation_bev_for_save= segmentation_bev_for_save[kept]
    #             segmentation_bev_saved_list.append(segmentation_bev_for_save)

    #             instance_bev_for_save = torch.stack((x_grid_bev, y_grid_bev), -1)
    #             instance_bev_for_save = instance_bev_for_save.view(-1, 2)
    #             instance_bev_label = instance_bev.unsqueeze(-1).view(-1,1)
    #             instance_bev_for_save = torch.cat((instance_bev_for_save, instance_bev_label), dim=-1)
    #             kept = instance_bev_for_save[:,-1]!=0
    #             instance_bev_for_save= instance_bev_for_save[kept]
    #             instance_bev_saved_list.append(instance_bev_for_save)

    #             if self.use_lyft:
    #                 results['height'].append(bbox_height)
    #                 x_grid_bev = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
    #                 x_grid_bev = x_grid_bev.view(self.dimension[0], 1).expand(self.dimension[0], self.dimension[1])
    #                 y_grid_bev = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
    #                 y_grid_bev = y_grid_bev.view(1, self.dimension[1]).expand(self.dimension[0], self.dimension[1])

    #                 pcd_height_for_save = torch.stack((x_grid_bev, y_grid_bev), -1)
    #                 pcd_height_for_save = pcd_height_for_save.view(-1, 2)
    #                 pcd_height_label = bbox_height.squeeze(0).view(-1,1)
    #                 pcd_height_for_save = torch.cat((pcd_height_for_save, pcd_height_label), dim=-1)
    #                 kept = pcd_height_for_save[:,-1]!=0
    #                 pcd_height_for_save= pcd_height_for_save[kept]
    #                 pcd_height_saved_list.append(pcd_height_for_save)
            
    #         segmentation_saved_list2 = [item.cpu().detach().numpy() for item in segmentation_saved_list]
    #         segmentation_bev_saved_list2 = [item.cpu().detach().numpy() for item in segmentation_bev_saved_list]
    #         # instance_saved_list2 = [item.cpu().detach().numpy() for item in instance_saved_list]
    #         instance_bev_saved_list2 = [item.cpu().detach().numpy() for item in instance_bev_saved_list]

    #         # np.savez(seg_label_path, segmentation_saved_list2)
    #         # np.savez(seg_bev_label_path, segmentation_bev_saved_list2)
    #         # np.savez(instance_bev_label_path, instance_bev_saved_list2)

    #         # ========
    #         obj_seg = np.array(segmentation_saved_list2, dtype=object)
    #         # np.savez(seg_label_path, segmentation_saved_list2=obj_seg)
    #         self.atomic_savez(seg_label_path, segmentation_saved_list2=obj_seg)

    #         obj_seg_bev = np.array(segmentation_bev_saved_list2, dtype=object)
    #         # np.savez(seg_bev_label_path, segmentation_bev_saved_list2=obj_seg_bev)
    #         self.atomic_savez(seg_bev_label_path, segmentation_bev_saved_list2=obj_seg_bev)

    #         obj_inst_bev = np.array(instance_bev_saved_list2, dtype=object)
    #         # np.savez(instance_bev_label_path, instance_bev_saved_list2=obj_inst_bev)
    #         self.atomic_savez(instance_bev_label_path, instance_bev_saved_list2=obj_inst_bev)



    #         if self.use_lyft:
    #             pcd_height_saved_list2 = [item.cpu().detach().numpy() for item in pcd_height_saved_list]
    #             # np.savez(pcd_height_label_path, pcd_height_saved_list2)
    #             obj_pcd_height = np.array(pcd_height_saved_list2, dtype=object)
    #             # np.savez(pcd_height_label_path, pcd_height_saved_list2=obj_pcd_height)
    #             self.atomic_savez(pcd_height_label_path, pcd_height_saved_list2=obj_pcd_height)
    #             results['height'] = torch.cat(results['height'], dim=0)

    #         results['segmentation'] = torch.cat(results['segmentation'], dim=0)
    #         # results['instance'] = torch.cat(results['instance'], dim=0)
    #         results['attribute_label'] =  torch.from_numpy(np.zeros((self.dimension[0], self.dimension[1], self.dimension[2]))).unsqueeze(0)
    #         results['segmentation_bev'] = torch.cat(results['segmentation_bev'], dim=0)
    #         results['instance_bev'] = torch.cat(results['instance_bev'], dim=0)
    #         results['flow_bev'] = self.get_flow_label(results, ignore_index=255)
    #         flow_bev_saved_list = []
    #         sequence_length = results['sequence_length']
    #         d0 = self.dimension[0]//4
    #         d1 = self.dimension[1]//4 
    #         for cnt in range(sequence_length):
    #             flow_bev = results['flow_bev'][cnt, ...]
    #             x_grid = torch.linspace(0, d0-1, d0, dtype=torch.long)
    #             x_grid = x_grid.view(d0, 1).expand(d0, d1)
    #             y_grid = torch.linspace(0, d1-1, d1, dtype=torch.long)
    #             y_grid = y_grid.view(1, d1).expand(d0, d1)
    #             flow_bev_for_save = torch.stack((x_grid, y_grid), -1)
    #             flow_bev_for_save = flow_bev_for_save.view(-1, 2)
    #             flow_bev_label = flow_bev.permute(1,2,0).view(-1,2)
    #             flow_bev_for_save = torch.cat((flow_bev_for_save, flow_bev_label), dim=-1)
    #             kept = (flow_bev_for_save[:,-1]!=255) & (flow_bev_for_save[:,-2]!=255)
    #             flow_bev_for_save= flow_bev_for_save[kept]
    #             flow_bev_saved_list.append(flow_bev_for_save)

    #         flow_bev_saved_list2 = [item.cpu().detach().numpy() for item in flow_bev_saved_list]
    #         # np.savez(flow_bev_label_path, flow_bev_saved_list2)
    #         obj_flow_bev = np.array(flow_bev_saved_list2, dtype=object)
    #         # np.savez(flow_bev_label_path, flow_bev_saved_list2=obj_flow_bev)
    #         self.atomic_savez(flow_bev_label_path, flow_bev_saved_list2=obj_flow_bev)

    #         if self.use_lyft:
    #             for key, value in results.items():
    #                 if key in ['sample_token', 'centerness', 'offset', 'flow_bev', 'time_receptive_field', "indices", 'height',\
    #                 'segmentation','segmentation_bev', 'instance_bev', 'attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
    #                     continue
    #                 results[key] = torch.cat(value, dim=0)
    #         else:
    #             for key, value in results.items():
    #                 if key in ['sample_token', 'centerness', 'offset', 'flow_bev', 'time_receptive_field', "indices", \
    #                 'segmentation', 'segmentation_bev', 'instance_bev', 'attribute_label','sequence_length', 'instance_dict', 'instance_map', 'input_dict', 'egopose_list','ego2lidar_list','scene_token']:
    #                     continue
    #                 results[key] = torch.cat(value, dim=0)

    #     return results


    def __call__(self, results):
        assert 'attribute_label' not in results.keys()
        assert 'segmentation_bev' not in results.keys()
        assert 'instance_bev' not in results.keys()
        assert 'flow_bev' not in results.keys()
        assert 'segmentation_instance3d' not in results.keys()
        assert 'segmentation_cls_instance3d' not in results.keys()
        assert 'gt_occ_inst' not in results.keys()
        assert 'gt_instance_centers_world' not in results.keys()
        assert 'gt_instance_centers_valid' not in results.keys()
        assert 'gt_instance_ids' not in results.keys()

        time_receptive_field = results['time_receptive_field']

        prefix = "MMO" if self.use_separate_classes else "GMO"
        
        if self.use_lyft:
            prefix = prefix + "_lyft"

        sample_key = (
            results['input_dict'][time_receptive_field - 1]['scene_token']
            + "_"
            + results['input_dict'][time_receptive_field - 1]['lidar_token']
        )

        # ---------------- paths ----------------
        seg_label_dir = os.path.join(self.ocf_dataset_path, prefix, "segmentation")
        if not os.path.exists(seg_label_dir):
            os.mkdir(seg_label_dir)
        seg_label_path = os.path.join(
            seg_label_dir,
            sample_key
        )

        seg_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "segmentation_bev")
        if not os.path.exists(seg_bev_label_dir):
            os.mkdir(seg_bev_label_dir)
        seg_bev_label_path = os.path.join(
            seg_bev_label_dir,
            sample_key
        )

        instance_bev_label_path = None
        if self.use_flow:
            instance_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "instance_bev")
            if not os.path.exists(instance_bev_label_dir):
                os.mkdir(instance_bev_label_dir)
            instance_bev_label_path = os.path.join(
                instance_bev_label_dir,
                sample_key
            )

        flow_bev_label_path = None
        if self.use_flow:
            flow_bev_label_dir = os.path.join(self.ocf_dataset_path, prefix, "flow_bev")
            if not os.path.exists(flow_bev_label_dir):
                os.mkdir(flow_bev_label_dir)
            flow_bev_label_path = os.path.join(
                flow_bev_label_dir,
                sample_key
            )

        pcd_height_label_path = None
        if self.use_lyft:
            pcd_height_label_dir = os.path.join(self.ocf_dataset_path, prefix, "pcd_height")
            if not os.path.exists(pcd_height_label_dir):
                os.mkdir(pcd_height_label_dir)
            pcd_height_label_path = os.path.join(
                pcd_height_label_dir,
                sample_key
            )

        seg_instance3d_label_path = None
        if self.load_segmentation_instance3d:
            seg_instance3d_label_dir = self.resolve_segmentation_instance3d_dir(prefix)
            if seg_instance3d_label_dir is not None:
                seg_instance3d_label_path = os.path.join(seg_instance3d_label_dir, sample_key)

        seg_cls_label_path = None
        if self.load_segmentation_cls_instance3d:
            seg_cls_label_dir = self.resolve_segmentation_cls_dir(prefix)
            if seg_cls_label_dir is None:
                raise FileNotFoundError(
                    "Could not resolve bboxcls segmentation directory from "
                    f"segmentation_cls_dataset_path={self.segmentation_cls_dataset_path!r} "
                    f"for prefix={prefix!r}."
                )
            seg_cls_label_path = os.path.join(seg_cls_label_dir, sample_key)

        gt_occ_inst_label_path = None
        if self.load_gt_occ_inst:
            gt_occ_inst_label_dir = self.resolve_gt_occ_inst_dir(prefix)
            if gt_occ_inst_label_dir is None:
                raise FileNotFoundError(
                    "Could not resolve gt_occ_inst directory from "
                    f"gt_occ_inst_dataset_path={self.gt_occ_inst_dataset_path!r} "
                    f"for prefix={prefix!r}."
                )
            gt_occ_inst_label_path = os.path.join(gt_occ_inst_label_dir, sample_key)

        # ---------------- small inline utils ----------------
        def to_tensor_safe(arr, dtype):
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr = np.vstack(arr)
            arr = np.asarray(arr, dtype=dtype)
            return torch.from_numpy(arr)

        def load_list_from_npz(npz_path, key):
            with np.load(npz_path, allow_pickle=True) as f:
                if key in f.files:
                    x = f[key]
                else:
                    x = f["arr_0"]
            if isinstance(x, np.ndarray) and x.dtype == object:
                return x.tolist()
            return list(x)

        def sparse_instance3d_to_dense(arr):
            dense = np.ones(
                (self.dimension[0], self.dimension[1], self.dimension[2]),
                dtype=np.int64,
            ) * int(self.background)

            rows = np.asarray(arr)
            if isinstance(rows, np.ndarray) and rows.dtype == object:
                if rows.size == 0:
                    rows = np.zeros((0, 5), dtype=np.int64)
                else:
                    rows = np.vstack(rows)
            rows = np.asarray(rows, dtype=np.int64)

            if rows.size == 0:
                return torch.from_numpy(dense)
            if rows.ndim != 2 or rows.shape[1] < 4:
                raise ValueError(f"segmentation_instance3d row shape is invalid: {rows.shape}")

            dense[rows[:, 0], rows[:, 1], rows[:, 2]] = rows[:, -1]
            return torch.from_numpy(dense)

        def sparse_segmentation3d_to_dense(arr):
            dense = np.zeros(
                (self.dimension[0], self.dimension[1], self.dimension[2]),
                dtype=np.int64,
            ) + int(self.background)

            rows = np.asarray(arr)
            if isinstance(rows, np.ndarray) and rows.dtype == object:
                if rows.size == 0:
                    rows = np.zeros((0, 4), dtype=np.int64)
                else:
                    rows = np.vstack(rows)
            rows = np.asarray(rows, dtype=np.int64)

            if rows.size == 0:
                return torch.from_numpy(dense)
            if rows.ndim != 2 or rows.shape[1] < 4:
                raise ValueError(f"segmentation_cls row shape is invalid: {rows.shape}")

            dense[rows[:, 0], rows[:, 1], rows[:, 2]] = rows[:, -1]
            return torch.from_numpy(dense)

        def load_segmentation_cls_sparse_list_or_raise():
            if seg_cls_label_path is None:
                raise FileNotFoundError("segmentation_cls label path is not resolved.")
            npz_path = seg_cls_label_path + ".npz"
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"bboxcls segmentation npz is missing: {npz_path}")

            gt_list = load_list_from_npz(npz_path, self.segmentation_cls_key)
            if len(gt_list) == 0:
                raise ValueError(f"bboxcls segmentation list is empty: {npz_path}")
            out = []
            for rows_raw in gt_list:
                rows = self._normalize_sparse_rows(
                    rows_raw,
                    min_cols=4,
                    label="segmentation_cls",
                )
                out.append(self._filter_sparse_rows_by_class(rows, class_col=3))
            return out

        def load_gt_occ_inst_sparse_list_or_raise(expected_seq_len):
            if gt_occ_inst_label_path is None:
                raise FileNotFoundError("gt_occ_inst label path is not resolved.")
            npz_path = gt_occ_inst_label_path + ".npz"
            if not os.path.exists(npz_path):
                raise FileNotFoundError(f"gt_occ_inst npz is missing: {npz_path}")

            gt_list = load_list_from_npz(npz_path, self.gt_occ_inst_key)
            self._validate_gt_occ_inst_sparse_list(
                sparse_list=gt_list,
                sample_key=sample_key,
                expected_seq_len=expected_seq_len,
            )
            out = []
            for rows_raw in gt_list:
                rows = self._normalize_sparse_rows(
                    rows_raw,
                    min_cols=5,
                    label="gt_occ_inst",
                )
                rows = self._filter_sparse_rows_by_class(rows, class_col=3)
                out.append(rows.astype(np.int64, copy=False))
            return out

        def build_segmentation_cls_tensor_from_sparse_list(gt_list):
            seg_cls_list = []
            for j in range(len(gt_list)):
                seg_cls = sparse_segmentation3d_to_dense(gt_list[j]).long()
                seg_cls_list.append(seg_cls.unsqueeze(0))

            return torch.cat(seg_cls_list, dim=0).long()

        # ========== data regen if the npz files are broken ==========
        need_regen = False

        check_list = [
            (seg_label_path + ".npz", "segmentation_saved_list2"),
        ]

        if self.load_segmentation_bev:
            check_list.append((seg_bev_label_path + ".npz", "segmentation_bev_saved_list2"))
        if self.use_flow:
            check_list.append((instance_bev_label_path + ".npz", "instance_bev_saved_list2"))
            check_list.append((flow_bev_label_path + ".npz", "flow_bev_saved_list2"))
        if self.load_segmentation_instance3d and (seg_instance3d_label_path is not None):
            check_list.append((seg_instance3d_label_path + ".npz", self.segmentation_instance3d_key))
        if self.load_segmentation_cls_instance3d and (seg_cls_label_path is not None):
            check_list.append((seg_cls_label_path + ".npz", self.segmentation_cls_key))
        if self.load_gt_occ_inst and (gt_occ_inst_label_path is not None):
            check_list.append((gt_occ_inst_label_path + ".npz", self.gt_occ_inst_key))

        if self.use_lyft:
            check_list.append((pcd_height_label_path + ".npz", "pcd_height_saved_list2"))

        if self.validate_cache:
            for p, key in check_list:
                if not os.path.exists(p):
                    need_regen = True
                    continue
                try:
                    with np.load(p, allow_pickle=True) as f:
                        _ = f[key]
                except Exception as e:
                    print(f"[BAD_NPZ] {p} key={key} err={repr(e)}", flush=True)
                    try:
                        os.remove(p)
                    except Exception as e2:
                        print(f"[BAD_NPZ_REMOVE_FAIL] {p} err={repr(e2)}", flush=True)
                    need_regen = True
        else:
            for p, _ in check_list:
                if not os.path.exists(p):
                    need_regen = True
                    break

        # ---------------- load cached labels ----------------
        segmentation_list = []
        if (not need_regen) and os.path.exists(seg_label_path + ".npz"):
            try:
                gt_list = load_list_from_npz(seg_label_path + ".npz", "segmentation_saved_list2")
                for j in range(len(gt_list)):
                    segmentation = np.zeros(
                        (self.dimension[0], self.dimension[1], self.dimension[2]),
                        dtype=np.float32
                    ) + float(self.background)

                    gt = to_tensor_safe(gt_list[j], np.int64)
                    segmentation[gt[:, 0].long(), gt[:, 1].long(), gt[:, 2].long()] = gt[:, -1].float()

                    segmentation_list.append(torch.from_numpy(segmentation).unsqueeze(0))
            except Exception as e:
                print(f"[BAD_NPZ] {seg_label_path}.npz err={repr(e)}", flush=True)
                try:
                    os.remove(seg_label_path + ".npz")
                except Exception:
                    pass
                need_regen = True

        segmentation_bev_list = []
        if self.load_segmentation_bev and (not need_regen) and os.path.exists(seg_bev_label_path + ".npz"):
            try:
                gt_list = load_list_from_npz(seg_bev_label_path + ".npz", "segmentation_bev_saved_list2")
                for j in range(len(gt_list)):
                    segmentation_bev = np.zeros(
                        (self.dimension[0], self.dimension[1]),
                        dtype=np.float32
                    ) + float(self.background)

                    gt = to_tensor_safe(gt_list[j], np.int64)
                    segmentation_bev[gt[:, 0].long(), gt[:, 1].long()] = gt[:, -1].float()

                    segmentation_bev_list.append(torch.from_numpy(segmentation_bev).unsqueeze(0))
            except Exception as e:
                print(f"[BAD_NPZ] {seg_bev_label_path}.npz err={repr(e)}", flush=True)
                try:
                    os.remove(seg_bev_label_path + ".npz")
                except Exception:
                    pass
                need_regen = True

        instance_bev_list = []
        if self.use_flow and (not need_regen) and os.path.exists(instance_bev_label_path + ".npz"):
            try:
                gt_list = load_list_from_npz(instance_bev_label_path + ".npz", "instance_bev_saved_list2")
                for j in range(len(gt_list)):
                    instance_bev = np.ones(
                        (self.dimension[0], self.dimension[1]),
                        dtype=np.float32
                    ) * float(self.background)

                    gt = to_tensor_safe(gt_list[j], np.int64)
                    instance_bev[gt[:, 0].long(), gt[:, 1].long()] = gt[:, -1].float()

                    instance_bev_list.append(torch.from_numpy(instance_bev).unsqueeze(0))
            except Exception as e:
                print(f"[BAD_NPZ] {instance_bev_label_path}.npz err={repr(e)}", flush=True)
                try:
                    os.remove(instance_bev_label_path + ".npz")
                except Exception:
                    pass
                need_regen = True

        flow_bev_list = []
        if self.use_flow and (not need_regen) and os.path.exists(flow_bev_label_path + ".npz"):
            try:
                gt_list = load_list_from_npz(flow_bev_label_path + ".npz", "flow_bev_saved_list2")
                for j in range(len(gt_list)):
                    flow_bev = np.ones(
                        (2, self.dimension[0] // 4, self.dimension[1] // 4),
                        dtype=np.float32
                    ) * 255.0

                    gt = to_tensor_safe(gt_list[j], np.float32)
                    flow_bev[:, gt[:, 0].long(), gt[:, 1].long()] = gt[:, 2:].permute(1, 0)

                    flow_bev_list.append(torch.from_numpy(flow_bev).unsqueeze(0))
            except Exception as e:
                print(f"[BAD_NPZ] {flow_bev_label_path}.npz err={repr(e)}", flush=True)
                try:
                    os.remove(flow_bev_label_path + ".npz")
                except Exception:
                    pass
                need_regen = True

        pcd_height_list = []
        if self.use_lyft and (not need_regen) and os.path.exists(pcd_height_label_path + ".npz"):
            try:
                gt_list = load_list_from_npz(pcd_height_label_path + ".npz", "pcd_height_saved_list2")
                for j in range(len(gt_list)):
                    pcd_height = np.zeros((self.dimension[0], self.dimension[1]), dtype=np.float32) + float(self.background)
                    gt = to_tensor_safe(gt_list[j], np.int64)
                    pcd_height[gt[:, 0].long(), gt[:, 1].long()] = gt[:, -1].float()
                    pcd_height_list.append(torch.from_numpy(pcd_height).unsqueeze(0))
            except Exception as e:
                print(f"[BAD_NPZ] {pcd_height_label_path}.npz err={repr(e)}", flush=True)
                try:
                    os.remove(pcd_height_label_path + ".npz")
                except Exception:
                    pass
                need_regen = True

        segmentation_instance3d_list = []
        segmentation_instance3d_sparse_list = None
        if self.load_segmentation_instance3d and (not need_regen) and (seg_instance3d_label_path is not None) and os.path.exists(seg_instance3d_label_path + ".npz"):
            try:
                gt_list = load_list_from_npz(seg_instance3d_label_path + ".npz", self.segmentation_instance3d_key)
                segmentation_instance3d_sparse_list = []
                for j in range(len(gt_list)):
                    rows = self._normalize_sparse_rows(
                        gt_list[j],
                        min_cols=4,
                        label="segmentation_instance3d",
                    )
                    rows = self._filter_sparse_rows_by_class(rows, class_col=-1)
                    segmentation_instance3d_sparse_list.append(rows)
                    segmentation_instance3d = sparse_instance3d_to_dense(rows).long()
                    segmentation_instance3d_list.append(segmentation_instance3d.unsqueeze(0))
            except Exception as e:
                print(f"[BAD_NPZ] {seg_instance3d_label_path}.npz err={repr(e)}", flush=True)
                segmentation_instance3d_sparse_list = None
                need_regen = True

        gt_occ_inst_sparse_list = None
        if self.load_gt_occ_inst:
            gt_occ_inst_sparse_list = load_gt_occ_inst_sparse_list_or_raise(
                expected_seq_len=results['sequence_length']
            )

        # ---------------- decide cache usage ----------------
        use_cache = (not need_regen) \
            and os.path.exists(seg_label_path + ".npz") \
            and (len(segmentation_list) > 0)

        if self.load_segmentation_bev:
            use_cache = use_cache and os.path.exists(seg_bev_label_path + ".npz") \
                and (len(segmentation_bev_list) > 0)

        if self.use_flow:
            use_cache = use_cache and os.path.exists(instance_bev_label_path + ".npz") \
                and os.path.exists(flow_bev_label_path + ".npz") \
                and (len(instance_bev_list) > 0) \
                and (len(flow_bev_list) > 0)
        if self.use_lyft:
            use_cache = use_cache and os.path.exists(pcd_height_label_path + ".npz") and (len(pcd_height_list) > 0)
        if self.load_segmentation_instance3d:
            use_cache = use_cache and (seg_instance3d_label_path is not None) \
                and os.path.exists(seg_instance3d_label_path + ".npz") \
                and (len(segmentation_instance3d_list) > 0)
        if self.load_gt_occ_inst:
            use_cache = use_cache and (gt_occ_inst_sparse_list is not None) \
                and (len(gt_occ_inst_sparse_list) == int(results['sequence_length']))

        if use_cache:
            results['segmentation'] = torch.cat(segmentation_list, dim=0)
            if self.load_segmentation_bev:
                results['segmentation_bev'] = torch.cat(segmentation_bev_list, dim=0)
            if self.use_flow:
                results['instance_bev'] = torch.cat(instance_bev_list, dim=0)
            if self.use_flow:
                results['flow_bev'] = torch.cat(flow_bev_list, dim=0).float()
            if self.load_segmentation_instance3d:
                results['segmentation_instance3d'] = torch.cat(segmentation_instance3d_list, dim=0).long()
                if segmentation_instance3d_sparse_list is None:
                    raise ValueError("segmentation_instance3d sparse list is missing while cache is used")
                centers_world, centers_valid, instance_ids = self.build_instance_center_world_targets(segmentation_instance3d_sparse_list)
                results['gt_instance_centers_world'] = centers_world
                results['gt_instance_centers_valid'] = centers_valid
                results['gt_instance_ids'] = instance_ids
                if self.load_segmentation_cls_instance3d:
                    segmentation_cls_sparse_list = load_segmentation_cls_sparse_list_or_raise()
                    self._validate_segmentation_cls_instance3d_alignment(
                        segmentation_instance3d_sparse_list=segmentation_instance3d_sparse_list,
                        segmentation_cls_sparse_list=segmentation_cls_sparse_list,
                        sample_key=sample_key,
                    )
                    seg_cls_tensor = build_segmentation_cls_tensor_from_sparse_list(
                        segmentation_cls_sparse_list
                    )
                    if seg_cls_tensor.shape != results['segmentation_instance3d'].shape:
                        raise ValueError(
                            "segmentation_cls_instance3d shape mismatch: "
                            f"{tuple(seg_cls_tensor.shape)} vs "
                            f"{tuple(results['segmentation_instance3d'].shape)}"
                        )
                    results['segmentation_cls_instance3d'] = torch.stack(
                        (seg_cls_tensor, results['segmentation_instance3d']),
                        dim=1,
                    ).long()
            if self.load_gt_occ_inst:
                results['gt_occ_inst'] = gt_occ_inst_sparse_list

            if self.use_lyft:
                results['height'] = torch.cat(pcd_height_list, dim=0)

            for key, value in results.items():
                if key in [
                    'sample_token', 'centerness', 'offset', 'flow_bev', 'time_receptive_field', "indices",
                    'segmentation', 'segmentation_bev', 'instance_bev', 'attribute_label',
                    'segmentation_instance3d', 'segmentation_cls_instance3d', 'gt_occ_inst',
                    'gt_instance_centers_world', 'gt_instance_centers_valid', 'gt_instance_ids',
                    'sequence_length', 'instance_dict', 'instance_map', 'input_dict',
                    'egopose_list', 'ego2lidar_list', 'scene_token', 'instance'
                ]:
                    continue
                if self.use_lyft and key == 'height':
                    continue
                if isinstance(value, (list, tuple)) and all(torch.is_tensor(v) for v in value):
                    results[key] = torch.cat(value, dim=0)

            return results

        # ---------------- regenerate and save (원래 else 블록 유지) ----------------
        results['segmentation'] = []
        if self.load_segmentation_bev:
            results['segmentation_bev'] = []
        if self.use_flow:
            results['instance_bev'] = []
        if self.load_segmentation_instance3d:
            results['segmentation_instance3d'] = []
            segmentation_instance3d_sparse_list = []

        segmentation_saved_list = []
        segmentation_bev_saved_list = []
        instance_bev_saved_list = []

        if self.use_lyft:
            results['height'] = []
            pcd_height_saved_list = []

        sequence_length = results['sequence_length']
        self.visible_instance_set = set()

        for self.counter in range(sequence_length):
            segmentation, instance, _, bbox_height = self.get_label(results)

            results['segmentation'].append(segmentation)
            if self.load_segmentation_bev:
                segmentation_bev = self.get_segmentation_bev(segmentation)
                results['segmentation_bev'].append(segmentation_bev.unsqueeze(0))
            if self.use_flow:
                instance_bev = self.get_instance_bev(instance)
                results['instance_bev'].append(instance_bev.unsqueeze(0))
            if self.load_segmentation_instance3d:
                results['segmentation_instance3d'].append(instance.long())
                instance_cur = instance.squeeze(0).long()
                coords = (instance_cur > 0).nonzero(as_tuple=False)
                if coords.numel() == 0:
                    sparse_rows = np.zeros((0, 5), dtype=np.int64)
                else:
                    inst_ids = instance_cur[coords[:, 0], coords[:, 1], coords[:, 2]].view(-1, 1)
                    semantic_placeholder = torch.ones_like(inst_ids)
                    sparse_rows = torch.cat((coords.long(), semantic_placeholder.long(), inst_ids.long()), dim=1).cpu().numpy()
                segmentation_instance3d_sparse_list.append(sparse_rows)

            x_grid = torch.linspace(0, self.dimension[0] - 1, self.dimension[0], dtype=torch.long)
            x_grid = x_grid.view(self.dimension[0], 1, 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
            y_grid = torch.linspace(0, self.dimension[1] - 1, self.dimension[1], dtype=torch.long)
            y_grid = y_grid.view(1, self.dimension[1], 1).expand(self.dimension[0], self.dimension[1], self.dimension[2])
            z_grid = torch.linspace(0, self.dimension[2] - 1, self.dimension[2], dtype=torch.long)
            z_grid = z_grid.view(1, 1, self.dimension[2]).expand(self.dimension[0], self.dimension[1], self.dimension[2])

            segmentation_for_save = torch.stack((x_grid, y_grid, z_grid), -1).view(-1, 3)
            segmentation_label = segmentation.squeeze(0).view(-1, 1)
            segmentation_for_save = torch.cat((segmentation_for_save, segmentation_label), dim=-1)
            kept = segmentation_for_save[:, -1] != 0
            segmentation_saved_list.append(segmentation_for_save[kept])

            if self.load_segmentation_bev or self.use_flow or self.use_lyft:
                x_grid_bev = torch.linspace(0, self.dimension[0] - 1, self.dimension[0], dtype=torch.long)
                x_grid_bev = x_grid_bev.view(self.dimension[0], 1).expand(self.dimension[0], self.dimension[1])
                y_grid_bev = torch.linspace(0, self.dimension[1] - 1, self.dimension[1], dtype=torch.long)
                y_grid_bev = y_grid_bev.view(1, self.dimension[1]).expand(self.dimension[0], self.dimension[1])

            if self.load_segmentation_bev:
                segmentation_bev_for_save = torch.stack((x_grid_bev, y_grid_bev), -1).view(-1, 2)
                segmentation_bev_label = segmentation_bev.unsqueeze(-1).view(-1, 1)
                segmentation_bev_for_save = torch.cat((segmentation_bev_for_save, segmentation_bev_label), dim=-1)
                kept = segmentation_bev_for_save[:, -1] != 0
                segmentation_bev_saved_list.append(segmentation_bev_for_save[kept])

            if self.use_flow:
                instance_bev_for_save = torch.stack((x_grid_bev, y_grid_bev), -1).view(-1, 2)
                instance_bev_label = instance_bev.unsqueeze(-1).view(-1, 1)
                instance_bev_for_save = torch.cat((instance_bev_for_save, instance_bev_label), dim=-1)
                kept = instance_bev_for_save[:, -1] != 0
                instance_bev_saved_list.append(instance_bev_for_save[kept])

            if self.use_lyft:
                results['height'].append(bbox_height)
                pcd_height_for_save = torch.stack((x_grid_bev, y_grid_bev), -1).view(-1, 2)
                pcd_height_label = bbox_height.squeeze(0).view(-1, 1)
                pcd_height_for_save = torch.cat((pcd_height_for_save, pcd_height_label), dim=-1)
                kept = pcd_height_for_save[:, -1] != 0
                pcd_height_saved_list.append(pcd_height_for_save[kept])

        if self.write_cache:
            segmentation_saved_list2 = [item.cpu().detach().numpy() for item in segmentation_saved_list]

            obj_seg = np.array(segmentation_saved_list2, dtype=object)
            self.atomic_savez(seg_label_path, segmentation_saved_list2=obj_seg)

            if self.load_segmentation_bev:
                segmentation_bev_saved_list2 = [item.cpu().detach().numpy() for item in segmentation_bev_saved_list]
                obj_seg_bev = np.array(segmentation_bev_saved_list2, dtype=object)
                self.atomic_savez(seg_bev_label_path, segmentation_bev_saved_list2=obj_seg_bev)

            if self.use_flow:
                instance_bev_saved_list2 = [item.cpu().detach().numpy() for item in instance_bev_saved_list]
                obj_inst_bev = np.array(instance_bev_saved_list2, dtype=object)
                self.atomic_savez(instance_bev_label_path, instance_bev_saved_list2=obj_inst_bev)

            if self.use_lyft:
                pcd_height_saved_list2 = [item.cpu().detach().numpy() for item in pcd_height_saved_list]
                obj_pcd_height = np.array(pcd_height_saved_list2, dtype=object)
                self.atomic_savez(pcd_height_label_path, pcd_height_saved_list2=obj_pcd_height)

        if self.use_lyft:
            results['height'] = torch.cat(results['height'], dim=0)

        results['segmentation'] = torch.cat(results['segmentation'], dim=0)
        if self.load_segmentation_bev:
            results['segmentation_bev'] = torch.cat(results['segmentation_bev'], dim=0)
        if self.use_flow:
            results['instance_bev'] = torch.cat(results['instance_bev'], dim=0)
        if self.load_segmentation_instance3d:
            results['segmentation_instance3d'] = torch.cat(results['segmentation_instance3d'], dim=0).long()
            centers_world, centers_valid, instance_ids = self.build_instance_center_world_targets(segmentation_instance3d_sparse_list)
            results['gt_instance_centers_world'] = centers_world
            results['gt_instance_centers_valid'] = centers_valid
            results['gt_instance_ids'] = instance_ids
            if self.load_segmentation_cls_instance3d:
                segmentation_cls_sparse_list = load_segmentation_cls_sparse_list_or_raise()
                self._validate_segmentation_cls_instance3d_alignment(
                    segmentation_instance3d_sparse_list=segmentation_instance3d_sparse_list,
                    segmentation_cls_sparse_list=segmentation_cls_sparse_list,
                    sample_key=sample_key,
                )
                seg_cls_tensor = build_segmentation_cls_tensor_from_sparse_list(
                    segmentation_cls_sparse_list
                )
                if seg_cls_tensor.shape != results['segmentation_instance3d'].shape:
                    raise ValueError(
                        "segmentation_cls_instance3d shape mismatch: "
                        f"{tuple(seg_cls_tensor.shape)} vs "
                        f"{tuple(results['segmentation_instance3d'].shape)}"
                    )
                results['segmentation_cls_instance3d'] = torch.stack(
                    (seg_cls_tensor, results['segmentation_instance3d']),
                    dim=1,
                ).long()
        if self.load_gt_occ_inst:
            results['gt_occ_inst'] = gt_occ_inst_sparse_list

        if self.use_flow:
            results['flow_bev'] = self.get_flow_label(results, ignore_index=255)

        if self.use_flow and self.write_cache:
            flow_bev_saved_list = []
            d0 = self.dimension[0] // 4
            d1 = self.dimension[1] // 4
            for cnt in range(sequence_length):
                flow_bev = results['flow_bev'][cnt, ...]
                x_grid = torch.linspace(0, d0 - 1, d0, dtype=torch.long).view(d0, 1).expand(d0, d1)
                y_grid = torch.linspace(0, d1 - 1, d1, dtype=torch.long).view(1, d1).expand(d0, d1)

                flow_bev_for_save = torch.stack((x_grid, y_grid), -1).view(-1, 2)
                flow_bev_label = flow_bev.permute(1, 2, 0).view(-1, 2)
                flow_bev_for_save = torch.cat((flow_bev_for_save, flow_bev_label), dim=-1)
                kept = (flow_bev_for_save[:, -1] != 255) & (flow_bev_for_save[:, -2] != 255)
                flow_bev_saved_list.append(flow_bev_for_save[kept])

            flow_bev_saved_list2 = [item.cpu().detach().numpy() for item in flow_bev_saved_list]
            obj_flow_bev = np.array(flow_bev_saved_list2, dtype=object)
            self.atomic_savez(flow_bev_label_path, flow_bev_saved_list2=obj_flow_bev)

        for key, value in results.items():
            if key in [
                'sample_token', 'centerness', 'offset', 'flow_bev', 'time_receptive_field', "indices",
                'segmentation', 'segmentation_bev', 'instance_bev', 'attribute_label',
                'segmentation_instance3d', 'segmentation_cls_instance3d', 'gt_occ_inst',
                'gt_instance_centers_world', 'gt_instance_centers_valid', 'gt_instance_ids',
                'sequence_length', 'instance_dict', 'instance_map', 'input_dict',
                'egopose_list', 'ego2lidar_list', 'scene_token', 'instance'
            ]:
                continue
            if self.use_lyft and key == 'height':
                continue
            if isinstance(value, (list, tuple)) and all(torch.is_tensor(v) for v in value):
                results[key] = torch.cat(value, dim=0)

        return results
