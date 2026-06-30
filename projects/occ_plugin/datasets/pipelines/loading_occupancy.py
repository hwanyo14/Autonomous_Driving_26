# Developed by Jingyi Xu based on the codebase of Cam4DOcc and PowerBEV 
# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
# https://github.com/BIT-XJY/EfficientOCF

import numpy as np
import numba as nb
from mmdet.datasets.builder import PIPELINES
import yaml, os
import torch
import torch.nn.functional as F
import copy
import time

@PIPELINES.register_module()
class LoadOccupancy(object):

    def __init__(self, 
                 to_float32=True, 
                 occ_path=None, ocf_dataset_path=None, 
                 grid_size=[512, 512, 40], 
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 
                 unoccupied=0, 
                 gt_resize_ratio=1, 
                 use_fine_occ=False, 
                 test_mode=False, 
                 use_lyft=False,
                 dt_path=None,
                 load_occ_dt=False,
                 dt_type='float16',
                 strict_dt=True,
                 load_height=True,
                 validate_height_cache=True,
                 write_height_cache=True,
                 exclude_occ_class_ids=()):
        '''
        Read sequential fine-grained occupancy labels from nuScenes-Occupancy if use_fine_occ=True
        '''
        self.to_float32 = to_float32
        self.occ_path = occ_path
        self.ocf_dataset_path = ocf_dataset_path
        self.grid_size = np.array(grid_size)
        self.unoccupied = unoccupied
        self.pc_range = np.array(pc_range)
        self.voxel_size = (self.pc_range[3:] - self.pc_range[:3]) / self.grid_size
        self.gt_resize_ratio = gt_resize_ratio
        self.use_fine_occ = use_fine_occ
        self.dimension = grid_size
        self.test_mode = test_mode
        self.use_lyft = use_lyft
        self.dt_path = dt_path
        self.load_occ_dt = load_occ_dt
        self.dt_type = dt_type
        self.strict_dt = strict_dt
        self.load_height = bool(load_height)
        self.validate_height_cache = bool(validate_height_cache)
        self.write_height_cache = bool(write_height_cache)
        self.exclude_occ_class_ids = tuple(int(v) for v in exclude_occ_class_ids)

    def _load_single_dt(self, scene_token, lidar_token):
        p = os.path.join(self.dt_path, f"scene_{scene_token}", "dt", f"{lidar_token}.npz")
        with np.load(p, allow_pickle=False) as f:
            if "field_cm" in f.files:      # int16 cm
                dt = f["field_cm"].astype(np.float32) * 0.01
            elif "field_u8" in f.files:    # uint8 quantized meters
                q = float(np.asarray(f["quant_scale_m"]).reshape(-1)[0]) if "quant_scale_m" in f.files else 0.1
                dt = f["field_u8"].astype(np.float32) * q
            elif "field" in f.files:       # float
                dt = f["field"].astype(np.float32)
            else:
                raise KeyError(f"no field key in {p}: {f.files}")

        # precompute_dt_full.py는 (Z,Y,X) 저장이므로 (X,Y,Z)로 맞춤
        if dt.shape == (self.dimension[2], self.dimension[1], self.dimension[0]):
            dt = np.transpose(dt, (2, 1, 0))
        elif dt.shape != tuple(self.dimension):
            raise ValueError(f"dt shape mismatch: {dt.shape} vs {self.dimension}")

        dt = dt.astype(np.float16 if self.dt_type == "float16" else np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(dt))

    def _decode_sparse_dt_list_to_txyz(self, dt_list_obj, seq_len, dmax_m, save_mode_meta):
        Z, Y, X = int(self.dimension[2]), int(self.dimension[1]), int(self.dimension[0])
        dt_tzyx = np.full((seq_len, Z, Y, X), float(dmax_m), dtype=np.float32)

        if (not isinstance(dt_list_obj, np.ndarray)) or (dt_list_obj.dtype != object):
            raise ValueError(f"occ_dt_saved_list2 must be object array, got type={type(dt_list_obj)} dtype={getattr(dt_list_obj, 'dtype', None)}")
        if len(dt_list_obj) != seq_len:
            raise ValueError(f"occ_dt_saved_list2 length mismatch: {len(dt_list_obj)} vs {seq_len}")

        for t in range(seq_len):
            rows = np.asarray(dt_list_obj[t])
            if rows.size == 0:
                continue
            if rows.ndim != 2 or rows.shape[1] < 4:
                raise ValueError(f"invalid sparse dt rows shape={rows.shape} at t={t}")
            x = rows[:, 0].astype(np.int32)
            y = rows[:, 1].astype(np.int32)
            z = rows[:, 2].astype(np.int32)
            v = rows[:, 3].astype(np.float32)

            # old exporter wrote int16-cm values in int32 container.
            if save_mode_meta == "int16_cm":
                v = v * 0.01

            valid = (0 <= x) & (x < X) & (0 <= y) & (y < Y) & (0 <= z) & (z < Z)
            dt_tzyx[t, z[valid], y[valid], x[valid]] = v[valid]

        return np.transpose(dt_tzyx, (0, 3, 2, 1))  # [T, X, Y, Z]

    def _load_seq_dt_window(self, results):
        seq_len = int(results["sequence_length"])
        present_idx = int(results["time_receptive_field"] - 1)
        present = results["input_dict"][present_idx]
        scene_token = str(present["scene_token"])
        lidar_token = str(present["lidar_token"])
        p = os.path.join(self.dt_path, f"{scene_token}_{lidar_token}.npz")

        with np.load(p, allow_pickle=True) as f:
            save_mode_meta = ""
            if "save_mode" in f.files:
                save_mode_meta = str(np.asarray(f["save_mode"]).reshape(-1)[0])

            expected_tokens = [str(results["input_dict"][i]["lidar_token"]) for i in range(seq_len)]
            if "frame_tokens" in f.files:
                frame_tokens = [str(x) for x in np.asarray(f["frame_tokens"]).tolist()]
                if frame_tokens != expected_tokens:
                    raise ValueError(
                        f"frame_tokens mismatch at {p}: expected={expected_tokens}, got={frame_tokens}"
                    )

            dt_txyz = None
            if "occ_dt_saved_list2" in f.files:
                arr = f["occ_dt_saved_list2"]
                # New format: dense [T,Z,Y,X] under occ_dt_saved_list2.
                if isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim == 4:
                    dt_tzyx = arr.astype(np.float32, copy=False)
                    if (save_mode_meta == "int16_cm") or (arr.dtype == np.int16):
                        dt_tzyx = dt_tzyx * 0.01
                    elif (save_mode_meta == "uint8") or (arr.dtype == np.uint8):
                        if "quant_scale_m" in f.files:
                            q = float(np.asarray(f["quant_scale_m"]).reshape(-1)[0])
                        else:
                            dmax_m = float(np.asarray(f["dmax_m"]).reshape(-1)[0]) if "dmax_m" in f.files else 20.0
                            q = dmax_m / 255.0
                        dt_tzyx = dt_tzyx * q
                    dt_txyz = np.transpose(dt_tzyx, (0, 3, 2, 1))
                else:
                    dmax_m = float(np.asarray(f["dmax_m"]).reshape(-1)[0]) if "dmax_m" in f.files else 20.0
                    dt_txyz = self._decode_sparse_dt_list_to_txyz(
                        dt_list_obj=arr,
                        seq_len=seq_len,
                        dmax_m=dmax_m,
                        save_mode_meta=save_mode_meta,
                    )
            elif "field_cm" in f.files:
                arr = f["field_cm"]
                if arr.ndim != 4:
                    raise ValueError(f"field_cm must be [T,Z,Y,X], got {arr.shape}")
                dt_txyz = np.transpose(arr.astype(np.float32) * 0.01, (0, 3, 2, 1))
            elif "field_u8" in f.files:
                arr = f["field_u8"]
                if arr.ndim != 4:
                    raise ValueError(f"field_u8 must be [T,Z,Y,X], got {arr.shape}")
                q = float(np.asarray(f["quant_scale_m"]).reshape(-1)[0]) if "quant_scale_m" in f.files else 0.1
                dt_txyz = np.transpose(arr.astype(np.float32) * q, (0, 3, 2, 1))
            elif "field" in f.files:
                arr = f["field"]
                if arr.ndim != 4:
                    raise ValueError(f"field must be [T,Z,Y,X], got {arr.shape}")
                dt_txyz = np.transpose(arr.astype(np.float32), (0, 3, 2, 1))
            else:
                raise KeyError(f"no DT key in {p}: {f.files}")

        if dt_txyz.shape[0] != seq_len:
            raise ValueError(f"DT sequence length mismatch: {dt_txyz.shape[0]} vs {seq_len}")
        exp_shape = (seq_len, int(self.dimension[0]) // 2, int(self.dimension[1]) // 2, int(self.dimension[2]) // 2)
        if dt_txyz.shape != exp_shape:
            raise ValueError(f"DT shape mismatch: {dt_txyz.shape} vs {exp_shape}")

        dt_txyz = dt_txyz.astype(np.float16 if self.dt_type == "float16" else np.float32, copy=False)
        return torch.from_numpy(np.ascontiguousarray(dt_txyz))
    
    def get_seq_occ_dt(self, results):
        seq_len = int(results["sequence_length"])
        present_idx = int(results["time_receptive_field"] - 1)
        present = results["input_dict"][present_idx]
        seq_path = os.path.join(self.dt_path, f"{present['scene_token']}_{present['lidar_token']}.npz")

        # Prefer one-file-per-window DT (aligned to present frame).
        if os.path.exists(seq_path):
            try:
                return self._load_seq_dt_window(results)
            except Exception:
                if self.strict_dt:
                    raise

        # Fallback: old per-frame DT layout.
        out = []
        zero_dtype = torch.float16 if self.dt_type == "float16" else torch.float32
        for i in range(seq_len):
            s = results["input_dict"][i]["scene_token"]
            l = results["input_dict"][i]["lidar_token"]
            try:
                out.append(self._load_single_dt(s, l))
            except Exception:
                if self.strict_dt:
                    raise
                out.append(torch.zeros(self.grid_size.tolist(), dtype=zero_dtype))
        return torch.stack(out, dim=0)  # [T, X, Y, Z]

    def get_seq_pseudo_occ(self, results):
        sequence_length = results['sequence_length']
        gt_occ_seq = []

        for count in range(sequence_length):
            processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
            processed_label = torch.from_numpy(processed_label)
            gt_occ_seq.append(processed_label)

        gt_occ_seq = torch.stack(gt_occ_seq)
        return gt_occ_seq

    def get_seq_occ(self, results, only_gt_occ=False):
        sequence_length = results['sequence_length']
        gt_occ_seq = []
        pcd_xyhl_list = []

        for count in range(sequence_length):

            if self.use_lyft == False:
                scene_token_cur = results['input_dict'][count]['scene_token']
                lidar_token_cur = results['input_dict'][count]['lidar_token']

                rel_path = 'scene_{0}/occupancy/{1}.npy'.format(scene_token_cur, lidar_token_cur)

                #  [z y x cls] or [z y x vx vy vz cls]
                pcd = np.load(os.path.join(self.occ_path, rel_path))
                pcd_label = pcd[..., -1:]
                pcd_label[pcd_label==0] = 255
                pcd_np_cor = self.voxel2world(pcd[..., [2,1,0]] + 0.5)
                untransformed_occ = copy.deepcopy(pcd_np_cor)

                egopose_list = results['egopose_list']
                ego2lidar_list = results['ego2lidar_list']
                time_receptive_field = results['time_receptive_field']
                present_global2ego = egopose_list[time_receptive_field - 1]
                present_ego2lidar = ego2lidar_list[time_receptive_field - 1]
                cur_global2ego = egopose_list[count]
                cur_ego2lidar = ego2lidar_list[count]

                pcd_np_cor = np.dot(cur_ego2lidar[1].inverse.rotation_matrix, pcd_np_cor.T)
                pcd_np_cor = pcd_np_cor.T
                pcd_np_cor = pcd_np_cor - cur_ego2lidar[0]
                # cur_ego -> global 
                pcd_np_cor = np.dot(cur_global2ego[1].inverse.rotation_matrix, pcd_np_cor.T)  
                pcd_np_cor = pcd_np_cor.T
                pcd_np_cor = pcd_np_cor - cur_global2ego[0]
                # global -> present_ego  
                pcd_np_cor = pcd_np_cor + present_global2ego[0]
                pcd_np_cor = np.dot(present_global2ego[1].rotation_matrix, pcd_np_cor.T)
                pcd_np_cor = pcd_np_cor.T
                # present_ego -> present_lidar
                pcd_np_cor = pcd_np_cor + present_ego2lidar[0]
                pcd_np_cor = np.dot(present_ego2lidar[1].rotation_matrix, pcd_np_cor.T) 
                pcd_np_cor = pcd_np_cor.T            

                pcd_np_cor = self.world2voxel(pcd_np_cor)

                # make sure the point is in the grid
                pcd_np_cor = np.clip(pcd_np_cor, np.array([0,0,0]), self.grid_size - 1)
                transformed_occ = copy.deepcopy(pcd_np_cor)
                pcd_np = np.concatenate([pcd_np_cor, pcd_label], axis=-1)

                if only_gt_occ == False:
                    pcd_np_filter = copy.deepcopy(pcd_np)
                    for otheridx in [0,1,7,8,11,12,13,14,15,16,17,18,255]:
                        pcd_np_filter[pcd_np_filter[:,-1]==otheridx,-1] = 0
                    for fgidx in [2,3,4,5,6,9,10]:
                        pcd_np_filter[pcd_np_filter[:,-1]==fgidx,-1] = 1
                
                    pcd_np_filter[:,0] = np.round(pcd_np_filter[:,0])
                    pcd_np_filter[:,1] = np.round(pcd_np_filter[:,1])
                    pcd_xyhl = np.zeros_like(pcd_np_filter)
                    height_bev = np.zeros((512,512))
                
                    xy_dict = {}
                    for i in range(len(pcd_np_filter)):
                        if pcd_np_filter[i][-1] != 0:
                            x = int(pcd_np_filter[i][0])
                            y = int(pcd_np_filter[i][1])
                            xy_pair = str(x) + '_' + str(y)
                            
                            if xy_pair in xy_dict.keys():
                                continue
                            else:
                                mask_cur = (pcd_np_filter[:,0] == x) & (pcd_np_filter[:,1] == y) & (pcd_np_filter[:,-1] != 0)
                                pcd_np_xy = pcd_np_filter[mask_cur]
                                pcd_np_xy = pcd_np_xy[:,2]
                                height_max = max(pcd_np_xy)
                                xy_dict[xy_pair] = height_max
                                
                                height_bev[x,y] = height_max
                            
            # 255: noise, 1-16 normal classes, 0 unoccupied
            pcd_np = pcd_np[np.lexsort((pcd_np_cor[:, 0], pcd_np_cor[:, 1], pcd_np_cor[:, 2])), :]
            pcd_np = pcd_np.astype(np.int64)
            processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
            processed_label = nb_process_label(processed_label, pcd_np)

            processed_label = torch.from_numpy(processed_label)

            # # TODO: hard coding
            # for otheridx in [0,1,7,8,11,12,13,14,15,16,17,18,255]:
            #     processed_label[processed_label==otheridx] = 0
            # for vehidx in [2,3,4,5,6,9,10]:
            #     processed_label[processed_label==vehidx] = 1            
            
            # processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.unoccupied
            gt_occ_seq.append(processed_label)
            if only_gt_occ == False:
                pcd_xyhl_list.append(torch.from_numpy(height_bev).unsqueeze(0))

        if only_gt_occ == False:
            return gt_occ_seq, pcd_xyhl_list
        else:
            return gt_occ_seq

    def __call__(self, results):
        
        if self.use_lyft == False:
            assert 'height' not in results.keys()

        time_receptive_field = results['time_receptive_field']

        prefix = "GMO"
        if self.use_lyft:
            prefix = prefix + "_lyft"
            results['gt_occ'] = self.get_seq_occ(results, only_gt_occ=True)
            return results

        if not self.load_height:
            results['gt_occ'] = self.get_seq_occ(results, only_gt_occ=True)
            if self.load_occ_dt:
                results["occ_dt"] = self.get_seq_occ_dt(results)
            return results

        height_bev_dir = os.path.join(self.ocf_dataset_path, prefix, "pcd_height")
        
        if not os.path.exists(height_bev_dir):
            os.mkdir(height_bev_dir)  
        height_bev_path = os.path.join(height_bev_dir, \
            results['input_dict'][time_receptive_field-1]['scene_token']+"_"+results['input_dict'][time_receptive_field-1]['lidar_token'])
        

        # ======== optional cache validation ========
        need_regen = False
        if self.validate_height_cache:
            check_list = [
                (height_bev_path + ".npz", "height_bev_saved_list2"),
            ]

            for p, key in check_list:
                if not os.path.exists(p):
                    need_regen = True
                    continue
                try:
                    with np.load(p, allow_pickle=True) as f:
                        if key not in f.files:
                            raise KeyError(f"missing key={key}, files={list(f.files)}")
                        _ = f[key]  # 실제로 로드가 되는지까지 확인
                except Exception as e:
                    print(f"[BAD_NPZ] {p} key={key} err={repr(e)}", flush=True)
                    try:
                        os.remove(p)
                    except Exception as e2:
                        print(f"[BAD_NPZ_REMOVE_FAIL] {p} err={repr(e2)}", flush=True)
                    need_regen = True


        pcd_xyhl_list = []
        load_ok = False
        if (not need_regen) and os.path.exists(height_bev_path+".npz"):
            try:
                results['gt_occ'] = self.get_seq_occ(results, only_gt_occ=True)

                # gt_pcd_xyhl_arr = np.load(height_bev_path+".npz",allow_pickle=True)['arr_0']
                with np.load(height_bev_path + ".npz", allow_pickle=True) as f:
                    gt_pcd_xyhl_arr = f["height_bev_saved_list2"]
                if isinstance(gt_pcd_xyhl_arr, np.ndarray) and gt_pcd_xyhl_arr.dtype == object:
                    gt_pcd_xyhl_arr = gt_pcd_xyhl_arr.tolist()
                else:
                    gt_pcd_xyhl_arr = list(gt_pcd_xyhl_arr)

                if not isinstance(gt_pcd_xyhl_arr, list) or len(gt_pcd_xyhl_arr) == 0:
                    raise ValueError("empty height_bev_saved_list2")

                for j in range(len(gt_pcd_xyhl_arr)):
                    pcd_xyhl = np.zeros((self.dimension[0], self.dimension[1]))
                    gt_pcd_xyhl = gt_pcd_xyhl_arr[j]

                    if isinstance(gt_pcd_xyhl, np.ndarray) and gt_pcd_xyhl.dtype == object:
                        gt_pcd_xyhl = np.vstack(gt_pcd_xyhl)

                    gt_pcd_xyhl = np.asarray(gt_pcd_xyhl, dtype=np.float32)
                    if gt_pcd_xyhl.ndim != 2 or gt_pcd_xyhl.shape[1] < 3:
                        raise ValueError(f"invalid height_bev shape={gt_pcd_xyhl.shape}")

                    gt_pcd_xyhl = torch.from_numpy(gt_pcd_xyhl)
                    pcd_xyhl[gt_pcd_xyhl[:, 0].long(), gt_pcd_xyhl[:, 1].long()] = gt_pcd_xyhl[:, -1]
                    pcd_xyhl = torch.from_numpy(pcd_xyhl).unsqueeze(0)
                    pcd_xyhl_list.append(pcd_xyhl)

                if len(pcd_xyhl_list) != results['sequence_length']:
                    raise ValueError(
                        f"height_bev length mismatch got={len(pcd_xyhl_list)} "
                        f"expected={results['sequence_length']}"
                    )

                results['height'] = torch.cat(pcd_xyhl_list, dim=0)
                load_ok = True
            except Exception as e:
                print(f"[BAD_NPZ] {height_bev_path}.npz err={repr(e)}", flush=True)
                try:
                    os.remove(height_bev_path + ".npz")
                except Exception:
                    pass
                need_regen = True
        
        if not load_ok:
            results['gt_occ'], pcd_xyhl_saved_list = self.get_seq_occ(results, only_gt_occ=False)
            results['height'] = torch.cat(pcd_xyhl_saved_list, dim=0)

            sequence_length = results['sequence_length']
            height_bev_saved_list = []
            for self.counter in range(sequence_length):
                height_bev = pcd_xyhl_saved_list[self.counter]

                x_grid_bev = torch.linspace(0, self.dimension[0]-1, self.dimension[0], dtype=torch.long)
                x_grid_bev = x_grid_bev.view(self.dimension[0], 1).expand(self.dimension[0], self.dimension[1])
                y_grid_bev = torch.linspace(0, self.dimension[1]-1, self.dimension[1], dtype=torch.long)
                y_grid_bev = y_grid_bev.view(1, self.dimension[1]).expand(self.dimension[0], self.dimension[1])
                
                height_bev_for_save = torch.stack((x_grid_bev, y_grid_bev), -1)
                height_bev_for_save = height_bev_for_save.view(-1, 2)
                height_bev_label = height_bev.view(-1,1)
                height_bev_for_save = torch.cat((height_bev_for_save, height_bev_label), dim=-1)
                kept = height_bev_for_save[:,-1]!=0
                height_bev_for_save= height_bev_for_save[kept]
                height_bev_saved_list.append(height_bev_for_save)

            if self.write_height_cache:
                height_bev_saved_list2 = [item.cpu().detach().numpy() for item in height_bev_saved_list]
                # np.savez(height_bev_path, height_bev_saved_list2)
                obj_pcd_height = np.array(height_bev_saved_list2, dtype=object)
                self.atomic_savez(height_bev_path, height_bev_saved_list2=obj_pcd_height)

        # print('Loading occupancy finish!')

        if self.load_occ_dt:
            results["occ_dt"] = self.get_seq_occ_dt(results)

        return results

    def voxel2world(self, voxel):
        """
        voxel: [N, 3]
        """
        return voxel * self.voxel_size[None, :] + self.pc_range[:3][None, :]

    def world2voxel(self, world):
        """
        world: [N, 3]
        """
        return (world - self.pc_range[:3][None, :]) / self.voxel_size[None, :]

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}'
        return repr_str

    def atomic_savez(self, path_no_ext, **kwargs):
        final = path_no_ext + ".npz"
        tmp_base = final + f".tmp.{os.getpid()}.{time.time_ns()}"
        tmp_file = tmp_base + ".npz"
        try:
            np.savez(tmp_base, **kwargs)  # 실제 파일은 tmp_base + ".npz"로 생성됨
            os.replace(tmp_file, final)
        finally:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        
        # from lidar to camera
        points = points.reshape(-1, 1, 3)
        points = points - trans.reshape(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        
        # from camera to raw pixel
        points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        
        # from raw pixel to transformed pixel
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        
        return points_uvd
    
# b1:boolean, u1: uint8, i2: int16, u2: uint16
@nb.jit('b1[:](i2[:,:],u2[:,:],b1[:])', nopython=True, cache=True, parallel=False)
def nb_process_img_points(basic_valid_occ, depth_canva, nb_valid_mask):
    # basic_valid_occ M 3
    # depth_canva H W
    # label_size = M   # for original occ, small: 2w mid: ~8w base: ~30w
    canva_idx = -1 * np.ones_like(depth_canva, dtype=np.int16)
    for i in range(basic_valid_occ.shape[0]):
        occ = basic_valid_occ[i]
        if occ[2] < depth_canva[occ[1], occ[0]]:
            if canva_idx[occ[1], occ[0]] != -1:
                nb_valid_mask[canva_idx[occ[1], occ[0]]] = False

            canva_idx[occ[1], occ[0]] = i
            depth_canva[occ[1], occ[0]] = occ[2]
            nb_valid_mask[i] = True
    return nb_valid_mask

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label_withvel(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label

# u1: uint8, u8: uint16, i8: int64
@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    
    return processed_label
