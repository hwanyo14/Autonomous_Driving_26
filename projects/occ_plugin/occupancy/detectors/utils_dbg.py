import torch
import torch.distributed as dist


class EfficientOCFDebugMixin:
    """Debugging and diagnostic methods for EfficientOCF."""

    def _grad_l2_wrt_params(self, loss_scalar: torch.Tensor, params):
        if (params is None) or (len(params) <= 0):
            return loss_scalar.detach().new_tensor(0.0)
        grads = torch.autograd.grad(
            loss_scalar,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        sq = None
        for g in grads:
            if g is None:
                continue
            v = g.detach().float().pow(2).sum()
            sq = v if sq is None else (sq + v)
        if sq is None:
            return loss_scalar.detach().new_tensor(0.0)
        return sq.sqrt()

    @staticmethod
    def _namespace_dbg_logs(losses: dict) -> dict:
        """Move dbg_* metrics under dbg/... namespace for cleaner TensorBoard grouping."""
        if not isinstance(losses, dict):
            return losses
        remap = []
        for key in list(losses.keys()):
            if isinstance(key, str) and key.startswith("dbg_"):
                remap.append((key, f"dbg/{key[4:]}"))
        for old_key, new_key in remap:
            losses[new_key] = losses.pop(old_key)
        return losses

    def _compute_gt2p_cooldown_scale(self, train_iter: int = None) -> float:
        """
        Linearly decay gt2p loss scale from 1.0 to min_scale.
        Uses query-train iteration (1-based in logs), converted to zero-based for intuitive config.
        """
        if not self.query_gt2p_cooldown_enabled:
            return 1.0

        it1 = int(self._query_train_iter if train_iter is None else train_iter)
        it0 = max(0, it1 - 1)  # convert to zero-based iteration
        start = int(self.query_gt2p_cooldown_start_iter)
        dur = int(self.query_gt2p_cooldown_iters)
        min_s = float(self.query_gt2p_cooldown_min_scale)

        if it0 <= start:
            return 1.0
        if dur <= 0:
            return min_s

        progress = float(it0 - start) / float(dur)
        progress = min(max(progress, 0.0), 1.0)
        scale = 1.0 - progress
        scale = max(min_s, scale)
        return float(scale)

    def _maybe_add_debug_grad_norms(
        self,
        losses: dict,
        gmo_dice_loss: dict = None,
        gt2p_unlabeled_loss: dict = None,
        gt2p_instance_labeled_loss: dict = None,
        inter_query_repel_loss: dict = None,
        dt_loss: dict = None,
    ) -> None:
        """Collect per-loss grad norms only when explicitly enabled."""
        if not self.debug_loss_grad_enabled:
            return
        if self.debug_loss_grad_every <= 0:
            return

        self._dbg_grad_iter += 1
        if (self._dbg_grad_iter % self.debug_loss_grad_every) != 0:
            return

        center_params = [p for p in self.query_head.center_head.parameters() if p.requires_grad]
        gaussian_module = getattr(self.query_head, "gaussian_head", None)
        gaussian_params = []
        if gaussian_module is not None:
            gaussian_params = [p for p in gaussian_module.parameters() if p.requires_grad]
        cls_module = getattr(self.query_head, "cls_head", None)
        cls_params = []
        if cls_module is not None:
            cls_params = [p for p in cls_module.parameters() if p.requires_grad]
        objectness_module = getattr(self.query_head, "objectness_head", None)
        objectness_params = []
        if objectness_module is not None:
            objectness_params = [p for p in objectness_module.parameters() if p.requires_grad]
        query_params = [p for p in self.query_head.parameters() if p.requires_grad]
        transformer_query_params = []
        transformer_query = getattr(self.transformer, "query", None)
        if isinstance(transformer_query, torch.nn.Parameter) and transformer_query.requires_grad:
            transformer_query_params = [transformer_query]

        def _pick_scalar(primary: dict, key: str):
            if isinstance(primary, dict):
                v = primary.get(key, None)
                if torch.is_tensor(v):
                    return v
            v = losses.get(key, None)
            if torch.is_tensor(v):
                return v
            return None

        probe = {}
        loss_dice = _pick_scalar(gmo_dice_loss, "loss_gmo_dice")
        if loss_dice is not None:
            probe["dice"] = loss_dice
        loss_gmo_focal = _pick_scalar(gmo_dice_loss, "loss_gmo_focal")
        if loss_gmo_focal is not None:
            probe["gmo_focal"] = loss_gmo_focal
        loss_gmo_bce = _pick_scalar(gmo_dice_loss, "loss_gmo_bce")
        if loss_gmo_bce is not None:
            probe["gmo_bce"] = loss_gmo_bce
        loss_gt2p = _pick_scalar(gt2p_unlabeled_loss, "loss_query_gt2p_unlabeled")
        if loss_gt2p is not None:
            probe["gt2p"] = loss_gt2p
        loss_gt2p_inst = _pick_scalar(gt2p_instance_labeled_loss, "loss_query_gt2p_instance_labeled")
        if loss_gt2p_inst is not None:
            probe["gt2p_inst"] = loss_gt2p_inst
        loss_repel = _pick_scalar(inter_query_repel_loss, "loss_query_intra_repel")
        if loss_repel is not None:
            probe["repel"] = loss_repel
        loss_objectness = _pick_scalar(None, "loss_query_objectness")
        if loss_objectness is not None:
            probe["objectness"] = loss_objectness
        loss_cls = _pick_scalar(None, "loss_query_cls")
        if loss_cls is not None:
            probe["cls"] = loss_cls
        loss_center_match = _pick_scalar(None, "loss_query_center_match")
        if loss_center_match is None:
            loss_center_match = _pick_scalar(None, "loss_query_inst_center_match")
        if loss_center_match is not None:
            probe["center_match"] = loss_center_match
        loss_feat_align = _pick_scalar(None, "loss_query_feat_align")
        if loss_feat_align is not None:
            probe["feat_align"] = loss_feat_align
        loss_self_bev_align = _pick_scalar(None, "loss_query_self_bev_align")
        if loss_self_bev_align is not None:
            probe["self_bev_align"] = loss_self_bev_align
        loss_decor = _pick_scalar(None, "loss_query_decor")
        if loss_decor is not None:
            probe["decor"] = loss_decor

        if self.debug_loss_grad_include_dt:
            loss_dt = _pick_scalar(dt_loss, "loss_query_dt")
            if loss_dt is not None:
                probe["dt"] = loss_dt

        if len(probe) <= 0:
            return

        for name, loss_scalar in probe.items():
            g_center = self._grad_l2_wrt_params(loss_scalar, center_params)
            g_gaussian = self._grad_l2_wrt_params(loss_scalar, gaussian_params)
            g_cls = self._grad_l2_wrt_params(loss_scalar, cls_params)
            g_objectness = self._grad_l2_wrt_params(loss_scalar, objectness_params)
            g_query = self._grad_l2_wrt_params(loss_scalar, query_params)
            g_transformer_query = self._grad_l2_wrt_params(loss_scalar, transformer_query_params)
            # "loss" prefix는 total loss 합산에 포함되므로 debug key로 분리.
            losses[f"dbg_grad_center_{name}"] = g_center
            losses[f"dbg_grad_gaussian_{name}"] = g_gaussian
            losses[f"dbg_grad_cls_{name}"] = g_cls
            losses[f"dbg_grad_objectness_{name}"] = g_objectness
            losses[f"dbg_grad_query_{name}"] = g_query
            losses[f"dbg_grad_transformer_query_{name}"] = g_transformer_query

    def _is_main_process(self) -> bool:
        if not dist.is_available():
            return True
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def _maybe_debug_print_segmentation_cls_instance3d(self, seg_cls_inst_tcxzy=None):
        if not self.debug_print_segmentation_cls_instance3d:
            return
        if self._dbg_printed_segmentation_cls_instance3d:
            return
        if seg_cls_inst_tcxzy is None:
            return

        if seg_cls_inst_tcxzy.dim() != 5 or seg_cls_inst_tcxzy.shape[1] != 2:
            raise ValueError(
                "debug segmentation_cls_instance3d expects normalized [T,2,X,Y,Z], "
                f"got {tuple(seg_cls_inst_tcxzy.shape)}"
            )

        seg_cls_inst_tcxzy = seg_cls_inst_tcxzy.detach()
        cls_txyz = seg_cls_inst_tcxzy[:, 0]
        inst_txyz = seg_cls_inst_tcxzy[:, 1]

        cls_occ = cls_txyz > 0
        inst_occ = inst_txyz > 0
        both_occ = cls_occ & inst_occ
        cls_only = cls_occ & (~inst_occ)
        inst_only = inst_occ & (~cls_occ)

        print(
            "[EfficientOCF] segmentation_cls_instance3d "
            f"shape={tuple(seg_cls_inst_tcxzy.shape)} "
            f"dtype={seg_cls_inst_tcxzy.dtype} "
            f"cls_nonzero={int(cls_occ.sum().item())} "
            f"inst_nonzero={int(inst_occ.sum().item())} "
            f"overlap={int(both_occ.sum().item())} "
            f"cls_only={int(cls_only.sum().item())} "
            f"inst_only={int(inst_only.sum().item())}",
            flush=True,
        )
        self._dbg_printed_segmentation_cls_instance3d = True
