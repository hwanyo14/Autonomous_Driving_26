import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class CrossAttentionModule(nn.Module):
    def __init__(self, 
                 embed_dim,
                 feat_dim, 
                 num_heads):
        super(CrossAttentionModule, self).__init__()
        
        self.embed_dim = embed_dim
        self.feat_dim = feat_dim
        self.num_heads = num_heads

        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.embed_dim, 
                                                    num_heads=self.num_heads, 
                                                    kdim=self.feat_dim,
                                                    vdim=self.feat_dim,
                                                    batch_first=True)
        

    def forward(self, query, key, value, need_weights=False, average_attn_weights=True):
        # Define the forward pass
        attn_output, attn_weights = self.multihead_attn(query, 
                                                        key, 
                                                        value,
                                                        need_weights=need_weights,
                                                        average_attn_weights=average_attn_weights)
        return attn_output, attn_weights


class TransformerModule(nn.Module):
    def __init__(self, 
                 feat_dim, 
                 geo_input_dim,
                 out_dim=None, 
                 num_queries=100, 
                 num_heads=4,
                 num_layers=1,
                 kv_resolutions=None,
                 num_cams=6,
                 embed_dim=128,
                 max_time=3,
                 query_id_reinject_scale=0.0,
                 ca_kv_identity_init=False,
                 ca_attn_tau=1.0,
                 query_decor_loss_weight=0.0,
                 query_decor_eps=1e-6,
                 query_attn_overlap_loss_weight=0.0,
                 query_attn_overlap_eps=1e-6,
                 attn_vis_every=0,
                 attn_vis_dir="./work_dirs/query_attn_vis",
                 attn_vis_max_frames=3,
                 attn_vis_max_queries=8,
                 attn_vis_overlay_alpha=0.45,
                 use_e_para=True,
                 use_e_cam=True,
                 use_e_time=True,
                 use_e_pos=True):
        super(TransformerModule, self).__init__()
        self.feat_dim = feat_dim
        self.num_queries = num_queries
        self.out_dim = out_dim
        self.geo_input_dim = geo_input_dim
        self.num_heads = num_heads
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        self.kv_resolutions = self._normalize_kv_resolutions(kv_resolutions)
        self.num_cams = num_cams
        self.embed_dim = embed_dim
        self.max_time = max_time
        self.query_id_reinject_scale = float(query_id_reinject_scale)
        self.ca_kv_identity_init = bool(ca_kv_identity_init)
        self.ca_attn_tau = float(ca_attn_tau)
        if self.ca_attn_tau <= 0.0:
            raise ValueError(f"ca_attn_tau must be positive, got {self.ca_attn_tau}")
        self.query_decor_loss_weight = float(query_decor_loss_weight)
        if self.query_decor_loss_weight < 0.0:
            raise ValueError(
                f"query_decor_loss_weight must be non-negative, got {self.query_decor_loss_weight}"
            )
        self.query_decor_eps = float(query_decor_eps)
        if self.query_decor_eps <= 0.0:
            raise ValueError(f"query_decor_eps must be positive, got {self.query_decor_eps}")
        self.query_attn_overlap_loss_weight = float(query_attn_overlap_loss_weight)
        if self.query_attn_overlap_loss_weight < 0.0:
            raise ValueError(
                "query_attn_overlap_loss_weight must be non-negative, "
                f"got {self.query_attn_overlap_loss_weight}"
            )
        self.query_attn_overlap_eps = float(query_attn_overlap_eps)
        if self.query_attn_overlap_eps <= 0.0:
            raise ValueError(f"query_attn_overlap_eps must be positive, got {self.query_attn_overlap_eps}")
        self.attn_vis_every = int(attn_vis_every)
        self.attn_vis_dir = str(attn_vis_dir)
        self.attn_vis_max_frames = int(attn_vis_max_frames)
        self.attn_vis_max_queries = int(attn_vis_max_queries)
        self.attn_vis_overlay_alpha = float(attn_vis_overlay_alpha)
        self._attn_vis_step = 0
        self._train_iter_synced = False
        self.use_e_para = bool(use_e_para)
        self.use_e_cam = bool(use_e_cam)
        self.use_e_time = bool(use_e_time)
        self.use_e_pos = bool(use_e_pos)

        self.query = nn.Parameter(torch.randn(self.num_queries, self.embed_dim), requires_grad=True)
        self.query_id_embed = nn.Embedding(num_embeddings=self.num_queries, embedding_dim=self.embed_dim)
        self.cam_id_embed = nn.Embedding(num_embeddings=num_cams, embedding_dim=self.feat_dim)
        self.time_embed = nn.Embedding(num_embeddings=max_time, embedding_dim=self.feat_dim)

        self.ca_layers = nn.ModuleList([
            CrossAttentionModule(
                embed_dim=self.embed_dim,
                feat_dim=self.feat_dim,
                num_heads=self.num_heads,
            )
            for _ in range(self.num_layers)
        ])
        if self.ca_kv_identity_init:
            self._init_ca1_kv_identity()
        self.sa_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                batch_first=True,
            )
            for _ in range(self.num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2),
                nn.GELU(),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
            )
            for _ in range(self.num_layers)
        ])

        self.q_norm1_layers = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_layers)])
        self.q_norm2_layers = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_layers)])
        self.q_norm3_layers = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(self.num_layers)])
        self.token_ln = nn.LayerNorm(self.feat_dim)
        self.q_anchor_norm_layers = nn.ModuleList([nn.LayerNorm(self.feat_dim) for _ in range(self.num_layers)])

        # ===== DBG knobs =====
        self.dbg_enable = False
        self.dbg_interval = 50
        self.dbg_iter = 0

        # embedding ablation switches (진단 3에서 사용)
        self.dbg_disable_e_cam = False
        self.dbg_disable_e_time = False
        self.dbg_disable_e_pos = False

        # grad hook storage (진단 2에서 사용)
        self._dbg_grad = {}
        self.last_ca1_attn_weights = None
        self.last_query_context_pool = None
        self.last_query_decor_loss = None
        self.last_query_decor_loss_raw = None
        self.last_query_attn_overlap_loss = None
        self.last_query_attn_overlap_loss_raw = None
        self._install_debug_grad_hooks()


    def _normalize_kv_resolutions(self, kv_resolutions):
        if kv_resolutions is None:
            return None
        kv_resolutions = tuple(tuple(int(v) for v in hw) for hw in kv_resolutions)
        if len(kv_resolutions) != self.num_layers:
            raise ValueError(
                "kv_resolutions length must match num_layers: "
                f"got {len(kv_resolutions)} vs {self.num_layers}"
            )
        for h, w in kv_resolutions:
            if h <= 0 or w <= 0:
                raise ValueError(f"kv_resolutions must be positive, got {(h, w)}")
        return kv_resolutions

    def _build_layer_kv_tokens(self, x, t, ncam, c, h, w):
        if self.kv_resolutions is None:
            s = int(ncam) * int(h) * int(w)
            return [x.reshape(t, s, c) for _ in range(self.num_layers)]

        x_hw = x.reshape(t, ncam, h, w, c).permute(0, 1, 4, 2, 3).reshape(t * ncam, c, h, w)
        layer_kv_tokens = []
        for target_h, target_w in self.kv_resolutions:
            if target_h == h and target_w == w:
                pooled = x_hw
            elif (target_h < h and target_w < w and h % target_h == 0 and w % target_w == 0):
                kernel = (h // target_h, w // target_w)
                pooled = F.avg_pool2d(x_hw, kernel_size=kernel, stride=kernel)
            else:
                pooled = F.adaptive_avg_pool2d(x_hw, output_size=(target_h, target_w))
            pooled = pooled.reshape(t, ncam, c, target_h, target_w)
            layer_kv_tokens.append(pooled.permute(0, 1, 3, 4, 2).reshape(t, ncam * target_h * target_w, c))
        return layer_kv_tokens

    def _is_rank0(self):
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    def set_train_iteration(self, train_iter: int, one_based: bool = True) -> None:
        step = int(train_iter)
        if not one_based:
            step += 1
        step = max(0, step)
        self._attn_vis_step = step
        self.dbg_iter = step
        self._train_iter_synced = True

    def _get_train_iteration(self, advance_if_unsynced: bool = False) -> int:
        if not self._train_iter_synced and advance_if_unsynced:
            self._attn_vis_step += 1
        step = int(self._attn_vis_step)
        self.dbg_iter = step
        return step

    @staticmethod
    def _fill_identity_like_(weight: torch.Tensor) -> None:
        """
        Initialize a 2D projection matrix as identity-like.
        For rectangular matrices, fills the main diagonal up to min(out_dim, in_dim).
        """
        if (not torch.is_tensor(weight)) or (weight.dim() != 2):
            return
        weight.zero_()
        diag = int(min(weight.shape[0], weight.shape[1]))
        if diag <= 0:
            return
        idx = torch.arange(diag, device=weight.device)
        weight[idx, idx] = 1.0

    @torch.no_grad()
    def _init_ca1_kv_identity(self) -> None:
        """
        Initialize CA1 key/value projection weights (W_k, W_v) as identity-like.
        Keeps query projection and out projection unchanged.
        """
        for ca_mod in self.ca_layers:
            self._init_mha_kv_identity(ca_mod.multihead_attn)

    @torch.no_grad()
    def _init_mha_kv_identity(self, mha: nn.MultiheadAttention) -> None:
        if not isinstance(mha, nn.MultiheadAttention):
            return

        in_proj_weight = getattr(mha, "in_proj_weight", None)
        if torch.is_tensor(in_proj_weight):
            e = int(mha.embed_dim)
            if in_proj_weight.dim() == 2 and in_proj_weight.shape[0] >= (3 * e):
                # in_proj_weight layout: [W_q; W_k; W_v]
                self._fill_identity_like_(in_proj_weight[e:2 * e, :])
                self._fill_identity_like_(in_proj_weight[2 * e:3 * e, :])
                in_proj_bias = getattr(mha, "in_proj_bias", None)
                if torch.is_tensor(in_proj_bias) and in_proj_bias.numel() >= (3 * e):
                    in_proj_bias[e:3 * e].zero_()
                return None

        # Separate projection weights path (when q/k/v dims differ).
        k_proj_weight = getattr(mha, "k_proj_weight", None)
        v_proj_weight = getattr(mha, "v_proj_weight", None)
        if torch.is_tensor(k_proj_weight):
            self._fill_identity_like_(k_proj_weight)
        if torch.is_tensor(v_proj_weight):
            self._fill_identity_like_(v_proj_weight)

    @torch.no_grad()
    def _maybe_save_query_attn_vis(
        self,
        attn_tqs: torch.Tensor,
        step: int,
        ncam: int,
        h: int,
        w: int,
        vis_images: torch.Tensor = None,
    ) -> None:
        vis_every = int(getattr(self, "attn_vis_every", 0))
        if vis_every <= 0:
            return
        if (int(step) % vis_every) != 0:
            return
        if not self._is_rank0():
            return
        if (attn_tqs is None) or (attn_tqs.dim() != 3):
            return

        try:
            import os
            from PIL import Image, ImageDraw
        except Exception:
            return

        vis_dir = str(getattr(self, "attn_vis_dir", "./work_dirs/query_attn_vis"))
        os.makedirs(vis_dir, exist_ok=True)

        T, Q, S = attn_tqs.shape
        max_frames = max(1, int(getattr(self, "attn_vis_max_frames", 3)))
        T_vis = min(int(T), max_frames)
        cam_span = int(h) * int(w)

        for t in range(T_vis):
            attn_qs = attn_tqs[t].detach().to(torch.float32).cpu()  # [Q, S]
            if attn_qs.numel() <= 0:
                continue

            # Row-wise normalization for readable contrast across queries.
            row_min = attn_qs.min(dim=1, keepdim=True).values
            row_max = attn_qs.max(dim=1, keepdim=True).values
            attn_norm = (attn_qs - row_min) / (row_max - row_min + 1e-6)
            heat_u8 = (attn_norm * 255.0).clamp(0.0, 255.0).to(torch.uint8).numpy()

            base_img = Image.fromarray(heat_u8, mode="L").convert("RGB")
            target_w = min(2400, max(512, int(base_img.width)))
            target_h = min(1200, max(256, int(base_img.height * 4)))
            if (target_w, target_h) != (base_img.width, base_img.height):
                vis_img = base_img.resize((target_w, target_h), resample=Image.BILINEAR)
            else:
                vis_img = base_img

            pad_top = 24
            canvas = Image.new("RGB", (vis_img.width, vis_img.height + pad_top), color=(255, 255, 255))
            canvas.paste(vis_img, (0, pad_top))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (4, 4),
                f"iter={int(step)} t={int(t)} attn[Q,S]=[{int(Q)},{int(S)}] cams={int(ncam)} hw={int(h)}x{int(w)}",
                fill=(0, 0, 0),
            )

            if cam_span > 0 and int(ncam) > 1:
                for cam_idx in range(1, int(ncam)):
                    src_x = (cam_idx * cam_span) / float(max(1, int(S)))
                    x_line = int(round(src_x * (vis_img.width - 1)))
                    draw.line([(x_line, pad_top), (x_line, pad_top + vis_img.height - 1)], fill=(255, 0, 0), width=1)

            out_path = os.path.join(vis_dir, f"iter_{int(step):06d}_t{int(t):02d}.png")
            canvas.save(out_path)

        if torch.is_tensor(vis_images) and vis_images.dim() == 5:
            self._maybe_save_query_attn_overlay_vis(
                attn_tqs=attn_tqs,
                vis_images=vis_images,
                step=step,
                ncam=ncam,
                h=h,
                w=w,
            )

    @torch.no_grad()
    def _maybe_save_query_attn_overlay_vis(
        self,
        attn_tqs: torch.Tensor,
        vis_images: torch.Tensor,
        step: int,
        ncam: int,
        h: int,
        w: int,
    ) -> None:
        if (attn_tqs is None) or (attn_tqs.dim() != 3):
            return
        if (vis_images is None) or (vis_images.dim() != 5):
            return
        try:
            import os
            from PIL import Image, ImageDraw
        except Exception:
            return

        vis_dir = str(getattr(self, "attn_vis_dir", "./work_dirs/query_attn_vis"))
        os.makedirs(vis_dir, exist_ok=True)

        T, Q, S = attn_tqs.shape
        if int(S) != int(ncam) * int(h) * int(w):
            return
        T_img, N_img, C_img, H_img, W_img = vis_images.shape
        if T_img <= 0 or N_img <= 0 or H_img <= 0 or W_img <= 0:
            return

        max_frames = max(1, int(getattr(self, "attn_vis_max_frames", 3)))
        max_queries = max(1, int(getattr(self, "attn_vis_max_queries", 8)))
        alpha = float(getattr(self, "attn_vis_overlay_alpha", 0.45))
        alpha = min(0.95, max(0.05, alpha))
        T_vis = min(int(T), int(T_img), max_frames)
        N_vis = min(int(ncam), int(N_img))
        Q_vis = min(int(Q), max_queries)
        if T_vis <= 0 or N_vis <= 0 or Q_vis <= 0:
            return

        # Randomly sample query rows from all Q for overlay visualization.
        if Q_vis < int(Q):
            q_sel = torch.randperm(int(Q), device=attn_tqs.device)[:Q_vis]
            q_sel = torch.sort(q_sel).values
        else:
            q_sel = torch.arange(int(Q), device=attn_tqs.device)
        q_sel_list = [int(v) for v in q_sel.detach().cpu().tolist()]

        img_tnchw = vis_images[:T_vis, :N_vis].detach().to(torch.float32).cpu()
        if int(C_img) == 1:
            img_tnchw = img_tnchw.repeat(1, 1, 3, 1, 1)
        elif int(C_img) >= 3:
            img_tnchw = img_tnchw[:, :, :3]
        else:
            return

        # Per-camera min-max normalization to displayable [0,1] regardless of input normalization.
        img_min = img_tnchw.amin(dim=(2, 3, 4), keepdim=True)
        img_max = img_tnchw.amax(dim=(2, 3, 4), keepdim=True)
        img_tnchw = (img_tnchw - img_min) / (img_max - img_min + 1e-6)
        img_tnchw = img_tnchw.clamp_(0.0, 1.0)

        attn_tqnhw = attn_tqs[:T_vis].view(T_vis, Q, int(ncam), int(h), int(w))
        attn_tqnhw = attn_tqnhw.index_select(dim=1, index=q_sel)[:, :, :N_vis]

        target_cell_w = 256
        target_cell_h = max(64, int(round(float(H_img) * float(target_cell_w) / float(max(1, W_img)))))
        gap = 4
        pad_top = 24
        canvas_w = N_vis * target_cell_w + (N_vis - 1) * gap
        canvas_h = pad_top + Q_vis * target_cell_h + (Q_vis - 1) * gap

        for t in range(T_vis):
            canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (4, 4),
                f"iter={int(step)} t={int(t)} overlay attn[Q,N,H,W]=[{Q_vis},{N_vis},{int(h)},{int(w)}]",
                fill=(0, 0, 0),
            )

            for q_vis_idx, q_global_idx in enumerate(q_sel_list):
                for cam_idx in range(N_vis):
                    base_chw = img_tnchw[t, cam_idx]  # [3,H,W]
                    attn_hw = attn_tqnhw[t, q_vis_idx, cam_idx].to(torch.float32).cpu()  # [h,w]
                    attn_up = F.interpolate(
                        attn_hw.unsqueeze(0).unsqueeze(0),
                        size=(H_img, W_img),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).squeeze(0)

                    a_min = attn_up.min()
                    a_max = attn_up.max()
                    attn_n = (attn_up - a_min) / (a_max - a_min + 1e-6)
                    attn_n = attn_n.clamp(0.0, 1.0)

                    # Red heat overlay.
                    out = base_chw.clone()
                    out[0] = (1.0 - alpha) * out[0] + alpha * attn_n
                    out[1] = (1.0 - alpha) * out[1]
                    out[2] = (1.0 - alpha) * out[2]
                    out_u8 = (out.clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(1, 2, 0).numpy()

                    cell = Image.fromarray(out_u8, mode="RGB")
                    if (cell.width != target_cell_w) or (cell.height != target_cell_h):
                        cell = cell.resize((target_cell_w, target_cell_h), resample=Image.BILINEAR)

                    x0 = cam_idx * (target_cell_w + gap)
                    y0 = pad_top + q_vis_idx * (target_cell_h + gap)
                    canvas.paste(cell, (x0, y0))
                    draw.text((x0 + 2, y0 + 2), f"q={q_global_idx} cam={cam_idx}", fill=(255, 255, 255))

            out_path = os.path.join(vis_dir, f"iter_{int(step):06d}_t{int(t):02d}_overlay.png")
            canvas.save(out_path)

    def set_embedding_usage(self, use_e_para=None, use_e_cam=None, use_e_time=None, use_e_pos=None):
        if use_e_para is not None:
            self.use_e_para = bool(use_e_para)
        if use_e_cam is not None:
            self.use_e_cam = bool(use_e_cam)
        if use_e_time is not None:
            self.use_e_time = bool(use_e_time)
        if use_e_pos is not None:
            self.use_e_pos = bool(use_e_pos)

    def _install_debug_grad_hooks(self):
        # backward 이후가 아니라 backward "중"에 호출됨
        def save_norm(key):
            def _hook(grad):
                if grad is None:
                    return grad
                # 너무 자주 저장하지 않기
                if (self.dbg_iter % self.dbg_interval) == 0:
                    self._dbg_grad[key] = float(grad.detach().norm().cpu())
                return grad
            return _hook

        # query token 자체가 학습되는지
        self.query.register_hook(save_norm("g_query_token"))

    def _compute_query_decor_loss_from_q2(self, q2: torch.Tensor) -> torch.Tensor:
        """
        q2: [B, Q, D]
        Computes cosine Gram off-diagonal squared penalty:
            L = mean_b [ sum_{i!=j} S_ij^2 / (Q*(Q-1)) ],
            S = normalize(X) @ normalize(X)^T
        """
        if (not torch.is_tensor(q2)) or (q2.dim() != 3):
            return self.query.sum() * 0.0
        b, q, _ = [int(v) for v in q2.shape]
        if q <= 1:
            return q2.sum() * 0.0

        x = F.normalize(q2.to(torch.float32), p=2, dim=-1, eps=self.query_decor_eps)  # [B,Q,D]
        sim = torch.bmm(x, x.transpose(1, 2)).clamp(-1.0, 1.0)  # [B,Q,Q]
        eye = torch.eye(q, device=sim.device, dtype=torch.bool).unsqueeze(0)  # [1,Q,Q]
        off = sim.masked_fill(eye, 0.0)
        denom = float(q * (q - 1))
        loss_b = off.pow(2).sum(dim=(1, 2)) / denom
        return loss_b.mean()

    def _compute_query_attn_overlap_loss_from_w1(self, w1: torch.Tensor) -> torch.Tensor:
        if (not torch.is_tensor(w1)) or (w1.dim() not in (3, 4)):
            return self.query.sum() * 0.0

        if w1.dim() == 4:
            attn = w1.mean(dim=1)
        else:
            attn = w1
        b, q, _ = [int(v) for v in attn.shape]
        if q <= 1:
            return attn.sum() * 0.0

        x = F.normalize(attn.to(torch.float32), p=2, dim=-1, eps=self.query_attn_overlap_eps)
        sim = torch.bmm(x, x.transpose(1, 2)).clamp(0.0, 1.0)
        eye = torch.eye(q, device=sim.device, dtype=torch.bool).unsqueeze(0)
        off = sim.masked_fill(eye, 0.0)
        return off.sum(dim=(1, 2)).div(float(q * (q - 1))).mean()

    @staticmethod
    def _format_ca1_attn_weights_tqnhw(
        attn_weights: torch.Tensor,
        t: int,
        ncam: int,
        h: int,
        w: int,
    ):
        """
        Normalize CA1 attention weights into a single shape: [T, Q, Ncam, H, W].
        Returns None on invalid/unexpected shape.
        """
        if not torch.is_tensor(attn_weights):
            return None

        aw = attn_weights
        if aw.dim() == 5:
            # [T, B, Hhead, Q, S] (average_attn_weights=False path)
            if int(aw.shape[1]) != 1:
                return None
            aw = aw[:, 0].mean(dim=1)
        elif aw.dim() == 4:
            # [T, B, Q, S] (average_attn_weights=True path)
            if int(aw.shape[1]) != 1:
                return None
            aw = aw[:, 0]
        elif aw.dim() != 3:
            return None

        if int(aw.shape[0]) != int(t):
            return None
        if aw.numel() <= 0:
            return None

        tt, qq, ss = [int(v) for v in aw.shape]
        expected_s = int(ncam) * int(h) * int(w)
        if ss != expected_s:
            return None

        aw = aw.view(tt, qq, int(ncam), int(h), int(w))
        if not bool(torch.isfinite(aw).all().item()):
            return None
        return aw
        

    def forward(
        self,
        x,
        return_attn_pool=False,
        return_attn_weights=False,
        average_attn_weights=True,
        vis_images=None,
    ):
        '''
            x: img_feat: [T, Ncam, C, H, W]

            query: [T, 1, num_queries, hidden_dim]
            pooled context (optional): [T, num_queries, C]
        '''
        T, Ncam, C, H, W = x.shape
        S = Ncam * H * W
        self.last_query_decor_loss = None
        self.last_query_decor_loss_raw = None
        self.last_query_attn_overlap_loss = None
        self.last_query_attn_overlap_loss_raw = None
        attn_step = self._get_train_iteration(advance_if_unsynced=True)
        attn_vis_now = (
            self.training
            and int(getattr(self, "attn_vis_every", 0)) > 0
            and (attn_step % int(self.attn_vis_every) == 0)
            and self._is_rank0()
        )

        # Interpretability-first source feature: raw context tokens before any transformer embedding.
        src_tokens = x.permute(0, 1, 3, 4, 2).reshape(T, S, C)

        dbg_now = self.training and self.dbg_enable and (self.dbg_iter % self.dbg_interval == 0)

        x, dbg_pack = self.get_cam_pos_embed(x, return_dbg=dbg_now)
        
        
        # Ablation 필요. 껐다 켰다. (img feat을 그대로 재사용하기 위한 목적이면 끄는 게 맞을 듯 함)
        # x = self.token_ln(x)
        layer_kv_tokens = self._build_layer_kv_tokens(x, T, Ncam, C, H, W)

        if dbg_now and self._is_rank0():
            # img token은 "pos+cam+time"이 이미 더해진 x이므로, 순수 img feat norm은 따로 보고 싶으면 get_cam_pos_embed에서 뽑아야 함
            # 여기서는 각 성분의 상대 크기만 확인
            with torch.no_grad():
                # x는 [T,Ncam,HW,C]
                xn = float(x.detach().norm(dim=-1).mean().cpu())
                msg = {
                    "it": self.dbg_iter,
                    "mean_norm_x_after_add": xn,
                }
                if dbg_pack is not None:
                    msg.update(dbg_pack)
            print(f"[DBG][tok-norm] {msg}")

        q0 = self.query.unsqueeze(0)
        query_id_embed = None
        if self.query_id_reinject_scale > 0.0:
            query_ids = torch.arange(self.num_queries, device=x.device)
            query_id_embed = self.query_id_embed(query_ids).unsqueeze(0).to(dtype=q0.dtype)

        q = q0
        q_list = []
        decor_terms = []
        attn_overlap_terms = []
        attn_list = [] if return_attn_weights else None
        attn_vis_list = [] if attn_vis_now else None
        pooled_list = [] if return_attn_pool else None
        use_attn_overlap_loss = self.training and self.query_attn_overlap_loss_weight > 0.0
        need_weights = bool(return_attn_pool or return_attn_weights or attn_vis_now or use_attn_overlap_loss)

        for t in range(T):
            q_in = q
            if query_id_embed is not None:
                q_in = q_in + (self.query_id_reinject_scale * query_id_embed)

            w1 = None
            for layer_idx in range(self.num_layers):
                # softmax(logit / tau) equivalent sharpening via query scaling.
                q_attn = q_in / self.ca_attn_tau

                # CA1
                kv = layer_kv_tokens[layer_idx][t].unsqueeze(0)  # [1, S_layer, C]
                out1, w1 = self.ca_layers[layer_idx](
                    q_attn,
                    kv,
                    kv,
                    need_weights=need_weights,
                    average_attn_weights=average_attn_weights,
                )
                q1 = self.q_norm1_layers[layer_idx](q_in + out1)
                if use_attn_overlap_loss:
                    attn_overlap_terms.append(self._compute_query_attn_overlap_loss_from_w1(w1))

                # SA1
                out2, _ = self.sa_layers[layer_idx](
                    q1, q1, q1, need_weights=False, average_attn_weights=True
                )
                q2 = self.q_norm2_layers[layer_idx](q1 + out2)

                # Decorrelate queries right after SA mixing to reduce query collapse.
                decor_terms.append(self._compute_query_decor_loss_from_q2(q2))

                # FFN
                q3 = self.q_norm3_layers[layer_idx](q2 + self.ffn_layers[layer_idx](q2))
                q_in = self.q_anchor_norm_layers[layer_idx](q3 + q0)

            q_list.append(q_in)
            if return_attn_weights:
                attn_list.append(w1)
            if attn_vis_now:
                if w1 is None:
                    raise RuntimeError("CA1 attention weights are required for visualization but got None.")
                if w1.dim() == 4:
                    attn_qs = w1.mean(dim=1).squeeze(0)  # [Q, S]
                elif w1.dim() == 3:
                    attn_qs = w1.squeeze(0)              # [Q, S]
                else:
                    raise ValueError(f"Unexpected CA1 attention shape for visualization: {tuple(w1.shape)}")
                attn_vis_list.append(attn_qs)
            if return_attn_pool:
                # w1: [1,Q,S] when average_attn_weights=True, else [1,H,Q,S].
                if w1 is None:
                    raise RuntimeError("CA1 attention weights are required for pooling but got None.")
                if w1.dim() == 4:
                    attn_qs = w1.mean(dim=1)  # [1,Q,S]
                elif w1.dim() == 3:
                    attn_qs = w1
                else:
                    raise ValueError(f"Unexpected CA1 attention shape: {tuple(w1.shape)}")
                pooled_qc = torch.bmm(attn_qs, src_tokens[t].unsqueeze(0)).squeeze(0)  # [Q,C]
                pooled_list.append(pooled_qc)
            q = q_in

        q_final = torch.stack(q_list, dim=0)  # [T, num_queries, hidden_dim]
        pooled_tqc = torch.stack(pooled_list, dim=0) if return_attn_pool else None
        attn_weights_raw = torch.stack(attn_list, dim=0) if return_attn_weights else None
        attn_weights = None
        if return_attn_weights and torch.is_tensor(attn_weights_raw):
            attn_weights = self._format_ca1_attn_weights_tqnhw(
                attn_weights=attn_weights_raw,
                t=T,
                ncam=Ncam,
                h=H,
                w=W,
            )
        if attn_vis_now and (attn_vis_list is not None) and (len(attn_vis_list) > 0):
            attn_tqs = torch.stack(attn_vis_list, dim=0)  # [T, Q, S]
            self._maybe_save_query_attn_vis(
                attn_tqs=attn_tqs,
                step=attn_step,
                ncam=Ncam,
                h=H,
                w=W,
                vis_images=vis_images,
            )
        self.last_query_context_pool = pooled_tqc
        self.last_ca1_attn_weights = attn_weights
        if len(decor_terms) > 0:
            decor_raw = torch.stack(decor_terms, dim=0).mean()
        else:
            decor_raw = q_final.sum() * 0.0
        self.last_query_decor_loss_raw = decor_raw
        self.last_query_decor_loss = decor_raw * float(self.query_decor_loss_weight)
        if len(attn_overlap_terms) > 0:
            attn_overlap_raw = torch.stack(attn_overlap_terms, dim=0).mean()
        else:
            attn_overlap_raw = q_final.sum() * 0.0
        self.last_query_attn_overlap_loss_raw = attn_overlap_raw
        self.last_query_attn_overlap_loss = attn_overlap_raw * float(self.query_attn_overlap_loss_weight)

        if dbg_now and self._is_rank0():
            gmsg = {
                "it": self.dbg_iter,
                "g_query_token": self._dbg_grad.get("g_query_token", None),
            }
            print(f"[DBG][grad] {gmsg}")

        if return_attn_pool and return_attn_weights:
            return q_final, pooled_tqc, attn_weights
        if return_attn_pool:
            return q_final, pooled_tqc
        if return_attn_weights:
            return q_final, attn_weights
        return q_final


    # def get_cam_pos_embed(self, x, mlp_input_seq):
    #     assert x.dim() == 5, "x must be [T, Ncam, C, H, W]"
    #     T, Ncam, C, H, W = x.shape

    #     # [T, Ncam, C, H, W] -> [T, Ncam, H*W, C]
    #     x = x.permute(0, 1, 3, 4, 2).reshape(T, Ncam, H * W, C)

    #     # Epos: [H*W, C]
    #     e_pos = self.build_2d_sincos_pos_embed(H, W, C, device=x.device, dtype=x.dtype)
    #     # x = x + e_pos[None, None, :, :]

    #     # Ecam: [Ncam, C]
    #     cam_ids = torch.arange(Ncam, device=x.device)
    #     e_cam = self.cam_id_embed(cam_ids)
    #     x = x + e_cam[None, :, None, :]

    #     # Et: [T, C]
    #     time_ids = torch.arange(T, device=x.device)
    #     e_t = self.time_embed(time_ids)
    #     x = x + e_t[:, None, None, :]

    #     return x

    def get_cam_pos_embed(self, x, return_dbg=False):
        assert x.dim() == 5, "x must be [T, Ncam, C, H, W]"
        T, Ncam, C, H, W = x.shape

        # [T,Ncam,C,H,W] -> [T,Ncam,HW,C]
        x = x.permute(0, 1, 3, 4, 2).reshape(T, Ncam, H * W, C)

        dbg = None
        if return_dbg:
            dbg = {}

        # Epos
        e_pos = None
        if self.use_e_pos:
            e_pos = self.build_2d_sincos_pos_embed(H, W, C, device=x.device, dtype=x.dtype)
            if self.dbg_disable_e_pos:
                e_pos = e_pos * 0.0
            x = x + e_pos[None, None, :, :]

        # Ecam
        e_cam = None
        if self.use_e_cam:
            cam_ids = torch.arange(Ncam, device=x.device)
            e_cam = self.cam_id_embed(cam_ids)  # [Ncam,C]
            if self.dbg_disable_e_cam:
                e_cam = e_cam * 0.0
            x = x + e_cam[None, :, None, :]

        # Et
        e_t = None
        if self.use_e_time:
            time_ids = torch.arange(T, device=x.device)
            e_t = self.time_embed(time_ids)  # [T,C]
            if self.dbg_disable_e_time:
                e_t = e_t * 0.0
            x = x + e_t[:, None, None, :]

        if return_dbg:
            with torch.no_grad():
                dbg["mean_norm_e_cam"] = 0.0 if e_cam is None else float(e_cam.detach().norm(dim=-1).mean().cpu())
                dbg["mean_norm_e_time"] = 0.0 if e_t is None else float(e_t.detach().norm(dim=-1).mean().cpu())
                dbg["mean_norm_e_pos"] = 0.0 if e_pos is None else float(e_pos.detach().norm(dim=-1).mean().cpu())
        return x, dbg


    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B,N,_,_ = rot.shape

        bda_ones = torch.ones((B,N),device=rot.device)
        bda_zeros = torch.zeros((B,N),device=rot.device)
        
        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda_ones,
                bda_zeros,
                bda_zeros,
                bda_ones,
                bda_ones,
            ], dim=-1)
        else:
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                post_rot[:, :, 0, 0],
                post_rot[:, :, 0, 1],
                post_tran[:, :, 0],
                post_rot[:, :, 1, 0],
                post_rot[:, :, 1, 1],
                post_tran[:, :, 1],
                bda_ones,
                bda_zeros,
                bda_zeros,
                bda_ones,
                bda_ones,
            ], dim=-1)
        
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        return mlp_input
    
    def build_2d_sincos_pos_embed(self, h: int, w: int, dim: int, temperature: float = 10000.0,
                                device=None, dtype=None) -> torch.Tensor:
        """
        2D sin-cos positional embedding
        반환 shape: [H*W, C]
        조건: dim % 4 == 0
        """
        assert dim % 4 == 0, f"dim must be divisible by 4, got dim={dim}"

        device = device if device is not None else torch.device("cpu")
        dtype = dtype if dtype is not None else torch.float32

        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=dtype),
            torch.arange(w, device=device, dtype=dtype),
            indexing="ij"
        )  # y,x: [H, W]

        # half of dim for y, half for x, and each uses sin and cos
        dim_each = dim // 4  # for sin and cos of y, and sin and cos of x

        omega = torch.arange(dim_each, device=device, dtype=dtype)
        omega = 1.0 / (temperature ** (omega / dim_each))  # [dim_each]

        # [H, W, dim_each]
        out_y = y[..., None] * omega[None, None, :]
        out_x = x[..., None] * omega[None, None, :]

        pos = torch.cat(
            [torch.sin(out_y), torch.cos(out_y), torch.sin(out_x), torch.cos(out_x)],
            dim=-1
        )  # [H, W, dim]

        pos = pos.reshape(h * w, dim)  # [H*W, C]
        return pos
    


if __name__ == "__main__":
    model = TransformerModule(
        feat_dim=256,
        out_dim=256,
        geo_input_dim=27,
        num_queries=100,
        num_heads=4,
        num_cams=6,
        embed_dim=128,
        max_time=3
    )

    x = torch.randn(3, 6, 256, 28, 50)  # [T, Ncam, C, H, W]
    output = model(x)
    # print(output.shape)
