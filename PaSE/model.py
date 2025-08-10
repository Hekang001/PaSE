# model.py
"""
PaSE utilities:
- PrototypeBank (per-modality prototypes with EMA updates)
- Entropic OT (Sinkhorn) and EntropicOTAligner (pairwise L_inter)
- Prototype-aware gating fusion (PGF) — integrates with existing router outputs
- Shapley-value based gradient modulation support (SGM) with helper to get param-groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from modules.Attention_softmoe import Block, Mlp  

class PrototypeBank:
    """Per-modality prototype bank with exponential moving average updates.
    Prototypes are class-level vectors: prototypes[c] in R^D.
    """
    def __init__(self, num_classes: int, dim: int, momentum: float = 0.98, device='cpu'):
        self.num_classes = int(num_classes)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.device = torch.device(device)
        # initialize zero prototypes; will be initialized from first batch if needed
        self.prototypes = torch.zeros(self.num_classes, self.dim, device=self.device)
        self.initialized = False

    @torch.no_grad()
    def init_from_batch(self, feats: torch.Tensor, labels: torch.Tensor):

        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                self.prototypes[c] = feats[mask].mean(dim=0).to(self.device)
        self.initialized = True

    @torch.no_grad()
    def momentum_update(self, feats: torch.Tensor, labels: torch.Tensor):
        """EMA update prototypes with batch features.
        """
        if not self.initialized:
            self.init_from_batch(feats, labels)
            return
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.any():
                mean_c = feats[mask].mean(dim=0).to(self.device)
                self.prototypes[c] = self.momentum * self.prototypes[c] + (1.0 - self.momentum) * mean_c

    def get(self):
        return self.prototypes

    def to(self, device):
        self.device = torch.device(device)
        self.prototypes = self.prototypes.to(self.device)


def sinkhorn(a: torch.Tensor, b: torch.Tensor, C: torch.Tensor, reg: float = 0.05, num_iters: int = 200):
    """Simple Sinkhorn implementation.
    returns transport plan Q [K,K]
    """
    K = C.shape[0]
    Kmat = torch.exp(-C / reg) + 1e-9  # avoid exact zeros
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(num_iters):
        u = a / (Kmat @ v)
        v = b / (Kmat.t() @ u)
    Q = torch.diag(u) @ Kmat @ torch.diag(v)
    Q = Q / Q.sum()
    return Q


class EntropicOTAligner:
    """Encapsulate pairwise entropic OT alignment between two prototype sets P & Q.
    Returns L_match, L_reg (consistency), Omega (structure-preserve), and combined L_inter.
    """
    def __init__(self, reg: float = 0.05, sinkhorn_iters: int = 200, device='cpu'):
        self.reg = float(reg)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.device = torch.device(device)

    def compute_cost(self, P: torch.Tensor, Q: torch.Tensor):
        # P: [K, D], Q: [K, D] -> C: [K, K] (squared euclidean)
        diff = P.unsqueeze(1) - Q.unsqueeze(0)  # [K, K, D]
        C = (diff ** 2).sum(dim=2)
        return C

    def align(self, P: torch.Tensor, Q: torch.Tensor, alpha: float = 0.1, beta: float = 0.05):
        K = P.shape[0]
        a = torch.full((K,), 1.0 / K, device=self.device)
        b = torch.full((K,), 1.0 / K, device=self.device)
        C_pq = self.compute_cost(P, Q).to(self.device)
        C_qp = C_pq.t()
        Q_pq = sinkhorn(a, b, C_pq, reg=self.reg, num_iters=self.sinkhorn_iters)
        Q_qp = sinkhorn(a, b, C_qp, reg=self.reg, num_iters=self.sinkhorn_iters)
        L_match = 0.5 * (torch.sum(Q_pq * C_pq) + torch.sum(Q_qp * C_qp))
        L_reg = torch.norm(Q_pq - Q_qp.t(), p='fro') ** 2
        Omega = 0.5 * (torch.norm(Q_pq - torch.eye(K, device=self.device), p='fro') ** 2 +
                       torch.norm(Q_qp - torch.eye(K, device=self.device), p='fro') ** 2)
        L_inter = L_match + alpha * L_reg + beta * Omega
        return {"Q_pq": Q_pq, "Q_qp": Q_qp, "L_match": L_match, "L_reg": L_reg, "Omega": Omega, "L_inter": L_inter}


class ShapleySGM:
    """Approximate Shapley values via Monte Carlo permutations and produce modulation phi.
    utility_fn: callable f(set_of_modalities) -> scalar utility (higher = better)
    """
    def __init__(self, modalities: List[str], utility_fn, n_permutations: int = 200, device='cpu'):
        self.modalities = list(modalities)
        self.n_permutations = int(n_permutations)
        self.utility_fn = utility_fn
        self.device = torch.device(device)

    def estimate_shapley(self):
        vals = {m: 0.0 for m in self.modalities}
        for _ in range(self.n_permutations):
            perm = self.modalities.copy()
            random.shuffle(perm)
            cur_set = set()
            cur_u = self.utility_fn(cur_set)
            for m in perm:
                prev_u = cur_u
                cur_set.add(m)
                cur_u = self.utility_fn(cur_set)
                marginal = cur_u - prev_u
                vals[m] += marginal
        for m in vals:
            vals[m] = vals[m] / float(self.n_permutations)
        return vals

    def modulation_factors(self, shapley_vals: Dict[str, float]):
        arr = torch.tensor([max(1e-9, shapley_vals[m]) for m in self.modalities], device=self.device)
        arr_norm = arr / arr.sum()
        psi_min = arr_norm.min()
        eps = 1e-9
        phi = torch.exp((psi_min + eps) / (arr_norm + eps) - 1.0)
        return {m: float(phi[i].item()) for i, m in enumerate(self.modalities)}


class PaSE(nn.Module):
    """
    PaSE utilities.
      - prototype banks per modality: self.prototype_banks = {'audio': PrototypeBank(...), ...}
      - aligner: self.aligner = EntropicOTAligner(...)
      - PGF: use prototype similarity for gating if requested (use_prototype_gate)
      - SGM helpers: method to estimate shapley phi and param-group mapping for gradient modulation
    """

    def __init__(self, args, adim, tdim, vdim, D_e, n_classes,
                 depth=4, num_heads=4, mlp_ratio=1, drop_rate=0, attn_drop_rate=0, no_cuda=False):
        super(PaSE, self).__init__()

        self.n_classes = n_classes
        self.D_e = D_e
        self.num_heads = num_heads
        D = 3 * D_e
        self.device = args.device
        self.no_cuda = no_cuda
        self.adim, self.tdim, self.vdim = adim, tdim, vdim
        self.out_dropout = args.drop_rate

        # input projections
        self.a_in_proj = nn.Sequential(nn.Linear(self.adim, D_e))
        self.t_in_proj = nn.Sequential(nn.Linear(self.tdim, D_e))
        self.v_in_proj = nn.Sequential(nn.Linear(self.vdim, D_e))
        self.dropout_a = nn.Dropout(args.drop_rate)
        self.dropout_t = nn.Dropout(args.drop_rate)
        self.dropout_v = nn.Dropout(args.drop_rate)

        # block = transformer-like processing per modality (reuse Block)
        self.block = Block(
            dim=D_e,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            depth=depth,
        )
        self.proj1 = nn.Linear(D, D)
        self.nlp_head_a = nn.Linear(D_e, n_classes)
        self.nlp_head_t = nn.Linear(D_e, n_classes)
        self.nlp_head_v = nn.Linear(D_e, n_classes)
        self.nlp_head = nn.Linear(D, n_classes)


        self.prototype_banks: Dict[str, PrototypeBank] = {}
        self.aligner = EntropicOTAligner(reg=0.05, sinkhorn_iters=200, device=self.device)
        self.prototype_gate_net = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.modalities = ['audio', 'text', 'vision']

    def init_prototype_banks(self, num_classes: Optional[int] = None, device: Optional[str] = None, momentum: float = 0.98):

        if num_classes is None:
            num_classes = self.n_classes
        if device is None:
            device = self.device
        De = self.D_e
        self.prototype_banks = {
            'audio': PrototypeBank(num_classes=num_classes, dim=De, momentum=momentum, device=device),
            'text': PrototypeBank(num_classes=num_classes, dim=De, momentum=momentum, device=device),
            'vision': PrototypeBank(num_classes=num_classes, dim=De, momentum=momentum, device=device),
        }

    def forward(self, inputfeats, input_features_mask=None, umask=None, first_stage=False,
                use_prototype_gate: bool = False, labels_for_gate: Optional[torch.Tensor] = None,
                update_prototypes: bool = True):
        """
        Extended forward:
        - use_prototype_gate: if True, combine original router weights with prototype similarity gates
          (when labels_for_gate is provided, use ground-truth class prototype for similarity;
           when labels_for_gate is None, use max proto similarity across classes)
        - update_prototypes: if True and labels_for_gate provided, update prototype banks with per-sample features

        Returns:
          hidden, out, out_a, out_t, out_v, weight_save
        """
        weight_save = []
        audio, text, video = inputfeats[:, :, :self.adim], inputfeats[:, :, self.adim:self.adim + self.tdim], \
                             inputfeats[:, :, self.adim + self.tdim:]
        seq_len, B, C = audio.shape
        audio, text, video = audio.permute(1, 0, 2), text.permute(1, 0, 2), video.permute(1, 0, 2)
        proj_a = self.dropout_a(self.a_in_proj(audio))
        proj_t = self.dropout_t(self.t_in_proj(text))
        proj_v = self.dropout_v(self.v_in_proj(video))

        input_mask = torch.clone(input_features_mask.permute(1, 0, 2))
        input_mask[umask == 0] = 0
        attn_mask = input_mask.transpose(1, 2).reshape(B, -1)


        weight_a = torch.softmax(proj_a, dim=-1)
        weight_t = torch.softmax(proj_t, dim=-1)
        weight_v = torch.softmax(proj_v, dim=-1)
        weight_save.append(np.array([weight_a.cpu().detach().numpy(), weight_t.cpu().detach().numpy(), weight_v.cpu().detach().numpy()]))
    

        x_a = self.block(proj_a, first_stage, attn_mask, 'a')
        x_t = self.block(proj_t, first_stage, attn_mask, 't')
        x_v = self.block(proj_v, first_stage, attn_mask, 'v')

        if first_stage:
            out_a = self.nlp_head_a(x_a)
            out_t = self.nlp_head_t(x_t)
            out_v = self.nlp_head_v(x_v)
            x = torch.cat([x_a, x_t, x_v], dim=1)
        else:
            # original combination via soft-MoE routing (three experts stacked)
            try:
                x_unweighted_a = x_a.reshape(B, seq_len, 3, self.D_e)
                x_unweighted_t = x_t.reshape(B, seq_len, 3, self.D_e)
                x_unweighted_v = x_v.reshape(B, seq_len, 3, self.D_e)
            except Exception:
                # fallback: if block produced [B, seq_len, D_e], replicate last dim
                x_unweighted_a = x_a.unsqueeze(2).repeat(1, 1, 3, 1)
                x_unweighted_t = x_t.unsqueeze(2).repeat(1, 1, 3, 1)
                x_unweighted_v = x_v.unsqueeze(2).repeat(1, 1, 3, 1)
            
            proto_gate_a = None
            proto_gate_t = None
            proto_gate_v = None

            if use_prototype_gate and len(self.prototype_banks) == 3:

                if labels_for_gate is not None:

                    flat_labels = labels_for_gate.reshape(-1)  # [B*seq_len]
                    # audio out features for gating: pick appropriate representation - use x_unweighted_a summed over experts
                    # contract experts: produce per-frame representation by summing over experts (before gating)
                    rep_a = torch.sum(x_unweighted_a, dim=2)  # [B, seq_len, D_e]
                    rep_t = torch.sum(x_unweighted_t, dim=2)
                    rep_v = torch.sum(x_unweighted_v, dim=2)
                    rep_a_flat = rep_a.reshape(-1, self.D_e)
                    rep_t_flat = rep_t.reshape(-1, self.D_e)
                    rep_v_flat = rep_v.reshape(-1, self.D_e)
                    # fetch class prototypes for each modality
                    proto_a = self.prototype_banks['audio'].get()[flat_labels].to(rep_a_flat.device)  # [B*seq_len, D_e]
                    proto_t = self.prototype_banks['text'].get()[flat_labels].to(rep_t_flat.device)
                    proto_v = self.prototype_banks['vision'].get()[flat_labels].to(rep_v_flat.device)
                    # cosine similarity
                    sim_a = F.cosine_similarity(rep_a_flat, proto_a, dim=1).reshape(B, seq_len, 1)  # [B, seq_len, 1]
                    sim_t = F.cosine_similarity(rep_t_flat, proto_t, dim=1).reshape(B, seq_len, 1)
                    sim_v = F.cosine_similarity(rep_v_flat, proto_v, dim=1).reshape(B, seq_len, 1)
                else:
                    # no labels: compute max similarity across all prototypes
                    rep_a = torch.sum(x_unweighted_a, dim=2)  # [B, seq_len, D_e]
                    rep_t = torch.sum(x_unweighted_t, dim=2)
                    rep_v = torch.sum(x_unweighted_v, dim=2)
                    # [B*seq_len, D_e]
                    Ra = rep_a.reshape(-1, self.D_e)
                    Rt = rep_t.reshape(-1, self.D_e)
                    Rv = rep_v.reshape(-1, self.D_e)
                    # prototypes [K, D]
                    proto_a_all = self.prototype_banks['audio'].get().to(Ra.device)
                    proto_t_all = self.prototype_banks['text'].get().to(Rt.device)
                    proto_v_all = self.prototype_banks['vision'].get().to(Rv.device)
                    # compute cosine matrix and take max per row
                    sim_a_mat = torch.matmul(F.normalize(Ra, dim=1), F.normalize(proto_a_all, dim=1).t())  # [B*seq_len, K]
                    sim_t_mat = torch.matmul(F.normalize(Rt, dim=1), F.normalize(proto_t_all, dim=1).t())
                    sim_v_mat = torch.matmul(F.normalize(Rv, dim=1), F.normalize(proto_v_all, dim=1).t())
                    sim_a = torch.max(sim_a_mat, dim=1)[0].reshape(B, seq_len, 1)
                    sim_t = torch.max(sim_t_mat, dim=1)[0].reshape(B, seq_len, 1)
                    sim_v = torch.max(sim_v_mat, dim=1)[0].reshape(B, seq_len, 1)

                # map sim -> gate via small network -> expand to 3 experts
                gate_a = self.prototype_gate_net(sim_a)  # [B, seq_len, 1]
                gate_t = self.prototype_gate_net(sim_t)
                gate_v = self.prototype_gate_net(sim_v)
                # produce per-expert gates by copying same gate across the 3 experts
                proto_gate_a = gate_a.repeat(1, 1, 3)  # [B, seq_len, 3]
                proto_gate_t = gate_t.repeat(1, 1, 3)
                proto_gate_v = gate_v.repeat(1, 1, 3)

            # combine router weight and proto gate multiplicatively if proto gate exists, else use router alone
            if proto_gate_a is not None:
                w_a = weight_a * proto_gate_a  # [B, seq_len, 3]
                w_t = weight_t * proto_gate_t
                w_v = weight_v * proto_gate_v
                # renormalize across experts
                w_a = w_a / (w_a.sum(dim=-1, keepdim=True) + 1e-8)
                w_t = w_t / (w_t.sum(dim=-1, keepdim=True) + 1e-8)
                w_v = w_v / (w_v.sum(dim=-1, keepdim=True) + 1e-8)
            else:
                w_a, w_t, w_v = weight_a, weight_t, weight_v

            # expand to feature dim and weight experts
            w_a_exp = w_a.unsqueeze(-1).repeat(1, 1, 1, self.D_e)   # [B, seq_len, 3, D_e]
            w_t_exp = w_t.unsqueeze(-1).repeat(1, 1, 1, self.D_e)
            w_v_exp = w_v.unsqueeze(-1).repeat(1, 1, 1, self.D_e)

            x_out_a = torch.sum(w_a_exp * x_unweighted_a, dim=2)  # [B, seq_len, D_e]
            x_out_t = torch.sum(w_t_exp * x_unweighted_t, dim=2)
            x_out_v = torch.sum(w_v_exp * x_unweighted_v, dim=2)
            x = torch.cat([x_out_a, x_out_t, x_out_v], dim=1)

            out_a, out_t, out_v = None, None, None  # in 2nd stage these are not used for separate heads

        x[attn_mask == 0] = 0

        x_a, x_t, x_v = x[:, :seq_len, :], x[:, seq_len:2*seq_len, :], x[:, 2*seq_len:, :]
        x_joint = torch.cat([x_a, x_t, x_v], dim=-1)
        res = x_joint
        u = F.relu(self.proj1(x_joint))
        u = F.dropout(u, p=self.out_dropout, training=self.training)
        hidden = u + res
        out = self.nlp_head(hidden)

        # Optionally: update prototype banks with this batch's per-modality representations
        if (not first_stage) and update_prototypes and len(self.prototype_banks) == 3:
            # we use x_out_a/x_out_t/x_out_v (the gated fused per-modality features) as features to update prototypes
            # expected shapes [B, seq_len, D_e] -> flatten
            feats_a = x_out_a.reshape(-1, self.D_e)
            feats_t = x_out_t.reshape(-1, self.D_e)
            feats_v = x_out_v.reshape(-1, self.D_e)
            labels_flat = None
            if labels_for_gate is not None:
                labels_flat = labels_for_gate.reshape(-1)
            else:
                # if no label provided, we cannot update prototypes reliably here.
                labels_flat = None

            if labels_flat is not None:
                # respect umask (assume umask shape [B, seq_len])
                mask_flat = umask.reshape(-1).bool()
                if mask_flat.sum() > 0:
                    sel_feats_a = feats_a[mask_flat]
                    sel_feats_t = feats_t[mask_flat]
                    sel_feats_v = feats_v[mask_flat]
                    sel_labels = labels_flat[mask_flat].to(self.device)
                    # convert to device
                    sel_feats_a = sel_feats_a.to(self.device)
                    sel_feats_t = sel_feats_t.to(self.device)
                    sel_feats_v = sel_feats_v.to(self.device)
                    # momentum update
                    self.prototype_banks['audio'].momentum_update(sel_feats_a, sel_labels)
                    self.prototype_banks['text'].momentum_update(sel_feats_t, sel_labels)
                    self.prototype_banks['vision'].momentum_update(sel_feats_v, sel_labels)

       
        return hidden, out, out_a, out_t, out_v, np.array(weight_save)

    def compute_intra_loss(self, out_a: torch.Tensor, out_t: torch.Tensor, out_v: torch.Tensor,
                           labels: torch.Tensor, umask: torch.Tensor):
        """Compute intra-class prototype loss (MSE between sample features and class prototype).
        out_*: [B, seq_len, D_e], labels: [B, seq_len], umask: [B, seq_len]
        returns scalar tensor
        """
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        count = 0
        # flatten and gather only valid frames (umask)
        def gather(feat):
            B, S, D = feat.shape
            feat_flat = feat.reshape(-1, D)
            return feat_flat

        flat_a = gather(out_a)
        flat_t = gather(out_t)
        flat_v = gather(out_v)
        flat_labels = labels.reshape(-1)
        flat_mask = umask.reshape(-1).bool()
        if flat_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        sel_a = flat_a[flat_mask].to(self.device)
        sel_t = flat_t[flat_mask].to(self.device)
        sel_v = flat_v[flat_mask].to(self.device)
        sel_labels = flat_labels[flat_mask].to(self.device)
        # prototypes
        prot_a = self.prototype_banks['audio'].get()[sel_labels].to(self.device)
        prot_t = self.prototype_banks['text'].get()[sel_labels].to(self.device)
        prot_v = self.prototype_banks['vision'].get()[sel_labels].to(self.device)
        total = F.mse_loss(sel_a, prot_a) + F.mse_loss(sel_t, prot_t) + F.mse_loss(sel_v, prot_v)
        return total

    def compute_ot_align_loss_all_pairs(self, alpha: float = 0.1, beta: float = 0.05):
        """Sum pairwise L_inter across modalities using current prototypes."""
        total = torch.tensor(0.0, device=self.device)
        mods = ['audio', 'text', 'vision']
        for a, b in combinations(mods, 2):
            P = self.prototype_banks[a].get().to(self.device)
            Q = self.prototype_banks[b].get().to(self.device)
            out = self.aligner.align(P, Q, alpha=alpha, beta=beta)
            total = total + out['L_inter']
        return total

    def estimate_shapley_and_get_phi(self, n_permutations: int = 200, rho: float = 0.6):

        modalities = self.modalities

        def utility_fn(S:set):
            if len(S) == 0:
                return 0.0
            L_inter_S = 0.0
            L_intra_S = 0.0
            for a, b in combinations(list(S), 2):
                out = self.aligner.align(self.prototype_banks[a].get().to(self.device),
                                         self.prototype_banks[b].get().to(self.device))
                L_inter_S += float(out['L_inter'].item())
            for m in S:
                L_intra_S += float(self.prototype_banks[m].get().var().item())
            return rho / (1.0 + L_inter_S) + (1.0 - rho) / (1.0 + L_intra_S)

        sgm = ShapleySGM(modalities, utility_fn, n_permutations=n_permutations, device=self.device)
        shap_vals = sgm.estimate_shapley()
        phi = sgm.modulation_factors(shap_vals)
        return shap_vals, phi

    def get_param_groups_by_modality(self) -> Dict[str, List[nn.Parameter]]:

        groups = {'audio': [], 'text': [], 'vision': [], 'fusion': []}
        for name, param in self.named_parameters():
            n = name.lower()
            if ('a_in_proj' in n) or ('audio' in n and 'vision' not in n and 'text' not in n):
                groups['audio'].append(param)
            elif ('t_in_proj' in n) or ('text' in n) or ('bert' in n):
                groups['text'].append(param)
            elif ('v_in_proj' in n) or ('vision' in n) or ('cnn' in n):
                groups['vision'].append(param)
            else:
                # fusion / head / router / prototype net etc.
                groups['fusion'].append(param)
        return groups


    def to(self, device):
        super().to(device)
        self.device = torch.device(device)
        for pb in self.prototype_banks.values():
            pb.to(device)
        return self

