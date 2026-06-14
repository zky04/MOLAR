"""MOLAR natural-parameter noisy-observation model."""

from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool


def bernoulli_js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    p2 = torch.stack([p, 1.0 - p], dim=-1)
    q2 = torch.stack([q, 1.0 - q], dim=-1)
    m = 0.5 * (p2 + q2)
    return 0.5 * (p2 * (p2 / m).log()).sum(-1) + 0.5 * (q2 * (q2 / m).log()).sum(-1)


class EdgeGatedGINConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, bond_dim: int):
        super().__init__(aggr="add")
        self.node_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.edge_gate = nn.Sequential(nn.Linear(bond_dim, out_dim), nn.Sigmoid())
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_proj(x)
        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_gate[0].in_features))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.mlp((1.0 + self.eps) * x + out)

    def message(self, x_j, edge_attr):
        if edge_attr.size(-1) == 1 and self.edge_gate[0].in_features != 1:
            edge_attr = edge_attr.expand(-1, self.edge_gate[0].in_features)
        return x_j * self.edge_gate(edge_attr)


def _make_evidence_head(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    head = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1, bias=False),
    )
    nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
    return head


class MOLAR(nn.Module):
    """Binary natural-parameter evidence + endpoint noise channel."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.gnn_input_dim = config.get("gnn_input_dim", 36)
        self.bond_input_dim = config.get("bond_input_dim", 12)
        self.text_input_dim = config.get("text_input_dim", 768)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.embedding_dim = config.get("embedding_dim", 256)
        self.endpoint_emb_dim = config.get("endpoint_emb_dim", 64)
        self.num_tasks = config.get("num_tasks", 1)
        self.gnn_layers = config.get("gnn_layers", 4)
        self.dropout = config.get("dropout", 0.2)
        self.noise_init_diag = float(config.get("noise_init_diag", 0.95))
        self.evidence_dropout = float(config.get("evidence_dropout", 0.0))
        self.method_strict = bool(config.get("method_strict", True))
        self.clean_head_mode = config.get("clean_head_mode", "additive")
        if self.clean_head_mode not in {"additive", "gated", "concat"}:
            raise ValueError("clean_head_mode must be one of {'additive', 'gated', 'concat'}")
        if self.method_strict and self.clean_head_mode == "gated":
            raise ValueError("MOLAR strict methods mode supports additive or concat-ablation heads, not gated")
        self.noise_channel_mode = config.get("noise_channel_mode", "endpoint_specific")
        if self.noise_channel_mode not in {"endpoint_specific", "fixed_identity", "shared"}:
            raise ValueError(
                "noise_channel_mode must be one of "
                "{'endpoint_specific', 'fixed_identity', 'shared'}"
            )
        if self.method_strict and self.evidence_dropout != 0.0:
            raise ValueError("MOLAR strict methods mode requires evidence_dropout=0.0")
        self.disable_graph_evidence = bool(config.get("disable_graph_evidence", False))
        self.disable_text_evidence = bool(config.get("disable_text_evidence", False))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = self.gnn_input_dim
        for _ in range(self.gnn_layers):
            self.convs.append(EdgeGatedGINConv(in_dim, self.hidden_dim, self.bond_input_dim))
            self.norms.append(nn.BatchNorm1d(self.hidden_dim))
            in_dim = self.hidden_dim
        self.graph_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.embedding_dim),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(self.text_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(self.embedding_dim),
        )

        self.endpoint_emb = nn.Embedding(self.num_tasks, self.endpoint_emb_dim)
        evidence_in = self.embedding_dim + self.endpoint_emb_dim
        self.graph_evidence = _make_evidence_head(evidence_in, self.hidden_dim, self.dropout)
        self.text_evidence = _make_evidence_head(evidence_in, self.hidden_dim, self.dropout)
        if self.clean_head_mode == "gated":
            gate_in = self.embedding_dim * 4 + self.endpoint_emb_dim + 2
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_in, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, 1),
            )
            self.fusion_head = nn.Sequential(
                nn.Linear(self.embedding_dim + self.endpoint_emb_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )
        elif self.clean_head_mode == "concat":
            self.concat_fusion_head = nn.Sequential(
                nn.Linear(self.embedding_dim * 2 + self.endpoint_emb_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )

        self.endpoint_base_logit = nn.Parameter(torch.zeros(self.num_tasks))
        noise_param_tasks = 1 if self.noise_channel_mode == "shared" else self.num_tasks
        self.a_plus = nn.Parameter(torch.full((noise_param_tasks,), self._diag_to_param(self.noise_init_diag)))
        self.a_minus = nn.Parameter(torch.full((noise_param_tasks,), self._diag_to_param(self.noise_init_diag)))

        print(
            "MOLAR: "
            f"{self.gnn_layers}-layer edge-gated GIN + residual log-evidence heads + "
            f"endpoint noise channel, clean_head={self.clean_head_mode}, "
            f"noise_channel={self.noise_channel_mode}"
        )

    @staticmethod
    def _diag_to_param(diag: float) -> float:
        diag = min(max(diag, 0.5001), 0.9999)
        target = 2.0 * diag - 1.0
        return float(torch.logit(torch.tensor(target)).item())

    @torch.no_grad()
    def initialize_endpoint_base(self, observed_pos_rate):
        rate = torch.as_tensor(observed_pos_rate, dtype=self.endpoint_base_logit.dtype)
        rate = rate.view(-1).clamp(1e-4, 1.0 - 1e-4)
        if rate.numel() != self.num_tasks:
            raise ValueError(f"expected {self.num_tasks} base rates, got {rate.numel()}")
        self.endpoint_base_logit.copy_(torch.logit(rate).to(self.endpoint_base_logit.device))

    def noise_channel_parameters(self):
        return [self.a_plus, self.a_minus]

    def encode_graph(self, batch):
        x, edge_index = batch.x, batch.edge_index
        edge_attr = getattr(batch, "edge_attr", None)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.graph_proj(global_mean_pool(x, batch.batch))

    def encode_text(self, text_emb: Optional[torch.Tensor]):
        if text_emb is None:
            return None
        if text_emb.dim() == 1:
            text_emb = text_emb.view(-1, self.text_input_dim)
        return self.text_encoder(text_emb)

    def _evidence(self, z_g, z_t):
        B = z_g.size(0)
        K = self.num_tasks
        endpoint_ids = torch.arange(K, device=z_g.device)
        e = self.endpoint_emb(endpoint_ids)
        e_b = e.unsqueeze(0).expand(B, K, -1)
        z_g_b = z_g.unsqueeze(1).expand(B, K, -1)
        z_t_b = z_t.unsqueeze(1).expand(B, K, -1)
        u_g = self.graph_evidence(torch.cat([z_g_b, e_b], dim=-1).reshape(B * K, -1)).view(B, K)
        u_t = self.text_evidence(torch.cat([z_t_b, e_b], dim=-1).reshape(B * K, -1)).view(B, K)
        return u_g, u_t

    def _apply_evidence_dropout(self, u_g, u_t):
        p = self.evidence_dropout
        if not self.training or p <= 0.0:
            return u_g, u_t
        p = min(max(p, 0.0), 0.95)
        draw = torch.rand_like(u_g)
        drop_graph = draw < (0.5 * p)
        drop_text = (draw >= (0.5 * p)) & (draw < p)
        return u_g.masked_fill(drop_graph, 0.0), u_t.masked_fill(drop_text, 0.0)

    def forward(self, batch, text_emb=None):
        z_g = self.encode_graph(batch)
        z_t = self.encode_text(text_emb)
        if z_t is None:
            z_t = torch.zeros_like(z_g)

        u_g, u_t = self._evidence(z_g, z_t)
        u_g, u_t = self._apply_evidence_dropout(u_g, u_t)
        if self.disable_graph_evidence:
            u_g = torch.zeros_like(u_g)
        if self.disable_text_evidence:
            u_t = torch.zeros_like(u_t)
        base = self.endpoint_base_logit.view(1, -1)
        logits_g_clean = base + u_g
        logits_t_clean = base + u_t
        p_g = torch.sigmoid(logits_g_clean)
        p_t = torch.sigmoid(logits_t_clean)
        a_gt = (1.0 - bernoulli_js(p_g, p_t) / torch.log(torch.tensor(2.0, device=z_g.device))).clamp(0, 1)

        if self.clean_head_mode in {"gated", "concat"}:
            B = z_g.size(0)
            K = self.num_tasks
            endpoint_ids = torch.arange(K, device=z_g.device)
            e = self.endpoint_emb(endpoint_ids)
            e_b = e.unsqueeze(0).expand(B, K, -1)
            z_g_b = z_g.unsqueeze(1).expand(B, K, -1)
            z_t_b = z_t.unsqueeze(1).expand(B, K, -1)
        if self.clean_head_mode == "gated":
            q_g = torch.maximum(p_g, 1.0 - p_g)
            q_t = torch.maximum(p_t, 1.0 - p_t)
            gate_in = torch.cat(
                [
                    z_g_b,
                    z_t_b,
                    (z_g_b - z_t_b).abs(),
                    e_b,
                    z_g_b * z_t_b,
                    (q_g - q_t).detach().unsqueeze(-1),
                    a_gt.detach().unsqueeze(-1),
                ],
                dim=-1,
            )
            r_g = torch.sigmoid(self.gate_mlp(gate_in).squeeze(-1))
            z_r = r_g.unsqueeze(-1) * z_g_b + (1.0 - r_g).unsqueeze(-1) * z_t_b
            logits_clean = self.fusion_head(torch.cat([z_r, e_b], dim=-1).reshape(B * K, -1)).view(B, K)
        elif self.clean_head_mode == "concat":
            fusion_in = torch.cat([z_g_b, z_t_b, e_b], dim=-1)
            logits_clean = self.concat_fusion_head(fusion_in.reshape(B * K, -1)).view(B, K)
            r_g = None
        else:
            logits_clean = base + u_g + u_t
            r_g = None
        p_clean = torch.sigmoid(logits_clean)

        if self.noise_channel_mode == "fixed_identity":
            d_plus = torch.ones((1, self.num_tasks), device=z_g.device, dtype=p_clean.dtype)
            d_minus = torch.ones((1, self.num_tasks), device=z_g.device, dtype=p_clean.dtype)
        else:
            d_plus = 0.5 * (1.0 + torch.sigmoid(self.a_plus)).view(1, -1)
            d_minus = 0.5 * (1.0 + torch.sigmoid(self.a_minus)).view(1, -1)
            if self.noise_channel_mode == "shared":
                d_plus = d_plus.expand(1, self.num_tasks)
                d_minus = d_minus.expand(1, self.num_tasks)
        p_tilde = d_plus * p_clean + (1.0 - d_minus) * (1.0 - p_clean)
        p_tilde = p_tilde.clamp(1e-6, 1.0 - 1e-6)

        abs_g = u_g.abs()
        abs_t = u_t.abs()
        m_g = r_g if r_g is not None else abs_g / (abs_g + abs_t + 1e-6)
        s_g = 2.0 * torch.sigmoid(u_g) - 1.0
        s_t = 2.0 * torch.sigmoid(u_t) - 1.0
        conflict = F.relu(-(s_g * s_t))

        return {
            "logits_clean": logits_clean,
            "logits_g_clean": logits_g_clean,
            "logits_t_clean": logits_t_clean,
            "p_clean": p_clean,
            "p_f": p_clean,
            "p_g": p_g,
            "p_t": p_t,
            "p_tilde": p_tilde,
            "u_g": u_g,
            "u_t": u_t,
            "z_g": z_g,
            "z_t": z_t,
            "d_plus": d_plus.expand_as(p_clean),
            "d_minus": d_minus.expand_as(p_clean),
            "conflict": conflict,
            "m_g": m_g,
            "m_t": 1.0 - m_g,
            "a_gt": a_gt,
        }

    @torch.no_grad()
    def ema_update(self, teacher, momentum=0.995):
        for student_param, teacher_param in zip(self.parameters(), teacher.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for student_buffer, teacher_buffer in zip(self.buffers(), teacher.buffers()):
            if torch.is_floating_point(student_buffer):
                teacher_buffer.data.mul_(momentum).add_(student_buffer.data, alpha=1.0 - momentum)
            else:
                teacher_buffer.data.copy_(student_buffer.data)

    def create_teacher(self):
        teacher = deepcopy(self)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        return teacher
