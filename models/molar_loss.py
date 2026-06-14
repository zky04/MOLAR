"""MOLAR loss for binary noisy-observation learning."""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn


def _as_task_tensor(x: torch.Tensor, num_tasks: int) -> torch.Tensor:
    if x.dim() == 1:
        return x.view(-1, num_tasks)
    return x.view(-1, num_tasks)


def clean_score(logits: torch.Tensor) -> torch.Tensor:
    return 2.0 * torch.sigmoid(logits) - 1.0


def bernoulli_js(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    p2 = torch.stack([p, 1.0 - p], dim=-1)
    q2 = torch.stack([q, 1.0 - q], dim=-1)
    m = 0.5 * (p2 + q2)
    return 0.5 * (p2 * (p2 / m).log()).sum(-1) + 0.5 * (q2 * (q2 / m).log()).sum(-1)


class MOLARLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.num_tasks = int(config.get("num_tasks", 1))
        self.method_strict = bool(config.get("method_strict", True))
        self.beta = float(config.get("beta_conflict", config.get("beta", 0.1)))
        self.beta_mod = float(config.get("beta_mod", 0.0))
        self.gamma = float(config.get("gamma_pert", config.get("gamma_temporal", config.get("gamma", 0.3))))
        self.evidence_l2 = float(config.get("evidence_l2", 0.0))
        self.unimodal_weight = float(config.get("unimodal_weight", 0.0))
        self.use_posterior_gating = bool(config.get("use_posterior_gating", True))
        self.consistency_target = config.get("consistency_target", config.get("temporal_consistency_target", "clean"))
        if self.consistency_target not in {"clean", "noisy"}:
            raise ValueError("consistency_target must be one of {'clean', 'noisy'}")
        self.supervised_target = config.get("supervised_target", "observed")
        if self.supervised_target not in {"observed", "clean", "teacher_smooth"}:
            raise ValueError("supervised_target must be one of {'observed', 'clean', 'teacher_smooth'}")
        self.min_reliability = float(config.get("min_reliability", config.get("rho", 0.2)))
        self.lambda_bootstrap = float(config.get("lambda_bootstrap", 1.0))
        self.reliability_mode = config.get("reliability_mode", "full")
        self.modality_loss_mode = config.get("modality_loss_mode", "triadic")
        valid_reliability = {"full", "observed_label", "teacher_only", "no_agreement"}
        valid_modality = {"triadic", "pairwise"}
        if self.reliability_mode not in valid_reliability:
            raise ValueError(
                f"Unknown reliability_mode={self.reliability_mode!r}; "
                f"expected one of {sorted(valid_reliability)}"
            )
        if self.modality_loss_mode not in valid_modality:
            raise ValueError(
                f"Unknown modality_loss_mode={self.modality_loss_mode!r}; "
                f"expected one of {sorted(valid_modality)}"
            )

        pos_weight = config.get("pos_weight")
        neg_weight = config.get("neg_weight")
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor(pos_weight, dtype=torch.float).view(1, -1))
        else:
            self.pos_weight = None
        if neg_weight is not None:
            self.register_buffer("neg_weight", torch.tensor(neg_weight, dtype=torch.float).view(1, -1))
        else:
            self.neg_weight = None
        self._validate_method_strict()

    def _weights_are_unity(self, weights) -> bool:
        if weights is None:
            return True
        tensor = torch.as_tensor(weights, dtype=torch.float)
        return bool(torch.allclose(tensor, torch.ones_like(tensor), atol=1e-7, rtol=0.0))

    def _validate_method_strict(self) -> None:
        if not self.method_strict:
            return
        violations = []
        if abs(self.beta_mod) > 1e-12:
            violations.append(f"beta_mod={self.beta_mod}")
        if abs(self.evidence_l2) > 1e-12:
            violations.append(f"evidence_l2={self.evidence_l2}")
        if abs(self.unimodal_weight) > 1e-12:
            violations.append(f"unimodal_weight={self.unimodal_weight}")
        if self.supervised_target != "observed":
            violations.append(f"supervised_target={self.supervised_target!r}")
        if not self._weights_are_unity(self.pos_weight):
            violations.append("pos_weight!=1")
        if not self._weights_are_unity(self.neg_weight):
            violations.append("neg_weight!=1")
        if violations:
            raise ValueError(
                "MOLARLoss strict methods mode implements exactly "
                "L_sup + beta*L_conf + gamma*L_pert with unweighted observed-label NLL. "
                "Disable method_strict only for legacy exploratory runs. Violations: "
                + ", ".join(violations)
            )

    def _denom(self, mask: torch.Tensor) -> torch.Tensor:
        return mask.sum().clamp_min(1.0)

    def _weighted_nll(self, p: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        p = p.clamp(1e-6, 1.0 - 1e-6)
        pos_w = self.pos_weight.to(p.device) if self.pos_weight is not None else 1.0
        neg_w = self.neg_weight.to(p.device) if self.neg_weight is not None else 1.0
        return -pos_w * labels * p.log() - neg_w * (1.0 - labels) * (1.0 - p).log()

    def _teacher_prob(self, outputs, teacher_outputs_w):
        if teacher_outputs_w is None:
            return outputs["p_clean"].detach()
        if "p_clean" in teacher_outputs_w:
            return teacher_outputs_w["p_clean"].detach()
        return teacher_outputs_w["p_f"].detach()

    def support_reliability(self, outputs, labels, mask, teacher_p, warmup=False):
        labels = _as_task_tensor(labels, self.num_tasks)
        mask = _as_task_tensor(mask, self.num_tasks)
        if warmup or self.reliability_mode == "observed_label":
            return torch.ones_like(labels) * mask, torch.ones_like(labels)

        teacher_p = teacher_p.detach()
        p_g = outputs["p_g"].detach()
        p_t = outputs["p_t"].detach()
        s_tch = labels * teacher_p + (1.0 - labels) * (1.0 - teacher_p)
        if self.reliability_mode == "teacher_only":
            support = s_tch
        else:
            s_g = labels * p_g + (1.0 - labels) * (1.0 - p_g)
            s_t = labels * p_t + (1.0 - labels) * (1.0 - p_t)
            support = (s_tch + s_g + s_t) / 3.0

        if self.reliability_mode in {"teacher_only", "no_agreement"}:
            agreement = torch.ones_like(labels)
        else:
            agreement = outputs.get("a_gt")
            if agreement is None:
                agreement = 1.0 - bernoulli_js(p_g, p_t) / math.log(2.0)
            agreement = agreement.detach().clamp(0, 1)

        w = self.min_reliability + (1.0 - self.min_reliability) * torch.sqrt(
            (support * agreement).clamp(0, 1)
        )
        return (w * mask).detach(), agreement

    def posterior_reliability(self, outputs, labels):
        labels = _as_task_tensor(labels, self.num_tasks)
        p = outputs["p_clean"].clamp(1e-6, 1.0 - 1e-6)
        pt = outputs["p_tilde"].clamp(1e-6, 1.0 - 1e-6)
        d_plus = outputs["d_plus"].clamp(1e-6, 1.0 - 1e-6)
        d_minus = outputs["d_minus"].clamp(1e-6, 1.0 - 1e-6)
        rel_pos = d_plus * p / pt
        rel_neg = d_minus * (1.0 - p) / (1.0 - pt)
        return (labels * rel_pos + (1.0 - labels) * rel_neg).clamp(0.0, 1.0)

    def supervised_loss(self, outputs, labels, mask, teacher_outputs_w=None, warmup=False):
        labels = _as_task_tensor(labels, self.num_tasks)
        mask = _as_task_tensor(mask, self.num_tasks)
        aux = {}
        if self.supervised_target == "teacher_smooth":
            teacher_p = self._teacher_prob(outputs, teacher_outputs_w)
            w, agreement = self.support_reliability(outputs, labels, mask, teacher_p, warmup=warmup)
            if abs(self.lambda_bootstrap - 1.0) < 1e-8:
                target = (w * labels + (1.0 - w) * teacher_p).detach()
                nll = self._weighted_nll(outputs["p_clean"], target)
                if self.unimodal_weight > 0.0:
                    nll = nll + self.unimodal_weight * (
                        self._weighted_nll(outputs["p_g"], target)
                        + self._weighted_nll(outputs["p_t"], target)
                    )
            else:
                nll = w * self._weighted_nll(outputs["p_clean"], labels) + self.lambda_bootstrap * (
                    1.0 - w
                ) * self._weighted_nll(outputs["p_clean"], teacher_p)
                if self.unimodal_weight > 0.0:
                    nll = nll + self.unimodal_weight * (
                        w * self._weighted_nll(outputs["p_g"], labels)
                        + self.lambda_bootstrap * (1.0 - w) * self._weighted_nll(outputs["p_g"], teacher_p)
                        + w * self._weighted_nll(outputs["p_t"], labels)
                        + self.lambda_bootstrap * (1.0 - w) * self._weighted_nll(outputs["p_t"], teacher_p)
                    )
            aux["w"] = w
            aux["agreement"] = agreement
        else:
            prob_key = "p_clean" if self.supervised_target == "clean" else "p_tilde"
            nll = self._weighted_nll(outputs[prob_key], labels)
            if self.unimodal_weight > 0.0:
                nll = nll + self.unimodal_weight * (
                    self._weighted_nll(outputs["p_g"], labels)
                    + self._weighted_nll(outputs["p_t"], labels)
                )
        return (mask * nll).sum() / self._denom(mask), aux

    def conflict_loss(self, outputs, labels, mask):
        mask = _as_task_tensor(mask, self.num_tasks)
        if self.use_posterior_gating:
            r = self.posterior_reliability(outputs, labels).detach()
        else:
            r = torch.ones_like(mask)
        s_g = clean_score(outputs["u_g"])
        s_t = clean_score(outputs["u_t"])
        conflict = torch.relu(-(s_g * s_t))
        return (mask * r * conflict).sum() / self._denom(mask), r, conflict

    def modality_agreement_loss(self, outputs, mask, w):
        mask = _as_task_tensor(mask, self.num_tasks)
        w = _as_task_tensor(w, self.num_tasks).detach()
        j_gt = bernoulli_js(outputs["p_g"], outputs["p_t"])
        if self.modality_loss_mode == "pairwise":
            loss = j_gt
        else:
            j_gf = bernoulli_js(outputs["p_g"], outputs["p_clean"])
            j_tf = bernoulli_js(outputs["p_t"], outputs["p_clean"])
            loss = j_gt + 0.5 * j_gf + 0.5 * j_tf
        return (mask * w * loss).sum() / self._denom(mask)

    def perturbation_loss(self, outputs_s, teacher_outputs_w, mask):
        mask = _as_task_tensor(mask, self.num_tasks)
        if self.consistency_target == "noisy":
            target = 2.0 * teacher_outputs_w["p_tilde"].detach() - 1.0
            pred = 2.0 * outputs_s["p_tilde"] - 1.0
        else:
            target = clean_score(teacher_outputs_w["logits_clean"]).detach()
            pred = clean_score(outputs_s["logits_clean"])
        loss = (pred - target).pow(2)
        return (mask * loss).sum() / self._denom(mask)

    def temporal_loss(self, outputs_s, teacher_outputs_w, mask):
        return self.perturbation_loss(outputs_s, teacher_outputs_w, mask)

    def evidence_l2_loss(self, outputs, mask):
        mask = _as_task_tensor(mask, self.num_tasks)
        loss = outputs["u_g"].pow(2) + outputs["u_t"].pow(2)
        return (mask * loss).sum() / self._denom(mask)

    def forward(
        self,
        outputs_w,
        labels,
        mask,
        outputs_s: Optional[Dict[str, torch.Tensor]] = None,
        teacher_outputs_w: Optional[Dict[str, torch.Tensor]] = None,
        warmup: bool = False,
        return_stats: bool = False,
    ):
        labels = _as_task_tensor(labels, self.num_tasks)
        mask = _as_task_tensor(mask, self.num_tasks)
        loss_sup, sup_aux = self.supervised_loss(
            outputs_w,
            labels,
            mask,
            teacher_outputs_w=teacher_outputs_w,
            warmup=warmup,
        )
        loss_conf_raw, reliability, conflict = self.conflict_loss(outputs_w, labels, mask)
        if self.beta_mod > 0.0:
            mod_w = sup_aux.get("w", mask)
            loss_mod_raw = self.modality_agreement_loss(outputs_w, mask, mod_w)
        else:
            loss_mod_raw = torch.zeros((), device=loss_sup.device)
        if outputs_s is not None and teacher_outputs_w is not None and not warmup:
            loss_pert_raw = self.perturbation_loss(outputs_s, teacher_outputs_w, mask)
        else:
            loss_pert_raw = torch.zeros((), device=loss_sup.device)
        loss_evidence_l2_raw = self.evidence_l2_loss(outputs_w, mask)

        loss = (
            loss_sup
            + self.beta * loss_conf_raw
            + self.beta_mod * loss_mod_raw
            + self.gamma * loss_pert_raw
            + self.evidence_l2 * loss_evidence_l2_raw
        )

        if return_stats:
            with torch.no_grad():
                valid = mask > 0
                mean = lambda x: x[valid].mean().item() if valid.any() else 0.0
                stats = {
                    "loss": loss.item(),
                    "loss_sup": loss_sup.item(),
                    "loss_conf": (self.beta * loss_conf_raw).item(),
                    "loss_mod": (self.beta_mod * loss_mod_raw).item(),
                    "loss_pert": (self.gamma * loss_pert_raw).item(),
                    "loss_temp": (self.gamma * loss_pert_raw).item(),
                    "loss_evidence_l2": (self.evidence_l2 * loss_evidence_l2_raw).item(),
                    "loss_conf_raw": loss_conf_raw.item(),
                    "loss_mod_raw": loss_mod_raw.item(),
                    "loss_pert_raw": loss_pert_raw.item(),
                    "loss_temp_raw": loss_pert_raw.item(),
                    "loss_evidence_l2_raw": loss_evidence_l2_raw.item(),
                    "mean_r": mean(reliability),
                    "mean_conflict": mean(conflict),
                    "mean_d_plus": mean(outputs_w["d_plus"]),
                    "mean_d_minus": mean(outputs_w["d_minus"]),
                    "mean_abs_u_g": mean(outputs_w["u_g"].abs()),
                    "mean_abs_u_t": mean(outputs_w["u_t"].abs()),
                    "mean_m_g": mean(outputs_w["m_g"]),
                    "mean_clean_conf": mean((outputs_w["p_clean"] - 0.5).abs() * 2.0),
                }
                if "w" in sup_aux:
                    stats["mean_w_supervised"] = mean(sup_aux["w"])
                    stats["mean_a_gt"] = mean(sup_aux["agreement"])
            return loss, stats
        return loss
