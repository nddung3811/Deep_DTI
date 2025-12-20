# src/models/uncertainty_gate.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mc_dropout_logits(forward_fn, n_samples: int = 6):
    logits = []
    for _ in range(n_samples):
        logits.append(forward_fn())      # (B,)
    s = torch.stack(logits, dim=0)       # (S,B)
    return s.mean(dim=0), s.var(dim=0, unbiased=False)


class PairGate(nn.Module):
    """
    Gate depends on uncertainties (var) from student & teacher.
    Input: (u_s, u_t) -> w in (0,1)
    """
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, u_s, u_t):
        x = torch.stack([u_s, u_t], dim=-1)          # (B,2)
        w = torch.sigmoid(self.net(x)).squeeze(-1)   # (B,)
        return w


class UncertaintyGatedFusion(nn.Module):
    """
    student = DeepPurpose seq model
    teacher = MIDTI graph model

    forward(...) returns:
      logit_fused, logit_student, logit_teacher, w, u_s, u_t

    Also provides kd_loss(logit_s, logit_t) for distillation.
    """
    def __init__(self, student_seq, teacher_midti, mc_samples=6, temperature=2.0, gate_hidden=32):
        super().__init__()
        self.student = student_seq
        self.teacher = teacher_midti
        self.mc_samples = int(mc_samples)
        self.T = float(temperature)
        self.gate = PairGate(hidden=int(gate_hidden))

    def forward(self, v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot, enable_mc=True):
        # ---- student logit ----
        def student_fn():
            out = self.student(v_d, v_p)      # (B,1) or (B,)
            return out.view(-1)

        # ---- teacher logit ----
        def teacher_fn():
            out = self.teacher(graphs, feat_drug, feat_prot, d_idx, p_idx)  # (B,)
            return out.view(-1)

        if enable_mc and self.mc_samples > 1:
            # enable dropout sampling
            self.student.train()
            self.teacher.train()
            logit_s, var_s = _mc_dropout_logits(student_fn, self.mc_samples)
            logit_t, var_t = _mc_dropout_logits(teacher_fn, self.mc_samples)
            u_s = var_s.detach()
            u_t = var_t.detach()
        else:
            logit_s = student_fn()
            logit_t = teacher_fn()
            u_s = torch.zeros_like(logit_s)
            u_t = torch.zeros_like(logit_t)

        w = self.gate(u_s, u_t)  # (B,)
        logit = w * logit_s + (1.0 - w) * logit_t

        return logit, logit_s, logit_t, w, u_s, u_t

    def kd_loss(self, logit_s, logit_t):
        """
        Distillation for binary:
          KL(teacher || student) on softened sigmoid probs.
        """
        T = self.T
        ps = torch.sigmoid(logit_s / T)
        pt = torch.sigmoid(logit_t / T).detach()

        eps = 1e-7
        ps = ps.clamp(eps, 1 - eps)
        pt = pt.clamp(eps, 1 - eps)

        kl = pt * torch.log(pt / ps) + (1 - pt) * torch.log((1 - pt) / (1 - ps))
        return kl.mean() * (T * T)
