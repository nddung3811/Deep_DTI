import torch
import torch.nn as nn


def _mc_dropout_logits(forward_fn, n_samples: int = 6):
    """
    Chạy forward nhiều lần với dropout (MC Dropout) để ước lượng:
      - logit trung bình (mean)
      - độ bất định (variance) theo từng mẫu trong batch
    forward_fn phải trả về tensor shape (B,).
    """
    logits = []
    for _ in range(int(n_samples)):
        logits.append(forward_fn())
    s = torch.stack(logits, dim=0)  # (S, B)
    return s.mean(dim=0), s.var(dim=0, unbiased=False)


class PairGate(nn.Module):
    """
    Gate nhận vào bất định của student và teacher:
      (u_s, u_t) -> w thuộc (0,1)
    w gần 1: ưu tiên student
    w gần 0: ưu tiên teacher
    """
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, u_s: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        x = torch.stack([u_s, u_t], dim=-1)          # (B, 2)
        w = torch.sigmoid(self.net(x)).squeeze(-1)   # (B,)
        return w


class UncertaintyGatedFusion(nn.Module):
    """
    Kết hợp 2 mô hình:
      - student: mô hình theo chuỗi (DeepPurpose)
      - teacher: mô hình đồ thị (MIDTI-like)

    forward(...) trả về:
      logit_fused, logit_student, logit_teacher, w, u_s, u_t

    kd_loss(logit_s, logit_t): loss distillation cho bài toán nhị phân.
    """
    def __init__(
        self,
        student_seq: nn.Module,
        teacher_midti: nn.Module,
        mc_samples: int = 6,
        temperature: float = 2.0,
        gate_hidden: int = 32,
    ):
        super().__init__()
        self.student = student_seq
        self.teacher = teacher_midti
        self.mc_samples = int(mc_samples)
        self.T = float(temperature)
        self.gate = PairGate(hidden=int(gate_hidden))

    def forward(
        self,
        v_d,
        v_p,
        d_idx,
        p_idx,
        graphs,
        feat_drug,
        feat_prot,
        enable_mc: bool = True,
    ):
        def student_fn():
            out = self.student(v_d, v_p)   # (B,1) hoặc (B,)
            return out.view(-1)            # (B,)

        def teacher_fn():
            out = self.teacher(graphs, feat_drug, feat_prot, d_idx, p_idx)  # (B,) hoặc (B,1)
            return out.view(-1)                                            # (B,)

        if enable_mc and self.mc_samples > 1:
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

        w = self.gate(u_s, u_t)
        logit = w * logit_s + (1.0 - w) * logit_t
        return logit, logit_s, logit_t, w, u_s, u_t

    def kd_loss(self, logit_s: torch.Tensor, logit_t: torch.Tensor) -> torch.Tensor:
        """
        Distillation cho nhị phân:
        dùng KL(teacher || student) trên xác suất sigmoid đã làm mềm theo nhiệt độ T.
        """
        T = self.T
        ps = torch.sigmoid(logit_s / T)
        pt = torch.sigmoid(logit_t / T).detach()

        eps = 1e-7
        ps = ps.clamp(eps, 1 - eps)
        pt = pt.clamp(eps, 1 - eps)

        kl = pt * torch.log(pt / ps) + (1 - pt) * torch.log((1 - pt) / (1 - ps))
        return kl.mean() * (T * T)
