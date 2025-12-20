import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GCNStack


class MHAtt(nn.Module):
    """
    Multi-head attention (phiên bản gọn):
    - Nhận vào (V, K, Q) dạng (B, L, dim)
    - Tách thành nhiều head, tính attention, rồi ghép lại.
    """
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.dk = dim // n_heads

        self.v = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.q = nn.Linear(dim, dim)
        self.merge = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, v, k, q):
        B = q.size(0)

        v = self.v(v).view(B, -1, self.n_heads, self.dk).transpose(1, 2)
        k = self.k(k).view(B, -1, self.n_heads, self.dk).transpose(1, 2)
        q = self.q(q).view(B, -1, self.n_heads, self.dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        att = self.drop(F.softmax(scores, dim=-1))

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
        return self.merge(out)


class SA(nn.Module):
    """
    Self-attention + residual + LayerNorm.
    Dùng để cho drug tự nhìn lại các view của nó, protein cũng vậy.
    """
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.mh = MHAtt(dim, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.drop(self.mh(x, x, x)))


class CrossAtt(nn.Module):
    """
    Cross-attention + residual + LayerNorm.
    x "nhìn" y: Q lấy từ x, còn K,V lấy từ y.
    """
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.mh = MHAtt(dim, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, y):
        return self.norm(x + self.drop(self.mh(y, y, x)))


class DIA_Block(nn.Module):
    """
    Một block tương tác sâu:
    - Self-attention riêng cho drug
    - Self-attention riêng cho protein
    - Drug attend sang protein
    - Protein attend sang drug
    """
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.s_drug = SA(dim, n_heads, dropout)
        self.s_prot = SA(dim, n_heads, dropout)
        self.d2p = CrossAtt(dim, n_heads, dropout)
        self.p2d = CrossAtt(dim, n_heads, dropout)

    def forward(self, drug, prot):
        drug = self.s_drug(drug)
        prot = self.s_prot(prot)
        drug2 = self.d2p(drug, prot)
        prot2 = self.p2d(prot, drug)
        return drug2, prot2


class SimpleMIDTI(nn.Module):
    """
    Nhánh đồ thị kiểu "MIDTI-like":
    - Xây 4 đồ thị: dd, pp, dp, ddpp
    - Chạy GCN 3 tầng trên từng đồ thị
    - Gom nhiều “view” (9 view) cho drug và protein
    - Cho tương tác qua các DIA block
    - MLP ra 1 logit cho mỗi cặp (drug, target)
    """
    def __init__(self, n_drug, n_prot, dim=256, n_heads=8, dia_layers=2, dropout=0.1, mlp_hidden=128):
        super().__init__()
        self.nD = n_drug
        self.nP = n_prot
        self.dim = dim

        self.g_dd = GCNStack(dim)
        self.g_pp = GCNStack(dim)
        self.g_dp = GCNStack(dim)
        self.g_ddpp = GCNStack(dim)

        self.dia = nn.ModuleList([DIA_Block(dim, n_heads, dropout) for _ in range(dia_layers)])
        self.dia_layers = dia_layers

        self.dr_lin = nn.Linear((dia_layers + 1) * dim, dim)
        self.pr_lin = nn.Linear((dia_layers + 1) * dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.Tanh(),
            nn.Linear(dim, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, graphs, feat_drug, feat_prot, drug_idx, prot_idx):
        nD, nP = self.nD, self.nP
        feat_joint = torch.cat([feat_drug, feat_prot], dim=0)

        dd1, dd2, dd3 = self.g_dd(graphs["dd"], feat_drug)
        pp1, pp2, pp3 = self.g_pp(graphs["pp"], feat_prot)
        dp1, dp2, dp3 = self.g_dp(graphs["dp"], feat_joint)
        hp1, hp2, hp3 = self.g_ddpp(graphs["ddpp"], feat_joint)

        dpD = [dp1[:nD], dp2[:nD], dp3[:nD]]
        dpP = [dp1[nD:], dp2[nD:], dp3[nD:]]
        hpD = [hp1[:nD], hp2[:nD], hp3[:nD]]
        hpP = [hp1[nD:], hp2[nD:], hp3[nD:]]

        drug_stack = torch.stack(
            [dd1, dd2, dd3, dpD[0], dpD[1], dpD[2], hpD[0], hpD[1], hpD[2]],
            dim=1
        )
        prot_stack = torch.stack(
            [pp1, pp2, pp3, dpP[0], dpP[1], dpP[2], hpP[0], hpP[1], hpP[2]],
            dim=1
        )

        drugs = drug_stack[drug_idx]
        prots = prot_stack[prot_idx]

        d_cat = drugs
        p_cat = prots
        for blk in self.dia:
            d2, p2 = blk(drugs, prots)
            d_cat = torch.cat([d_cat, d2], dim=-1)
            p_cat = torch.cat([p_cat, p2], dim=-1)
            drugs, prots = d2, p2

        d_final = self.dr_lin(d_cat)
        p_final = self.pr_lin(p_cat)

        d_vec = d_final.mean(dim=1)
        p_vec = p_final.mean(dim=1)

        x = torch.cat([d_vec, p_vec], dim=1)
        logit = self.mlp(x).squeeze(-1)
        return logit
