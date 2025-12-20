import numpy as np
import torch


def _normalize_adj(edge_index: np.ndarray, edge_weight: np.ndarray, n: int, device):
    """
    Chuẩn hoá ma trận kề theo kiểu GCN:
      A_hat = A + I
      D_hat = tổng trọng số theo hàng của A_hat
      Trả về: D_hat^(-1/2) * A_hat * D_hat^(-1/2) ở dạng sparse tensor.

    Tham số:
      edge_index: (2, E) gồm (row, col)
      edge_weight: (E,)
      n: số nút
    """
    self_loop = np.arange(n, dtype=np.int64)
    ei = np.concatenate([edge_index, np.stack([self_loop, self_loop], axis=0)], axis=1)
    ew = np.concatenate([edge_weight, np.ones(n, dtype=np.float32)], axis=0)

    row, col = ei[0], ei[1]
    deg = np.zeros(n, dtype=np.float32)
    np.add.at(deg, row, ew)

    deg_inv_sqrt = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    norm_w = ew * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    indices = torch.tensor(ei, dtype=torch.long, device=device)
    values = torch.tensor(norm_w, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


def knn_cosine_graph(emb: np.ndarray, k: int):
    """
    Tạo đồ thị kNN theo cosine similarity:
      - Chuẩn hoá embedding theo L2
      - Tính ma trận cosine sim
      - Với mỗi nút, lấy top-k láng giềng có sim cao nhất (bỏ self-loop)
      - Sau đó đối xứng hoá đồ thị bằng cách lấy trọng số lớn nhất cho mỗi cặp (u,v)
        rồi thêm hai chiều u->v và v->u.

    Trả về:
      edge_index: (2, E)
      edge_weight: (E,)
    """
    x = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -1.0)

    k = int(min(k, sim.shape[1] - 1))
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]

    rows = np.repeat(np.arange(sim.shape[0]), k)
    cols = idx.reshape(-1)
    w = sim[rows, cols].astype(np.float32)

    best = {}
    for r, c, ww in zip(rows, cols, w):
        if r == c:
            continue
        a, b = (r, c) if r < c else (c, r)
        best[(a, b)] = max(best.get((a, b), -1e9), float(ww))

    e_r, e_c, e_w = [], [], []
    for (a, b), ww in best.items():
        e_r += [a, b]
        e_c += [b, a]
        e_w += [ww, ww]

    edge_index = np.array([e_r, e_c], dtype=np.int64)
    edge_weight = np.array(e_w, dtype=np.float32)
    return edge_index, edge_weight


def build_midti_graphs(
    drug_emb: np.ndarray,
    prot_emb: np.ndarray,
    dp_pairs: np.ndarray,
    k_dd: int,
    k_pp: int,
    device
):
    """
    Dựng 4 đồ thị phục vụ nhánh MIDTI-like:

    1) dd: đồ thị drug-drug (kNN theo cosine) -> chuẩn hoá GCN
    2) pp: đồ thị protein-protein (kNN theo cosine) -> chuẩn hoá GCN
    3) dp: đồ thị hai phía drug-protein, dựng trên joint graph (nD+nP)
         với cạnh 2 chiều drug<->protein và weight=1, rồi chuẩn hoá GCN
    4) ddpp: đồ thị hợp nhất trên joint graph = dd + (pp dịch chỉ số) + dp

    dp_pairs: (E,2) với mỗi dòng là [drug_local_index, prot_local_index]
    """
    nD = drug_emb.shape[0]
    nP = prot_emb.shape[0]

    dd_ei, dd_ew = knn_cosine_graph(drug_emb, k_dd)
    pp_ei, pp_ew = knn_cosine_graph(prot_emb, k_pp)

    dd = _normalize_adj(dd_ei, dd_ew, nD, device)
    pp = _normalize_adj(pp_ei, pp_ew, nP, device)

    rows = dp_pairs[:, 0].astype(np.int64)
    cols = (nD + dp_pairs[:, 1]).astype(np.int64)

    e_r = np.concatenate([rows, cols], axis=0)
    e_c = np.concatenate([cols, rows], axis=0)
    e_w = np.ones_like(e_r, dtype=np.float32)

    dp_ei = np.stack([e_r, e_c], axis=0)
    dp = _normalize_adj(dp_ei, e_w, nD + nP, device)

    dd_idx = dd.indices().detach().cpu().numpy()
    dd_val = dd.values().detach().cpu().numpy()

    pp_idx = pp.indices().detach().cpu().numpy()
    pp_idx_shift = pp_idx.copy()
    pp_idx_shift[0] += nD
    pp_idx_shift[1] += nD
    pp_val = pp.values().detach().cpu().numpy()

    dp_idx = dp.indices().detach().cpu().numpy()
    dp_val = dp.values().detach().cpu().numpy()

    ddpp_idx = np.concatenate([dd_idx, pp_idx_shift, dp_idx], axis=1)
    ddpp_val = np.concatenate([dd_val, pp_val, dp_val], axis=0)

    ddpp = torch.sparse_coo_tensor(
        torch.tensor(ddpp_idx, dtype=torch.long, device=device),
        torch.tensor(ddpp_val, dtype=torch.float32, device=device),
        (nD + nP, nD + nP),
    ).coalesce()

    return {"dd": dd, "pp": pp, "dp": dp, "ddpp": ddpp}
