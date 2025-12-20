import numpy as np
import torch

def _normalize_adj(edge_index: np.ndarray, edge_weight: np.ndarray, n: int, device):
    """
    Compute D^-1/2 (A+I) D^-1/2 in sparse form.
    edge_index: (2,E)
    """
    # add self loops
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
    Build directed kNN graph by cosine similarity, then symmetrize with max weight.
    """
    x = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = x @ x.T
    np.fill_diagonal(sim, -1.0)

    k = int(min(k, sim.shape[1] - 1))
    idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]  # (N,k)

    rows = np.repeat(np.arange(sim.shape[0]), k)
    cols = idx.reshape(-1)
    w = sim[rows, cols].astype(np.float32)

    # symmetrize by max
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
    dp_pairs: np.ndarray,   # (E,2): [drug_local, prot_local]
    k_dd: int,
    k_pp: int,
    device
):
    """
    Return dict:
      dd: (nD,nD) sparse
      pp: (nP,nP) sparse
      dp: (nD+nP,nD+nP) sparse  (bipartite)
      ddpp: union sparse (nD+nP,nD+nP)
    """
    nD = drug_emb.shape[0]
    nP = prot_emb.shape[0]

    # DD / PP KNN graphs
    dd_ei, dd_ew = knn_cosine_graph(drug_emb, k_dd)
    pp_ei, pp_ew = knn_cosine_graph(prot_emb, k_pp)

    dd = _normalize_adj(dd_ei, dd_ew, nD, device)
    pp = _normalize_adj(pp_ei, pp_ew, nP, device)

    # DP bipartite on joint graph
    rows = dp_pairs[:, 0].astype(np.int64)
    cols = (nD + dp_pairs[:, 1]).astype(np.int64)

    e_r = np.concatenate([rows, cols], axis=0)
    e_c = np.concatenate([cols, rows], axis=0)
    e_w = np.ones_like(e_r, dtype=np.float32)

    dp_ei = np.stack([e_r, e_c], axis=0)
    dp = _normalize_adj(dp_ei, e_w, nD + nP, device)

    # ddpp union: dd + shifted pp + dp
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
