import os
import time
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torch.optim.lr_scheduler as lr_scheduler

from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from loguru import logger

from DeepPurpose import utils as dp_utils

from .HDN import get_model
from .Utils import csv_record, check_dir, class_metrics, setup_seed, POS_LABEL, NEG_LABEL
from .midti import SimpleMIDTI, build_midti_graphs
from .models.uncertainty_gate import UncertaintyGatedFusion



# ---------- Local CSV loader ----------
def load_local_dataset(name: str):
    """
    Try:
      data/{name}.csv
      data/{name.lower()}.csv
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../MINDG
    cands = [
        os.path.join(base, "data", f"{name}.csv"),
        os.path.join(base, "data", f"{name.lower()}.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"Cannot find dataset csv in: {cands}")


def make_binary_labels(df: pd.DataFrame, name: str):
    """
    Convention (standard):
      POS_LABEL = 1, NEG_LABEL = 0
    If Label exists -> keep
    If Y exists -> binarize (DAVIS/BindingDB_Kd/KIBA rules)
    """
    df = df.copy()

    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(int)
        return df

    if "Y" not in df.columns:
        raise ValueError("Need 'Label' or 'Y' column in CSV.")

    y = pd.to_numeric(df["Y"], errors="coerce")
    df = df.dropna(subset=["Drug_ID", "Target_ID", "Drug", "Target", "Y"])

    if name.upper() == "DAVIS":
        pY = -np.log10(y.values.astype(float) * 1e-9 + 1e-12)
        df["pY"] = pY
        df["Label"] = np.where(df["pY"] >= 7.0, POS_LABEL, NEG_LABEL).astype(int)

    elif name == "BindingDB_Kd":
        pY = -np.log10(y.values.astype(float) * 1e-9 + 1e-12)
        df["pY"] = pY
        df["Label"] = np.where(df["pY"] >= 7.6, POS_LABEL, NEG_LABEL).astype(int)

    elif name.upper() == "KIBA":
        df["Label"] = np.where(y.values.astype(float) >= 12.1, POS_LABEL, NEG_LABEL).astype(int)

    else:
        raise ValueError(f"Unknown dataset rule for binarize: {name}")

    return df


def sample_stat(df: pd.DataFrame):
    neg_n = int((df["Label"] == NEG_LABEL).sum())
    pos_n = int((df["Label"] == POS_LABEL).sum())
    logger.info(f"neg/pos = {neg_n}/{pos_n} | neg%={100*neg_n/max(1,neg_n+pos_n):.2f}%")
    return neg_n, pos_n


def df_data_preprocess(df, undersampling=True):
    df = df.dropna().copy()
    df["Drug_ID"] = df["Drug_ID"].astype(str)
    neg_n, pos_n = sample_stat(df)

    if undersampling:
        neg_df = df[df["Label"] == NEG_LABEL].iloc[:pos_n]
        pos_df = df[df["Label"] == POS_LABEL]
        df = pd.concat([pos_df, neg_df], ignore_index=True)
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    sample_stat(df)
    return df


def df_data_split(df, frac=(0.7, 0.1, 0.2)):
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    total = len(df)
    t1 = int(total * frac[0])
    t2 = int(total * (frac[0] + frac[1]))
    train = df.iloc[:t1].copy()
    valid = df.iloc[t1:t2].copy()
    test  = df.iloc[t2:].copy()
    sample_stat(train); sample_stat(valid); sample_stat(test)
    return train, valid, test


def dti_df_process(df: pd.DataFrame):
    """
    Prepare DeepPurpose encoding columns:
      Seq_Drug, Seq_Target, Seq_Label + encoded drug_encoding, target_encoding
    """
    df2 = pd.DataFrame({
        "Seq_Drug": df["Drug"].values,
        "Seq_Target": df["Target"].values,
        "Seq_Label": df["Label"].values,
        "Graph_Drug": df["Drug_ID"].astype(str).values,
        "Graph_Target": df["Target_ID"].values
    })

    df2 = dp_utils.encode_drug(df2, "MPNN", column_name="Seq_Drug")
    df2 = dp_utils.encode_protein(df2, "CNN", column_name="Seq_Target")
    return df2


class DTI_Dataset_MIDTI(data.Dataset):
    """
    Returns:
      v_d, v_p, y, drug_local_idx, prot_local_idx
    """
    def __init__(self, df, drug_id2local, prot_id2local):
        self.df = df.reset_index(drop=True)
        self.drug_id2local = drug_id2local
        self.prot_id2local = prot_id2local

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        did = str(row.Graph_Drug)
        pid = row.Graph_Target

        d_idx = self.drug_id2local[did]
        p_idx = self.prot_id2local[pid]

        v_d = row.drug_encoding
        v_p = dp_utils.protein_2_embed(row.target_encoding)
        y = float(row.Seq_Label)
        return v_d, v_p, y, d_idx, p_idx


@torch.no_grad()
def calc_score(fusion_model, loader, device, graphs, feat_drug, feat_prot):
    fusion_model.eval()
    y_true, y_score = [], []
    for v_d, v_p, y, d_idx, p_idx in tqdm(loader, desc="metrics"):
        d_idx = torch.as_tensor(d_idx, dtype=torch.long, device=device)
        p_idx = torch.as_tensor(p_idx, dtype=torch.long, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)

        logit, _, _, _, _, _ = fusion_model(
            v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot,
            enable_mc=False
        )
        y_true.append(y.detach().cpu().numpy())
        y_score.append(logit.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)  # logits

    # logits -> probability
    prob = 1.0 / (1.0 + np.exp(-y_score))

    # threshold 0.5 on probability  <=> threshold 0 on logit
    y_pred = (prob >= 0.5).astype(int)  # positive=1, negative=0

    auroc = roc_auc_score(y_true, prob)
    auprc = average_precision_score(y_true, prob)

    m = class_metrics(y_true, y_pred)
    m["auroc"] = float(auroc)
    m["auprc"] = float(auprc)
    return m


def run(name="DAVIS", seed=10):
    setup_seed(seed)

    # ---- hyperparams ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    epochs = 100
    lr = 5e-4
    step_size = 10

    # ---- paths ----
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    out_root = os.path.join(base, "output", name, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    model_dir = os.path.join(base, "output", "models")
    check_dir(out_root)
    check_dir(model_dir)
    log_fd = logger.add(os.path.join(out_root, "train.log"))

    # ---- load df ----
    df_raw = load_local_dataset(name)
    df_raw = make_binary_labels(df_raw, name)
    df_raw = df_data_preprocess(df_raw, undersampling=True)

    train_df, valid_df, test_df = df_data_split(df_raw)
    train_df = dti_df_process(train_df)
    valid_df = dti_df_process(valid_df)
    test_df  = dti_df_process(test_df)

    # ---- id maps for MIDTI graph ----
    drug_ids = df_raw["Drug_ID"].astype(str).unique().tolist()
    prot_ids = df_raw["Target_ID"].unique().tolist()
    drug_id2local = {d:i for i,d in enumerate(drug_ids)}
    prot_id2local = {p:i for i,p in enumerate(prot_ids)}
    nD, nP = len(drug_ids), len(prot_ids)

    # dp edges from full df_raw (not split): (E,2)
    dp_pairs = np.stack([
        df_raw["Drug_ID"].astype(str).map(drug_id2local).values,
        df_raw["Target_ID"].map(prot_id2local).values,
    ], axis=1)

    # ---- dataloaders ----
    train_loader = data.DataLoader(
        DTI_Dataset_MIDTI(train_df, drug_id2local, prot_id2local),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=dp_utils.mpnn_collate_func
    )
    valid_loader = data.DataLoader(
        DTI_Dataset_MIDTI(valid_df, drug_id2local, prot_id2local),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=dp_utils.mpnn_collate_func
    )
    test_loader = data.DataLoader(
        DTI_Dataset_MIDTI(test_df, drug_id2local, prot_id2local),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=dp_utils.mpnn_collate_func
    )

    # ---- build models ----
    seq_model = get_model().model.to(device)

    dim = 256
    feat_drug = nn.Parameter(torch.randn(nD, dim, device=device) * 0.01)
    feat_prot = nn.Parameter(torch.randn(nP, dim, device=device) * 0.01)

    midti = SimpleMIDTI(nD, nP, dim=dim, n_heads=8, dia_layers=2, dropout=0.1, mlp_hidden=128).to(device)

    fusion = UncertaintyGatedFusion(
        student_seq=seq_model,
        teacher_midti=midti,
        mc_samples=6,
        temperature=2.0,
        gate_hidden=32
    ).to(device)

    # ---- optimizer ----
    optimizer = torch.optim.Adam(list(fusion.parameters()) + [feat_drug, feat_prot], lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    ckpt = os.path.join(model_dir, f"teacher_gated_{name}_e{epochs}.pt")
    best_auprc = -1.0
    best_ckpt = os.path.join(model_dir, f"teacher_gated_{name}_best.pt")
    patience = 20
    no_improve = 0

    def rebuild_graphs():
        return build_midti_graphs(
            drug_emb=feat_drug.detach().cpu().numpy(),
            prot_emb=feat_prot.detach().cpu().numpy(),
            dp_pairs=dp_pairs,
            k_dd=10,
            k_pp=10,
            device=device
        )

    graphs = rebuild_graphs()

    # ---- train ----
    logger.info("Start training Teacher-Gated + Distill (Seq student, MIDTI teacher)...")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        fusion.train()
        graphs = rebuild_graphs()  # refresh each epoch
        total_loss = 0.0

        y_score_train = []
        y_true_train = []

        for bi, (v_d, v_p, y, d_idx, p_idx) in enumerate(tqdm(train_loader, desc=f"epoch {ep}")):
            optimizer.zero_grad()

            d_idx = torch.as_tensor(d_idx, dtype=torch.long, device=device)
            p_idx = torch.as_tensor(p_idx, dtype=torch.long, device=device)
            y = torch.as_tensor(y, dtype=torch.float32, device=device)

            logit, logit_s, logit_t, w, u_s, u_t = fusion(
                v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot,
                enable_mc=True
            )

            # supervised on fused output
            L_sup = F.binary_cross_entropy_with_logits(logit.view(-1), y)

            # distill: pull student toward teacher
            L_kd = fusion.kd_loss(logit_s.view(-1), logit_t.view(-1))

            # gate reg to avoid collapse
            L_reg = ((w - 0.5) ** 2).mean()

            loss = L_sup + 0.1 * L_kd + 0.01 * L_reg
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            # track for train auroc
            y_true_train.append(y.detach().cpu().numpy())
            y_score_train.append(logit.detach().cpu().numpy())

            csv_record(os.path.join(out_root, "loss.csv"), {
                "epoch": ep,
                "batch": bi,
                "lr": optimizer.param_groups[0]["lr"],
                "loss": float(loss.item()),
                "avg_loss": total_loss / (bi + 1),
            })

        scheduler.step()

        y_true_train = np.concatenate(y_true_train)
        y_score_train = np.concatenate(y_score_train)
        auroc_train = roc_auc_score(y_true_train, y_score_train)

        val = calc_score(fusion, valid_loader, device, graphs, feat_drug, feat_prot)
        val["epoch"] = ep
        val["epoch_loss"] = total_loss / max(1, len(train_loader))
        csv_record(os.path.join(out_root, "val_metrics.csv"), val)

        logger.info(
            f"epoch {ep}/{epochs} | loss={val['epoch_loss']:.4f} | "
            f"auroc_train={auroc_train:.4f} | auroc_val={val['auroc']:.4f} | "
            f"auprc_val={val['auprc']:.4f} | f1_val={val['f1']:.4f}"
        )
        # ---- save best (by AUPRC) ----
        if val["auprc"] > best_auprc:
            best_auprc = val["auprc"]
            no_improve = 0
            torch.save({
                "fusion": fusion.state_dict(),
                "feat_drug": feat_drug.detach().cpu(),
                "feat_prot": feat_prot.detach().cpu(),
                "drug_ids": drug_ids,
                "prot_ids": prot_ids,
                "dim": dim,
                "epoch": ep,
                "auprc": best_auprc
            }, best_ckpt)
            logger.info(f"✔ New BEST model saved (epoch={ep}, auprc={best_auprc:.4f})")
        else:
            no_improve += 1
            logger.info(f"No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            logger.warning("Early stopping triggered!")
            break

    # ---- save ----
    torch.save({
        "fusion": fusion.state_dict(),
        "feat_drug": feat_drug.detach().cpu(),
        "feat_prot": feat_prot.detach().cpu(),
        "drug_ids": drug_ids,
        "prot_ids": prot_ids,
        "dim": dim
    }, ckpt)
    logger.info(f"Saved: {ckpt} | total_time={time.time()-t0:.2f}s")
    """
    payload = torch.load(best_ckpt, map_location=device)
    logger.info(f"Load BEST model from epoch {payload.get('epoch')} | auprc={payload.get('auprc')}")

    fusion.load_state_dict(payload["fusion"])
    feat_drug.data = payload["feat_drug"].to(device)
    feat_prot.data = payload["feat_prot"].to(device)
    graphs = rebuild_graphs()

    test = calc_score(fusion, test_loader, device, graphs, feat_drug, feat_prot)
    csv_record(os.path.join(out_root, "test_metrics.csv"), test)
    print("TEST:", test)"""

    logger.remove(log_fd)


if __name__ == "__main__":
    run("DAVIS")
