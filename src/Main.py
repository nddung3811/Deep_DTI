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

def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x

#  ĐỌC DATASET TỪ FILE CSV
def load_local_dataset(name: str):
    """
    Thử đọc file theo 2 tên:
      - data/{name}.csv
      - data/{name.lower()}.csv

    Trả về: pandas.DataFrame
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [
        os.path.join(base, "data", f"{name}.csv"),
        os.path.join(base, "data", f"{name.lower()}.csv"),
    ]
    for p in cands:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Không tìm thấy file CSV trong thư mục data/ (thử tên hoa/thường).")


#  TẠO NHÃN NHỊ PHÂN (0/1)
def make_binary_labels(df: pd.DataFrame, name: str):
    df = df.copy()

    if "Label" in df.columns:
        df["Label"] = df["Label"].astype(int)
        return df

    if "Y" not in df.columns:
        raise ValueError("CSV cần có cột 'Label' hoặc 'Y'.")

    #  DROP NA TRƯỚC
    df = df.dropna(subset=["Drug_ID", "Target_ID", "Drug", "Target", "Y"])

    #  LẤY y SAU KHI DROP
    y = pd.to_numeric(df["Y"], errors="coerce").values.astype(float)

    if name.upper() == "DAVIS":
        pY = -np.log10(y * 1e-9 + 1e-12)
        df["pY"] = pY
        df["Label"] = (df["pY"] >= 7.0).astype(int)

    elif name == "BindingDB_Kd":
        pY = -np.log10(y * 1e-9 + 1e-12)
        df["pY"] = pY
        df["Label"] = (df["pY"] >= 7.6).astype(int)

    elif name.upper() == "KIBA":
        df["Label"] = (y >= 12.1).astype(int)

    else:
        raise ValueError(f"Dataset chưa hỗ trợ rule binarize: {name}")

    return df



def sample_stat(df: pd.DataFrame):
    """In thống kê số lượng âm/dương (NEG/POS) để kiểm tra mất cân bằng."""
    neg_n = int((df["Label"] == NEG_LABEL).sum())
    pos_n = int((df["Label"] == POS_LABEL).sum())
    logger.info(f"neg/pos = {neg_n}/{pos_n} | neg%={100*neg_n/max(1,neg_n+pos_n):.2f}%")
    return neg_n, pos_n


#  TIỀN XỬ LÝ: CÂN BẰNG (UNDERSAMPLE)
def df_data_preprocess(df, undersampling=True):
    """
    - Drop NaN
    - Ép Drug_ID về string
    - (tuỳ chọn) Undersampling: cắt bớt lớp âm để cân bằng với lớp dương
    """
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


#  CHIA TRAIN/VALID/TEST
def df_data_split(df, frac=(0.7, 0.1, 0.2)):
    """
    Chia dữ liệu theo tỉ lệ (train, valid, test).
    Mặc định: 70% / 10% / 20%
    """
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    total = len(df)
    t1 = int(total * frac[0])
    t2 = int(total * (frac[0] + frac[1]))

    train = df.iloc[:t1].copy()
    valid = df.iloc[t1:t2].copy()
    test  = df.iloc[t2:].copy()

    sample_stat(train)
    sample_stat(valid)
    sample_stat(test)
    return train, valid, test


#  CHUẨN BỊ CỘT ENCODING CHO DeepPurpose
def dti_df_process(df: pd.DataFrame):
    """
    Chuẩn hoá dataframe để DeepPurpose encode:
      - Seq_Drug, Seq_Target, Seq_Label
      - Graph_Drug, Graph_Target (để map sang index cho MIDTI)

    Sau đó:
      - encode_drug (MPNN)
      - encode_protein (CNN)
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


#  DATASET WRAPPER: trả cả index drug/protein
class DTI_Dataset_MIDTI(data.Dataset):
    """
    Trả về:
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


#  ĐÁNH GIÁ (VALID/TEST)
@torch.no_grad()
def calc_score(fusion_model, loader, device, graphs, feat_drug, feat_prot):
    """
    Tính metrics:
      - AUROC, AUPRC (tính trên probability)
      - các chỉ số phân loại khác từ class_metrics (threshold 0.5)
    """
    fusion_model.eval()
    y_true, y_score = [], []

    for v_d, v_p, y, d_idx, p_idx in tqdm(loader, desc="metrics"):
        v_d = to_device(v_d, device)
        v_p = to_device(v_p, device)

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

    # dự đoán nhị phân theo ngưỡng 0.5
    y_pred = (prob >= 0.5).astype(int)

    auroc = roc_auc_score(y_true, prob)
    auprc = average_precision_score(y_true, prob)

    m = class_metrics(y_true, y_pred)
    m["auroc"] = float(auroc)
    m["auprc"] = float(auprc)
    return m
@torch.no_grad()
def export_test_csv(
    fusion_model,
    loader,
    device,
    graphs,
    feat_drug,
    feat_prot,
    save_path
):
    fusion_model.eval()

    rows = []

    for v_d, v_p, y, d_idx, p_idx in tqdm(loader, desc="export test csv"):
        v_d = to_device(v_d, device)
        v_p = to_device(v_p, device)
        d_idx = torch.as_tensor(d_idx, dtype=torch.long, device=device)
        p_idx = torch.as_tensor(p_idx, dtype=torch.long, device=device)
        y = torch.as_tensor(y, dtype=torch.float32, device=device)

        logit, logit_s, logit_t, w, u_s, u_t = fusion_model(
            v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot,
            enable_mc=False
        )

        prob_f = torch.sigmoid(logit)
        prob_s = torch.sigmoid(logit_s)
        prob_t = torch.sigmoid(logit_t)

        for i in range(len(y)):
            rows.append({
                "Drug_ID": int(d_idx[i].cpu()),
                "Target_ID": int(p_idx[i].cpu()),
                "Label": int(y[i].cpu()),
                "Student": float(prob_s[i].cpu()),
                "Teacher": float(prob_t[i].cpu()),
                "Fusion": float(prob_f[i].cpu()),
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✔ Test CSV saved to {save_path}")


#  HÀM CHẠY CHÍNH
def run(name="DAVIS", seed=10):
    """
    Pipeline tổng:
      1) Set seed
      2) Load & binarize dataset
      3) Preprocess + split train/valid/test
      4) Encode DeepPurpose inputs
      5) Xây ID map cho MIDTI
      6) Tạo dp_pairs CHỈ từ TRAIN để tránh leakage
      7) Build model: Student (seq) + Teacher (MIDTI) + Uncertainty Gate
      8) Train + validate + early stopping + lưu best theo AUPRC
    """
    setup_seed(seed)

    # ---- siêu tham số ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device = {device}")
    if device.type == "cuda":
        logger.info(f"GPU = {torch.cuda.get_device_name(0)}")
    batch_size = 32
    epochs = 100
    lr = 5e-4
    step_size = 10

    # ---- thư mục output/log/model ----
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
    out_root = os.path.join(base, "output", name, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    model_dir = os.path.join(base, "output", "models")
    check_dir(out_root)
    check_dir(model_dir)
    log_fd = logger.add(os.path.join(out_root, "train.log"))

    # ---- load dataframe ----
    df_raw = load_local_dataset(name)
    df_raw = make_binary_labels(df_raw, name)
    df_raw = df_data_preprocess(df_raw, undersampling=True)

    # ---- split & encode ----
    train_df, valid_df, test_df = df_data_split(df_raw)
    train_df = dti_df_process(train_df)
    valid_df = dti_df_process(valid_df)
    test_df  = dti_df_process(test_df)

    # ---- map ID -> local index cho MIDTI ----
    drug_ids = df_raw["Drug_ID"].astype(str).unique().tolist()
    prot_ids = df_raw["Target_ID"].unique().tolist()
    drug_id2local = {d: i for i, d in enumerate(drug_ids)}
    prot_id2local = {p: i for i, p in enumerate(prot_ids)}
    nD, nP = len(drug_ids), len(prot_ids)

    # ---- dp edges: CHỈ LẤY TỪ TRAIN để tránh leakage ----
    dp_pairs = np.stack([
        train_df["Graph_Drug"].astype(str).map(drug_id2local).values,
        train_df["Graph_Target"].map(prot_id2local).values,
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
    # student: DeepPurpose (seq)
    seq_model = get_model().model.to(device)

    # embedding learnable cho drug/protein (đưa vào build graph MIDTI)
    dim = 256
    feat_drug = nn.Parameter(torch.randn(nD, dim, device=device) * 0.01)
    feat_prot = nn.Parameter(torch.randn(nP, dim, device=device) * 0.01)

    # teacher: SimpleMIDTI (graph)
    midti = SimpleMIDTI(
        nD, nP, dim=dim, n_heads=8, dia_layers=2,
        dropout=0.1, mlp_hidden=128
    ).to(device)

    # fusion: uncertainty-gated + distillation
    fusion = UncertaintyGatedFusion(
        student_seq=seq_model,
        teacher_midti=midti,
        mc_samples=6,
        temperature=2.0,
        gate_hidden=32
    ).to(device)

    # ---- optimizer + scheduler ----
    optimizer = torch.optim.Adam(list(fusion.parameters()) + [feat_drug, feat_prot], lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    ckpt = os.path.join(model_dir, f"teacher_gated_{name}_e{epochs}.pt")
    best_auprc = -1.0
    best_ckpt = os.path.join(model_dir, f"teacher_gated_{name}_best.pt")
    patience = 10
    no_improve = 0

    def rebuild_graphs():
        """
        Mỗi epoch rebuild lại graph dựa trên embedding hiện tại (feat_drug/feat_prot).
        dp_pairs đã cố định từ TRAIN để tránh leakage.
        """
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
    logger.info("Bắt đầu train Teacher-Gated + Distill (student=seq, teacher=MIDTI)...")
    t0 = time.time()

    for ep in range(1, epochs + 1):
        fusion.train()
        graphs = rebuild_graphs()  # cập nhật graph mỗi epoch
        total_loss = 0.0

        y_score_train = []
        y_true_train = []

        for bi, (v_d, v_p, y, d_idx, p_idx) in enumerate(tqdm(train_loader, desc=f"epoch {ep}")):
            optimizer.zero_grad()

            v_d = to_device(v_d, device)
            v_p = to_device(v_p, device)

            if ep == 1 and bi == 0:
                def first_tensor(x):
                    if torch.is_tensor(x): return x
                    if isinstance(x, (list, tuple)):
                        for t in x:
                            r = first_tensor(t)
                            if r is not None: return r
                    if isinstance(x, dict):
                        for t in x.values():
                            r = first_tensor(t)
                            if r is not None: return r
                    return None

                td = first_tensor(v_d)
                tp = first_tensor(v_p)
                logger.info(f"first tensor v_d device: {td.device if td is not None else 'NONE'}")
                logger.info(f"first tensor v_p device: {tp.device if tp is not None else 'NONE'}")
                logger.info(f"fusion device: {next(fusion.parameters()).device}")

            d_idx = torch.as_tensor(d_idx, dtype=torch.long, device=device)
            p_idx = torch.as_tensor(p_idx, dtype=torch.long, device=device)
            y = torch.as_tensor(y, dtype=torch.float32, device=device)

            logit, logit_s, logit_t, w, u_s, u_t = fusion(
                v_d, v_p, d_idx, p_idx, graphs, feat_drug, feat_prot,
                enable_mc=True
            )

            # loss supervised (fused)
            L_sup = F.binary_cross_entropy_with_logits(logit.view(-1), y)

            # loss distillation (kéo student học theo teacher)
            L_kd = fusion.kd_loss(logit_s.view(-1), logit_t.view(-1))

            # regularize gate để tránh collapse
            L_reg = ((w - 0.5) ** 2).mean()

            loss = L_sup + 0.1 * L_kd + 0.01 * L_reg
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

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

        # ---- lưu best theo AUPRC ----
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
            logger.info(f"✔ Lưu BEST model (epoch={ep}, auprc={best_auprc:.4f})")
        else:
            no_improve += 1
            logger.info(f"Không cải thiện ({no_improve}/{patience})")

        if no_improve >= patience:
            logger.warning("Kích hoạt early stopping!")
            break

    # ---- lưu checkpoint cuối ----
    torch.save({
        "fusion": fusion.state_dict(),
        "feat_drug": feat_drug.detach().cpu(),
        "feat_prot": feat_prot.detach().cpu(),
        "drug_ids": drug_ids,
        "prot_ids": prot_ids,
        "dim": dim
    }, ckpt)
    logger.info(f"Đã lưu: {ckpt} | total_time={time.time()-t0:.2f}s")
    # ---- load best model ----
    payload = torch.load(best_ckpt, map_location=device)
    fusion.load_state_dict(payload["fusion"])
    feat_drug.data = payload["feat_drug"].to(device)
    feat_prot.data = payload["feat_prot"].to(device)
    graphs = rebuild_graphs()

    # ---- export test csv ----
    export_test_csv(
        fusion_model=fusion,
        loader=test_loader,
        device=device,
        graphs=graphs,
        feat_drug=feat_drug,
        feat_prot=feat_prot,
        save_path=os.path.join(out_root, "test_predictions.csv")
    )
        

    logger.remove(log_fd)


if __name__ == "__main__":
    
    run("KIBA")

    
    