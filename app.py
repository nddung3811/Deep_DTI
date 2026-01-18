import streamlit as st
import torch
import numpy as np
import pandas as pd

from src.HDN import get_model
from src.midti import SimpleMIDTI, build_midti_graphs
from src.models.uncertainty_gate import UncertaintyGatedFusion
from src.Utils import setup_seed

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = r"C:\Users\dgx38\PycharmProjects\PythonProject1\output\models\teacher_gated_DaVIS_best.pt"
DATASET_CSV = "data/DAVIS.csv"   # đổi nếu cần

DIM = 256
MC_SAMPLES = 6

# -----------------------------
# LOAD MODEL (CACHE)
# -----------------------------
@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    drug_ids = ckpt["drug_ids"]
    prot_ids = ckpt["prot_ids"]
    nD, nP = len(drug_ids), len(prot_ids)

    # Student
    seq_model = get_model().model.to(DEVICE)

    # Teacher
    midti = SimpleMIDTI(
        n_drug=nD,
        n_prot=nP,
        dim=DIM,
        n_heads=8,
        dia_layers=2,
        dropout=0.1,
        mlp_hidden=128
    ).to(DEVICE)

    fusion = UncertaintyGatedFusion(
        student_seq=seq_model,
        teacher_midti=midti,
        mc_samples=MC_SAMPLES,
        temperature=2.0,
        gate_hidden=32
    ).to(DEVICE)

    fusion.load_state_dict(ckpt["fusion"])
    fusion.eval()

    feat_drug = ckpt["feat_drug"].to(DEVICE)
    feat_prot = ckpt["feat_prot"].to(DEVICE)

    return fusion, feat_drug, feat_prot, drug_ids, prot_ids


# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATASET_CSV)


# -----------------------------
# BUILD GRAPHS
# -----------------------------
def build_graphs(feat_drug, feat_prot, dp_pairs):
    return build_midti_graphs(
        drug_emb=feat_drug.detach().cpu().numpy(),
        prot_emb=feat_prot.detach().cpu().numpy(),
        dp_pairs=dp_pairs,
        k_dd=10,
        k_pp=10,
        device=DEVICE
    )


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="DTI Demo", layout="centered")
st.title("🔬 Drug–Target Interaction Demo")
st.markdown("**Uncertainty-Gated Student–Teacher Model**")

setup_seed(10)

fusion, feat_drug, feat_prot, drug_ids, prot_ids = load_model()
df = load_data()

# mapping
drug_id2local = {d: i for i, d in enumerate(drug_ids)}
prot_id2local = {p: i for i, p in enumerate(prot_ids)}

dp_df = df[["Drug_ID", "Target_ID"]].copy()

dp_df["Drug_ID"] = dp_df["Drug_ID"].astype(str).map(drug_id2local)
dp_df["Target_ID"] = dp_df["Target_ID"].map(prot_id2local)

# 🔥 LOẠI BỎ CÁC CẶP KHÔNG MAP ĐƯỢC
dp_df = dp_df.dropna()

dp_pairs = dp_df.values.astype(np.int64)


graphs = build_graphs(feat_drug, feat_prot, dp_pairs)

# -----------------------------
# INPUT
# -----------------------------
st.subheader("🧪 Select Drug – Protein Pair")

drug_id = st.selectbox("Drug ID", drug_ids)
prot_id = st.selectbox("Protein ID", prot_ids)

if st.button("🔍 Predict Interaction"):
    d_idx = torch.tensor([drug_id2local[drug_id]], device=DEVICE)
    p_idx = torch.tensor([prot_id2local[prot_id]], device=DEVICE)

    # dummy inputs cho student (không dùng trực tiếp)
    v_d = torch.zeros((1, 1), device=DEVICE)
    v_p = torch.zeros((1, 1), device=DEVICE)

    with torch.no_grad():
        logit, logit_s, logit_t, w, u_s, u_t = fusion(
            v_d, v_p, d_idx, p_idx,
            graphs, feat_drug, feat_prot,
            enable_mc=True
        )

    prob_s = torch.sigmoid(logit_s).item()
    prob_t = torch.sigmoid(logit_t).item()
    prob_f = torch.sigmoid(logit).item()

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Student",
        f"{prob_s:.3f}",
        f"Uncertainty: {u_s.item():.4f}"
    )

    col2.metric(
        "Teacher",
        f"{prob_t:.3f}",
        f"Uncertainty: {u_t.item():.4f}"
    )

    col3.metric(
        "Fusion (Final)",
        f"{prob_f:.3f}",
        f"Gate w: {w.item():.3f}"
    )

    st.markdown("### 🧠 Explanation")
    if u_s.item() < u_t.item():
        st.success("Student has lower uncertainty → Fusion trusts **Student** more.")
    else:
        st.success("Teacher has lower uncertainty → Fusion trusts **Teacher** more.")
