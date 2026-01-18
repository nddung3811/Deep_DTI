import streamlit as st
import torch
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from src.HDN import get_model
from src.midti import SimpleMIDTI, build_midti_graphs
from src.models.uncertainty_gate import UncertaintyGatedFusion
from src.Utils import setup_seed
from DeepPurpose import utils as dp_utils

# =============================
# CONFIG
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CKPT_PATH = r"C:\Users\dgx38\PycharmProjects\PythonProject1\output\models\teacher_gated_DAVIS_best.pt"
DATASET_CSV = r"data\DAVIS.csv"

DIM = 256
MC_SAMPLES = 3
MAX_DP_EDGES = 1500
K_DD = 5
K_PP = 5

# =============================
# PAGE
# =============================
st.set_page_config(page_title="DTI Demo", layout="centered")
st.title("🔬 Drug–Target Interaction Demo")
st.markdown("**Student–Teacher Model with Uncertainty-Gated Fusion**")

setup_seed(10)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    drug_ids = ckpt["drug_ids"]
    prot_ids = ckpt["prot_ids"]
    nD, nP = len(drug_ids), len(prot_ids)

    # Student (DeepPurpose)
    seq_model = get_model().model.to(DEVICE)

    # Teacher (MIDTI)
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

    drug_id2local = {d: i for i, d in enumerate(drug_ids)}
    prot_id2local = {p: i for i, p in enumerate(prot_ids)}

    return fusion, feat_drug, feat_prot, drug_ids, prot_ids, drug_id2local, prot_id2local


# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    return pd.read_csv(DATASET_CSV)


# =============================
# BUILD GRAPH (CACHED)
# =============================
@st.cache_resource
def build_graphs_cached(_feat_drug, _feat_prot, dp_pairs):
    return build_midti_graphs(
        drug_emb=_feat_drug.detach().cpu().numpy(),
        prot_emb=_feat_prot.detach().cpu().numpy(),
        dp_pairs=dp_pairs,
        k_dd=K_DD,
        k_pp=K_PP,
        device=DEVICE
    )


# =============================
# INIT
# =============================
fusion, feat_drug, feat_prot, drug_ids, prot_ids, drug_id2local, prot_id2local = load_model()
df = load_data()

st.success("✅ Model loaded. Ready for prediction.")

# =============================
# INPUT
# =============================
st.subheader("🧪 Input Drug & Protein")

smiles = st.text_input(
    "Drug SMILES",
    value="CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
)

fasta = st.text_area(
    "Protein FASTA",
    value="MSTGAVALAGLLLLPG..."
)

drug_id = st.selectbox("Reference Drug ID (for teacher graph)", drug_ids)
prot_id = st.selectbox("Reference Protein ID (for teacher graph)", prot_ids)

# =============================
# PREDICT
# =============================
if st.button("🔍 Predict Interaction"):

    # ---- Encode student inputs ----
    try:
        with st.spinner("🔬 Encoding SMILES & FASTA for Student..."):

            df_tmp = pd.DataFrame({
                "Drug": [smiles],
                "Target": [fasta]
            })

            df_tmp = dp_utils.encode_drug(
                df_tmp,
                drug_encoding="MPNN",
                column_name="Drug"
            )

            df_tmp = dp_utils.encode_protein(
                df_tmp,
                target_encoding="CNN",
                column_name="Target"
            )

            # ---- Drug: MPNN tuple (đã đúng shape) ----
            v_d = df_tmp.iloc[0].drug_encoding
            v_d = tuple(torch.tensor(x).to(DEVICE) for x in v_d)

            # ---- Protein: MUST embed ----
            v_p = dp_utils.protein_2_embed(df_tmp.iloc[0].target_encoding)
            v_p = torch.tensor(v_p).unsqueeze(0).to(DEVICE)

        student_ok = True
    except Exception as e:
        st.warning(f"⚠️ Student encoding failed: {e}")
        student_ok = False

    # ---- Build teacher graph ----
    with st.spinner("🔧 Building interaction graph (demo mode)..."):
        dp_df = df[["Drug_ID", "Target_ID"]].copy()
        dp_df["Drug_ID"] = dp_df["Drug_ID"].astype(str).map(drug_id2local)
        dp_df["Target_ID"] = dp_df["Target_ID"].map(prot_id2local)
        dp_df = dp_df.dropna()

        if len(dp_df) > MAX_DP_EDGES:
            dp_df = dp_df.sample(n=MAX_DP_EDGES, random_state=42)

        dp_pairs = dp_df.values.astype(np.int64)
        graphs = build_graphs_cached(feat_drug, feat_prot, dp_pairs)

    d_idx = torch.tensor([drug_id2local[drug_id]], device=DEVICE)
    p_idx = torch.tensor([prot_id2local[prot_id]], device=DEVICE)

    # ---- Forward ----
    with torch.no_grad():
        if student_ok:
            logit, logit_s, logit_t, w, u_s, u_t = fusion(
                v_d, v_p,
                d_idx, p_idx,
                graphs, feat_drug, feat_prot,
                enable_mc=True
            )
        else:
            logit_t = fusion.teacher(
                graphs, feat_drug, feat_prot, d_idx, p_idx
            )
            logit = logit_t
            logit_s = None
            w, u_s = None, None
            u_t = torch.tensor(0.0)

    # =============================
    # OUTPUT
    # =============================
    st.subheader("📊 Prediction Results")

    prob_f = torch.sigmoid(logit).item()
    st.metric("Final Prediction", f"{prob_f:.3f}")

    if student_ok:
        ps = torch.sigmoid(logit_s).item()
        pt = torch.sigmoid(logit_t).item()

        c1, c2, c3 = st.columns(3)
        c1.metric("Student", f"{ps:.3f}", f"Unc: {u_s.item():.4f}")
        c2.metric("Teacher", f"{pt:.3f}", f"Unc: {u_t.item():.4f}")
        c3.metric("Gate w", f"{w.item():.3f}")

        st.markdown("### 🧠 Explanation")
        if u_s.item() < u_t.item():
            st.info("Student has lower uncertainty → Fusion trusts **Student** more.")
        else:
            st.info("Teacher has lower uncertainty → Fusion trusts **Teacher** more.")
    else:
        st.info("Student branch disabled due to encoding error. Showing teacher-only result.")
