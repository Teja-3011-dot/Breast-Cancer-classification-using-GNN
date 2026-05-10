import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import random

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data

# ── Uncomment once your model file is in place ──────────────────────────────
# from models.gnn_model import GCN
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================
# THEME / GLOBAL STYLE
# ============================================================

st.set_page_config(
    page_title="Breast Cancer · GNN Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT   = "#E84393"   # vibrant pink – medical / cancer-awareness palette
ACCENT2  = "#7B2FBE"   # deep violet
BG_CARD  = "#0F0F1A"
BG_PLOT  = "#0A0A14"
TEXT_MUT = "#8888AA"

st.markdown(f"""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root reset ── */
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: #070710;
    color: #E8E8F0;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {BG_CARD};
    border-right: 1px solid #1E1E32;
}}
section[data-testid="stSidebar"] * {{ color: #C8C8DC !important; }}

/* ── Hero banner ── */
.hero {{
    background: linear-gradient(135deg, #12012A 0%, #1A0535 50%, #0D0820 100%);
    border: 1px solid #2A1050;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}}
.hero::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, {ACCENT}44 0%, transparent 70%);
    pointer-events: none;
}}
.hero h1 {{
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
}}
.hero p {{
    color: {TEXT_MUT};
    font-size: 1rem;
    margin: 0;
}}

/* ── Section headers ── */
.section-title {{
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {ACCENT};
    margin: 2rem 0 0.8rem;
    border-left: 3px solid {ACCENT};
    padding-left: 0.7rem;
}}

/* ── Cards ── */
.card {{
    background: {BG_CARD};
    border: 1px solid #1E1E32;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}}

/* ── Prediction result boxes ── */
.pred-benign {{
    background: linear-gradient(135deg, #0A2A1A, #0D3520);
    border: 2px solid #22C55E;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
}}
.pred-malignant {{
    background: linear-gradient(135deg, #2A0A0A, #350D0D);
    border: 2px solid {ACCENT};
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
}}
.pred-label {{
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}}
.pred-sub {{
    font-size: 0.85rem;
    color: {TEXT_MUT};
    margin-top: 0.3rem;
}}

/* ── Number inputs ── */
input[type=number] {{
    background: #12122A !important;
    border: 1px solid #2A2A4A !important;
    border-radius: 8px !important;
    color: #E8E8F0 !important;
}}

/* ── Buttons ── */
div.stButton > button {{
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
    letter-spacing: 0.06em;
    padding: 0.65rem 2.2rem;
    width: 100%;
    transition: opacity 0.2s;
}}
div.stButton > button:hover {{ opacity: 0.85; }}

/* ── Metric chips ── */
.metric-chip {{
    display: inline-block;
    background: #16163A;
    border: 1px solid #2A2A5A;
    border-radius: 999px;
    padding: 0.25rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: {ACCENT};
    margin-right: 0.4rem;
}}
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA LOADING  (cached so it only runs once)
# ============================================================

@st.cache_data
def load_data():
    dataset = load_breast_cancer()
    X, y    = dataset.data, dataset.target

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float)

    A = kneighbors_graph(X_scaled, n_neighbors=5,
                         mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(A.nonzero()), dtype=torch.long)

    data_graph = Data(x=X_tensor, edge_index=edge_index)
    return dataset, X, y, X_scaled, scaler, data_graph, edge_index


dataset, X, y, X_scaled, scaler, data_graph, edge_index = load_data()
feature_names = dataset.feature_names


# ============================================================
# MODEL  (replace stub with real GCN once available)
# ============================================================

@st.cache_resource
def load_model():
    # ── Stub so the app runs without the model file ──────────────────────────
    class _GCNStub(torch.nn.Module):
        def forward(self, x, edge_index):
            # Returns random logits; swap this out for real GCN
            return torch.randn(x.size(0), 2)
    return _GCNStub().eval()

    # ── Real model (uncomment when ready) ────────────────────────────────────
    # model = GCN(input_dim=X.shape[1], hidden_dim=16, output_dim=2)
    # model.eval()
    # return model

model = load_model()


# ============================================================
# SIDEBAR  –  dataset info + controls
# ============================================================

with st.sidebar:
    st.markdown("### 🧬 Dataset")
    st.markdown(f"""
    <div class='card'>
      <span class='metric-chip'>{X.shape[0]} samples</span>
      <span class='metric-chip'>{X.shape[1]} features</span>
      <br><br>
      <small style='color:{TEXT_MUT}'>
        Breast Cancer Wisconsin (Diagnostic)<br>
        357 benign · 212 malignant
      </small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Graph Config")
    k_neighbors = st.slider("k-NN neighbors", 3, 15, 5)
    sample_size = st.slider("Visualisation sample size", 20, 150, 50)

    st.markdown("### 🏗️ Model")
    st.markdown(f"""
    <div class='card'>
      <small style='color:{TEXT_MUT}'>
        Architecture: <b style='color:#E8E8F0'>GCN</b><br>
        Hidden dim: <b style='color:#E8E8F0'>16</b><br>
        Output classes: <b style='color:#E8E8F0'>2</b>
      </small>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HERO BANNER
# ============================================================

st.markdown("""
<div class='hero'>
  <h1>Breast Cancer · GNN Classifier</h1>
  <p>Graph Neural Network analysis of the Wisconsin Diagnostic dataset —
     enter patient features below to receive a classification.</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# VISUALISATIONS
# ============================================================

col_viz1, col_viz2 = st.columns(2)

# ── Feature Distribution ────────────────────────────────────────────────────
with col_viz1:
    st.markdown("<p class='section-title'>Feature Distribution</p>",
                unsafe_allow_html=True)

    fig1, ax1 = plt.subplots(figsize=(5, 3))
    fig1.patch.set_facecolor(BG_PLOT)
    ax1.set_facecolor(BG_PLOT)

    n, bins, patches = ax1.hist(X_scaled.flatten(), bins=60, color=ACCENT2,
                                 edgecolor='none', alpha=0.85)
    # gradient-colour the bars
    for i, patch in enumerate(patches):
        t = i / len(patches)
        r = int(0xE8 * t + 0x7B * (1 - t))
        g = int(0x43 * t + 0x2F * (1 - t))
        b = int(0x93 * t + 0xBE * (1 - t))
        patch.set_facecolor(f"#{r:02x}{g:02x}{b:02x}")

    ax1.set_xlabel("Scaled Value", color=TEXT_MUT, fontsize=8)
    ax1.set_ylabel("Frequency",    color=TEXT_MUT, fontsize=8)
    ax1.tick_params(colors=TEXT_MUT, labelsize=7)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#1E1E32")
    fig1.tight_layout()
    st.pyplot(fig1)

# ── Graph Structure ──────────────────────────────────────────────────────────
with col_viz2:
    st.markdown("<p class='section-title'>Graph Structure (sample)</p>",
                unsafe_allow_html=True)

    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    nodes_available = list(G.nodes())
    n_sample = min(sample_size, len(nodes_available))
    sample_nodes = random.sample(nodes_available, n_sample)
    H = G.subgraph(sample_nodes)

    node_colors = [
        "#22C55E" if y[n] == 1 else ACCENT
        for n in H.nodes()
    ]

    fig2, ax2 = plt.subplots(figsize=(5, 3))
    fig2.patch.set_facecolor(BG_PLOT)
    ax2.set_facecolor(BG_PLOT)

    pos = nx.spring_layout(H, seed=42, k=0.5)
    nx.draw_networkx_edges(H, pos, ax=ax2,
                           edge_color="#2A2A4A", width=0.6, alpha=0.7)
    nx.draw_networkx_nodes(H, pos, ax=ax2,
                           node_color=node_colors, node_size=35, alpha=0.95)

    benign_patch    = mpatches.Patch(color="#22C55E", label="Benign")
    malignant_patch = mpatches.Patch(color=ACCENT,   label="Malignant")
    ax2.legend(handles=[benign_patch, malignant_patch],
               facecolor=BG_CARD, edgecolor="#1E1E32",
               labelcolor="#E8E8F0", fontsize=7, loc="upper right")

    ax2.axis("off")
    fig2.tight_layout()
    st.pyplot(fig2)


# ── Class balance bar ────────────────────────────────────────────────────────
st.markdown("<p class='section-title'>Class Balance</p>",
            unsafe_allow_html=True)

benign_pct    = (y == 1).sum() / len(y) * 100
malignant_pct = 100 - benign_pct

fig3, ax3 = plt.subplots(figsize=(8, 0.55))
fig3.patch.set_facecolor(BG_PLOT)
ax3.set_facecolor(BG_PLOT)
ax3.barh(0, benign_pct,               color="#22C55E", height=0.5, label="Benign")
ax3.barh(0, malignant_pct, left=benign_pct, color=ACCENT, height=0.5, label="Malignant")
ax3.set_xlim(0, 100)
ax3.axis("off")
ax3.legend(facecolor=BG_CARD, edgecolor="#1E1E32",
           labelcolor="#E8E8F0", fontsize=8,
           loc="center right", bbox_to_anchor=(1.12, 0.5))
for spine in ax3.spines.values():
    spine.set_visible(False)
fig3.tight_layout(pad=0)
st.pyplot(fig3)


# ============================================================
# PATIENT INPUT FORM
# ============================================================

st.markdown("<p class='section-title'>Patient Feature Input</p>",
            unsafe_allow_html=True)

st.markdown("""
<div class='card'>
  <small style='color:#8888AA'>
    Enter the 30 diagnostic measurements for the patient.
    Default values are pre-filled with dataset means.
  </small>
</div>
""", unsafe_allow_html=True)

# Lay out inputs in 3 columns for compactness
user_input  = []
cols        = st.columns(3)
col_groups  = [feature_names[i::3] for i in range(3)]
idx_tracker = [0, 0, 0]

all_features_flat = list(feature_names)
input_values      = {}

cols3 = [st.columns(3) for _ in range(10)]   # 10 rows × 3 cols = 30 features

for row_idx in range(10):
    for col_idx in range(3):
        feat_idx = row_idx * 3 + col_idx
        if feat_idx < len(all_features_flat):
            fname = all_features_flat[feat_idx]
            default_val = float(np.mean(X[:, feat_idx]))
            val = cols3[row_idx][col_idx].number_input(
                label=fname,
                value=default_val,
                format="%.4f",
                key=f"feat_{feat_idx}"
            )
            input_values[fname] = val

user_input = [input_values[f] for f in all_features_flat]


# ============================================================
# PREDICT
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_col, _ = st.columns([1, 2])

with predict_col:
    predict_clicked = st.button("🔬  Run GNN Prediction")

if predict_clicked:
    user_array  = np.array(user_input).reshape(1, -1)
    user_scaled = scaler.transform(user_array)
    user_tensor = torch.tensor(user_scaled, dtype=torch.float)

    # Append the new node to the graph (no edges — isolated node inference)
    extended_x = torch.cat([data_graph.x, user_tensor], dim=0)

    with torch.no_grad():
        output     = model(extended_x, data_graph.edge_index)
        logits     = output[-1]                      # last node = query patient
        probs      = torch.softmax(logits, dim=-1)
        prediction = torch.argmax(logits).item()

    benign_prob    = probs[1].item() * 100
    malignant_prob = probs[0].item() * 100

    st.markdown("<p class='section-title'>Prediction Result</p>",
                unsafe_allow_html=True)

    res_col, conf_col = st.columns([3, 2])

    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class='pred-benign'>
              <p class='pred-label' style='color:#22C55E'>✓ Benign</p>
              <p class='pred-sub'>The GNN classifies this sample as <b>non-malignant</b>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pred-malignant'>
              <p class='pred-label' style='color:{ACCENT}'>⚠ Malignant</p>
              <p class='pred-sub'>The GNN classifies this sample as <b>malignant</b>. Please consult a physician.</p>
            </div>
            """, unsafe_allow_html=True)

    with conf_col:
        st.markdown("<p class='section-title'>Confidence</p>",
                    unsafe_allow_html=True)

        fig4, ax4 = plt.subplots(figsize=(3.5, 2.6))
        fig4.patch.set_facecolor(BG_PLOT)
        ax4.set_facecolor(BG_PLOT)

        bars = ax4.barh(
            ["Malignant", "Benign"],
            [malignant_prob, benign_prob],
            color=[ACCENT, "#22C55E"],
            height=0.45,
            edgecolor="none"
        )
        ax4.set_xlim(0, 100)
        ax4.set_xlabel("Probability (%)", color=TEXT_MUT, fontsize=8)
        ax4.tick_params(colors=TEXT_MUT, labelsize=8)
        for spine in ax4.spines.values():
            spine.set_edgecolor("#1E1E32")
        for bar, val in zip(bars, [malignant_prob, benign_prob]):
            ax4.text(val + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center",
                     color="#E8E8F0", fontsize=8)
        fig4.tight_layout()
        st.pyplot(fig4)

    st.markdown("""
    <div class='card' style='margin-top:1rem;'>
      <small style='color:#8888AA'>
        ⚠️ <b>Disclaimer:</b> This tool is for research and educational purposes only.
        It is not a substitute for professional medical diagnosis or advice.
      </small>
    </div>
    """, unsafe_allow_html=True)