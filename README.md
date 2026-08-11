# Transformer Dynamics Lab: Empirical Representation State-Space & Layer Collapse Analysis

An interactive deep learning research platform for analyzing hidden state dynamics, token representation collapse, singular value entropy, and Neural Stiffness Index ($NSI$) across Transformer architectures.

---

## 🌟 Key Research Capabilities

1. **4 Core Mathematical Metrics Engine:**
   - **Pairwise Distance:** Mean $L_2$ Euclidean distance across sequence tokens per layer $\|X_i^{(l)} - X_j^{(l)}\|_2$.
   - **Cosine Similarity (SAM):** Pairwise directional alignment quantifying representation convergence $\frac{X_i \cdot X_j}{\|X_i\| \|X_j\|}$.
   - **Effective Rank:** Dimensional diversity of state space computed via singular value entropy $H(S) = -\sum p_k \log p_k$.
   - **Neural Stiffness Index (NSI):** Relative layer-to-layer feature transformation velocity $NSI_l = \frac{\|X^{(l+1)} - X^{(l)}\|_F}{\|X^{(l)}\|_F}$.

2. **4-Stage Analytical Workflow Framework:**
   - **Stage 1 — Observation:** Real-time metrics curves, 2D PCA token trajectories, and paper benchmark visuals (GPT-2 vs. BERT empirical figures).
   - **Stage 2 — Understanding:** Visual Sentence Layer-by-Layer Transformation Flowchart, interactive layer slider ($0 \dots L$), and structured 4-card mechanics grid (*Layer Stage*, *Stiffness Diagnosis*, *Math Formulation*, *Sub-component Breakdown*).
   - **Stage 3 — Controlled Experimentation:** Single-variable parameter ablations including layer pruning, block range pruning (LaCo), positional encoding zeroing, and sub-layer scaling.
   - **Stage 4 — Results & Recommendations:** Baseline vs. modified metric overlays, percentage shifts, and literature-backed architectural pruning advice.

3. **Multi-Model Support:**
   - `GPT-2` (Autoregressive 12-layer)
   - `DistilGPT2` (Lightweight Autoregressive 6-layer)
   - `BERT Base` (Bidirectional 12-layer)
   - `RoBERTa Base` (Bidirectional 12-layer)
   - `ODE Attention Simulator` (Continuous Dynamical Self-Attention Model)

---

## ⚡ Quickstart Guide

### Prerequisites
- Python 3.8+
- PyTorch 2.0+

### Installation & Launch

```bash
# 1. Clone Repository
git clone https://github.com/YOUR_USERNAME/transformer-dynamics.git
cd transformer-dynamics

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Launch Platform
python start.py
```

Open **`http://localhost:8050`** in your browser.

---

## 🔬 Literature References & Theoretical Foundation

- **LaCo (2024):** Layer Collapse in Deep Transformers & Block Pruning.
- **ShortGPT (2024):** Redundancy and Middle-Layer Pruning in Autoregressive LLMs.
- **SLEB (2024):** Streamlining Layer Elimination in Transformer Architectures.
- **Neural Stiffness Index (2025):** Differential State Space Velocity and Stability in Self-Attention Networks.

---

## 📁 Repository Structure

```
transformer-dynamics/
├── backend/
│   ├── app.py                # FastAPI REST Server & Endpoints
│   ├── dynamics_engine.py    # PyTorch Forward Tensor Hooks & Token Decoding
│   ├── metrics.py            # Core Metrics Engine (Distance, Cosine Sim, SVD Rank, NSI)
│   └── test_dynamics.py      # Unit Test Suite
├── frontend/
│   ├── index.html            # 4-Stage Single Page App Interface
│   └── src/
│       └── app.js            # REST client, Plotly.js chart rendering & state
├── start.py                  # One-command launcher script
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```
