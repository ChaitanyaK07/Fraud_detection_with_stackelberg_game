# HAD-Stack: Hamiltonian-Guided Stackelberg Game for Adversarially Robust Fraud Detection

**Paper:** *A Hamiltonian Stackelberg Game Approach for Adversarially Robust Fraud Detection in Digital Banking Transactions*  
**Submitted to:** DSAA 2026 (Double-Blind)

---

## Overview

HAD-Stack is a physics-inspired adversarial training framework for fraud detection
in UPI and banking transactions. Instead of manually tuning adversarial
hyperparameters, it uses three self-consistent quantum Hamiltonian solvers
to dynamically compute:

- **H2 spectral gap** → attacker perturbation budget ε each game round
- **H1 ground-state energy** → defender regularisation weight λ each round
- **H29 equilibrium field** → Nash equilibrium stopping criterion

The result is a Transformer-based fraud detector that self-regulates as the
adversarial threat level changes, requiring no manual hyperparameter tuning
for the adversarial training loop.

---

## Results Summary

| Metric | Baseline | HAD-Stack (clean) | HAD-Stack (under attack) |
|--------|----------|-------------------|--------------------------|
| AUC-ROC | 0.9957 | 0.9961 | **1.0000** |
| F1-Score | 0.9290 | 0.9353 | 0.9934 |
| Precision | 0.9373 | **0.9854** | 0.9870 |
| Recall | 0.9208 | 0.8900 | **1.0000** |
| False Positives | 36 | **18 (−50%)** | — |

Statistical significance: p < 0.000001 (paired t-test, n=1,000 bootstrap resamples)

---

## Repository Structure

```
HAD-Stack/
├── README.md               # this file
├── requirements.txt        # pip dependencies
├── data/
│   ├── README_data.md      # instructions for placing the dataset
│   └── fraud_dataset.csv   # place dataset here (see data/README_data.md)
├── 1_data_prep.py          # leakage audit, feature engineering, train/val/test split
├── 2_baseline_model.py     # FraudTransformer training (clean data)
├── 3_robust_model.py       # Hamiltonian-Stackelberg adversarial training
├── 4_evaluation.py         # full evaluation: clean, evasion, poisoning, t-test
└── models/                 # checkpoints saved here at runtime
    ├── baseline_best.pt
    └── adversarial_best.pt
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/HAD-Stack.git
cd HAD-Stack

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset
# See data/README_data.md for instructions
```

---

## Running the Pipeline

Run the four scripts in order. Each one saves its outputs so the next
script can load them.

```bash
# Step 1 — Leakage audit, feature engineering, save processed splits
python 1_data_prep.py

# Step 2 — Train baseline FraudTransformer on clean data
python 2_baseline_model.py

# Step 3 — Hamiltonian-Stackelberg adversarial training
python 3_robust_model.py

# Step 4 — Full evaluation (clean, evasion attack, poisoning, statistical test)
python 4_evaluation.py
```

Each script prints its progress and final metrics to stdout.
Trained model checkpoints are saved to `models/`.

---

## Dataset

The dataset contains 26,393 UPI and banking transactions with 65 raw features.
See `data/README_data.md` for where to obtain it and how to place it.

**Important — leakage audit:**  
The dataset is synthetically generated. 25+ features directly encode the
class label (e.g. `handle_verification_status` is 100% correlated with fraud).
`1_data_prep.py` identifies and removes all leaking features automatically,
retaining 18 genuinely informative features. Running the model on the raw
65 features produces trivial AUC = 1.0 — this is label memorisation, not learning.

---

## Architecture

**FraudTransformer (baseline)**
```
Input: 18 features (post-leakage audit)
  → Linear projection: 18 → 64
  → Transformer block x2: 4-head attention (d_k=16), FFN 64→64, LayerNorm, Dropout(0.4)
  → MLP head: 64 → 32 → Dropout(0.4) → 1
  → Sigmoid → fraud probability
Parameters: 72,577
Loss: BCE with class weight 4.81 (handles 4.8:1 imbalance)
```

**HAD-Stack (robust)**  
Same architecture. Training loop is wrapped in a Stackelberg game:
each round, H2 sets ε → attacker crafts FGSM+poison batch → H1 sets λ →
defender retrains with L2 penalty → H29 checks Nash equilibrium → stop if converged.

---

## Reproducibility

All random seeds are fixed to 42:
```python
import random, numpy as np, torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

---

## Citation

```
@article{hadstack2026,
  title   = {A Hamiltonian Stackelberg Game Approach for Adversarially Robust
             Fraud Detection in Digital Banking Transactions},
  author  = {Anonymous},
  journal = {DSAA 2026 (under review)},
  year    = {2026}
}
```
