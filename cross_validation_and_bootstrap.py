"""
=============================================================================
FILE 7: CROSS-VALIDATION + BOOTSTRAP CONFIDENCE INTERVALS
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Transactions

Implements two new-additions requirements:
  - Stratified k-fold cross-validation (k=5) for BOTH the FraudTransformer
    baseline and the XGBoost baseline, on the 18 leakage-audited features.
  - Bootstrap confidence intervals (95%, 1000 resamples) for every headline
    metric (AUC, F1, Precision, Recall, Specificity) on the held-out test set,
    for baseline Transformer, HAD-Stack Transformer, and XGBoost.

Transformer CV uses a compact training budget (fewer epochs) purely so that
5 folds finish in reasonable CPU time; the architecture and loss are identical
to baseline_model.py.
=============================================================================
"""
import numpy as np, torch, torch.nn as nn, json, os, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
import xgboost as xgb
from baseline_model import TabularTransformer, FocalLoss

DEVICE   = torch.device("cpu")
OUT      = "processed_data"
EVAL_DIR = "evaluation_results"
MODEL_DIR= "models"
os.makedirs(EVAL_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)


def load_all():
    # Re-pool train+val for CV; keep test held out for bootstrap CIs.
    Xtr = np.load(f"{OUT}/X_train.npy"); ytr = np.load(f"{OUT}/y_train.npy")
    Xva = np.load(f"{OUT}/X_val.npy");   yva = np.load(f"{OUT}/y_val.npy")
    Xte = np.load(f"{OUT}/X_test.npy");  yte = np.load(f"{OUT}/y_test.npy")
    X_cv = np.concatenate([Xtr, Xva]).astype(np.float32)
    y_cv = np.concatenate([ytr, yva]).astype(np.int64)
    return X_cv, y_cv, Xte.astype(np.float32), yte.astype(np.int64)


def metric_set(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    return dict(auc=roc_auc_score(y, p),
                f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                specificity=cm[0,0]/(cm[0,:].sum()+1e-9))


# -------------------------------------------------------------------------
# Transformer training (compact budget for CV speed)
# -------------------------------------------------------------------------
def train_transformer(Xtr, ytr, Xva, yva, epochs=20, bs=256, lr=3e-4):
    torch.manual_seed(SEED)
    model = TabularTransformer(n_features=Xtr.shape[1]).to(DEVICE)
    crit  = FocalLoss(gamma=2.0, alpha=0.75)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    n = len(Xtr_t); best_auc = 0.0; best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            xb, yb = Xtr_t[idx].to(DEVICE), ytr_t[idx].to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = 1/(1+np.exp(-model(torch.tensor(Xva).to(DEVICE)).cpu().numpy()))
        a = roc_auc_score(yva, pv)
        if a > best_auc:
            best_auc = a
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def transformer_probs(model, X):
    return 1/(1+np.exp(-model(torch.tensor(X, dtype=torch.float32).to(DEVICE)).cpu().numpy()))


def train_xgb(Xtr, ytr, Xva, yva):
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_lambda=1.0, reg_alpha=0.1,
        scale_pos_weight=spw, eval_metric='auc',
        random_state=SEED, n_jobs=4, tree_method='hist')
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


# -------------------------------------------------------------------------
# (A) 5-FOLD STRATIFIED CROSS-VALIDATION
# -------------------------------------------------------------------------
def cross_validation(k=5):
    print("="*70)
    print(f"(A) {k}-FOLD STRATIFIED CROSS-VALIDATION")
    print("="*70)
    X, y, _, _ = load_all()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    cv = {'transformer': [], 'xgboost': []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xva, yva = X[va_idx], y[va_idx]
        t0 = time.time()

        # Transformer
        tf = train_transformer(Xtr, ytr, Xva, yva, epochs=20)
        m_tf = metric_set(yva, transformer_probs(tf, Xva))
        cv['transformer'].append(m_tf)

        # XGBoost
        xg = train_xgb(Xtr, ytr, Xva, yva)
        m_xg = metric_set(yva, xg.predict_proba(Xva)[:, 1])
        cv['xgboost'].append(m_xg)

        print(f"  Fold {fold}/{k} ({time.time()-t0:5.1f}s) | "
              f"TF  AUC={m_tf['auc']:.4f} F1={m_tf['f1']:.4f} | "
              f"XGB AUC={m_xg['auc']:.4f} F1={m_xg['f1']:.4f}")

    summary = {}
    print("-"*70)
    for name in ['transformer', 'xgboost']:
        agg = {}
        for met in ['auc', 'f1', 'precision', 'recall', 'specificity']:
            vals = np.array([f[met] for f in cv[name]])
            agg[met] = dict(mean=float(vals.mean()), std=float(vals.std()),
                            values=[float(v) for v in vals])
        summary[name] = agg
        print(f"  {name:<12} | AUC {agg['auc']['mean']:.4f}+/-{agg['auc']['std']:.4f}"
              f" | F1 {agg['f1']['mean']:.4f}+/-{agg['f1']['std']:.4f}"
              f" | Prec {agg['precision']['mean']:.4f}+/-{agg['precision']['std']:.4f}"
              f" | Rec {agg['recall']['mean']:.4f}+/-{agg['recall']['std']:.4f}")

    with open(f"{EVAL_DIR}/cross_validation.json", 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


# -------------------------------------------------------------------------
# (B) BOOTSTRAP CONFIDENCE INTERVALS on the held-out test set
# -------------------------------------------------------------------------
def bootstrap_ci(y, p, n_boot=1000, alpha=0.05):
    rng = np.random.default_rng(SEED)
    n = len(y)
    acc = {m: [] for m in ['auc', 'f1', 'precision', 'recall', 'specificity']}
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        if y[idx].sum() < 2 or (y[idx] == 0).sum() < 2:
            continue
        ms = metric_set(y[idx], p[idx])
        for m in acc:
            acc[m].append(ms[m])
    ci = {}
    for m, vals in acc.items():
        vals = np.array(vals)
        ci[m] = dict(mean=float(vals.mean()),
                     lo=float(np.percentile(vals, 100*alpha/2)),
                     hi=float(np.percentile(vals, 100*(1-alpha/2))))
    return ci


def bootstrap_all():
    print("\n" + "="*70)
    print("(B) BOOTSTRAP 95% CONFIDENCE INTERVALS (n=1000, test set)")
    print("="*70)
    X, y, Xte, yte = load_all()
    out = {}

    # Baseline Transformer (saved checkpoint)
    bt = TabularTransformer(n_features=Xte.shape[1]).to(DEVICE)
    bt.load_state_dict(torch.load(f"{MODEL_DIR}/baseline_best.pt", map_location=DEVICE))
    bt.eval()
    p_base = transformer_probs(bt, Xte)
    out['baseline'] = bootstrap_ci(yte, p_base)

    # HAD-Stack Transformer (saved checkpoint)
    if os.path.exists(f"{MODEL_DIR}/adversarial_best.pt"):
        at = TabularTransformer(n_features=Xte.shape[1]).to(DEVICE)
        at.load_state_dict(torch.load(f"{MODEL_DIR}/adversarial_best.pt", map_location=DEVICE))
        at.eval()
        p_adv = transformer_probs(at, Xte)
        out['had_stack'] = bootstrap_ci(yte, p_adv)

    # XGBoost (retrain on train+val for the held-out test bootstrap)
    Xtr = np.load(f"{OUT}/X_train.npy"); ytr = np.load(f"{OUT}/y_train.npy")
    Xva = np.load(f"{OUT}/X_val.npy");   yva = np.load(f"{OUT}/y_val.npy")
    xg = train_xgb(Xtr, ytr, Xva, yva)
    p_xgb = xg.predict_proba(Xte)[:, 1]
    out['xgboost'] = bootstrap_ci(yte, p_xgb)

    for name, ci in out.items():
        print(f"  {name}")
        for m, v in ci.items():
            print(f"     {m:<12} {v['mean']:.4f}  [{v['lo']:.4f}, {v['hi']:.4f}]")
    with open(f"{EVAL_DIR}/bootstrap_ci.json", 'w') as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    cross_validation(k=5)
    bootstrap_all()
    print("\n[DONE] Cross-validation + bootstrap CI results saved.")
