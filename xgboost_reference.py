"""
=============================================================================
FILE 5: XGBOOST BASELINE + ATTACK TRANSFERABILITY
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Transactions

Purpose (per new-additions requirements):
  - Train an XGBoost fraud detector on the SAME 18 leakage-audited features.
  - Demonstrate that the Hamiltonian-constrained adversarial attacks
    (evasion + poisoning) are GENERAL: they degrade BOTH the Transformer
    and the XGBoost model. Attacks are crafted on the Transformer (white-box)
    and TRANSFERRED to XGBoost (black-box) -> transfer attack.
=============================================================================
"""
import numpy as np, torch, pickle, os, json, warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix, average_precision_score)
from baseline_model import TabularTransformer
from adversarial_game import HamiltonianPerturbationBudget, HamiltonianAttacker

DEVICE   = torch.device("cpu")
OUT      = "processed_data"
MODEL_DIR= "models"
EVAL_DIR = "evaluation_results"
os.makedirs(EVAL_DIR, exist_ok=True)
SEED = 42
np.random.seed(SEED)


def load():
    d = {}
    for s in ['X_train','X_val','X_test','y_train','y_val','y_test']:
        d[s] = np.load(f"{OUT}/{s}.npy")
    return d


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred)
    return dict(auc=roc_auc_score(y, p), ap=average_precision_score(y, p),
                f1=f1_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                specificity=cm[0,0]/(cm[0,:].sum()+1e-9),
                cm=cm)


def train_xgboost(d):
    print("="*70); print("XGBOOST FRAUD DETECTOR"); print("="*70)
    spw = (d['y_train']==0).sum() / (d['y_train']==1).sum()
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_lambda=1.0, reg_alpha=0.1,
        scale_pos_weight=spw, eval_metric='auc',
        random_state=SEED, n_jobs=4, tree_method='hist',
    )
    clf.fit(d['X_train'], d['y_train'],
            eval_set=[(d['X_val'], d['y_val'])], verbose=False)
    p_test = clf.predict_proba(d['X_test'])[:,1]
    m = metrics(d['y_test'], p_test)
    print(f"  AUC-ROC   : {m['auc']:.4f}")
    print(f"  F1-Score  : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  Spec.     : {m['specificity']:.4f}")
    cm = m['cm']
    print(f"  CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")
    clf.save_model(f"{MODEL_DIR}/xgboost_model.json")
    return clf, m, p_test


def transformer_for_attack(n_features):
    model = TabularTransformer(n_features=n_features).to(DEVICE)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/baseline_best.pt", map_location=DEVICE))
    model.eval()
    return model


def run():
    d = load()
    n_features = d['X_train'].shape[1]
    clf, m_clean, p_clean = train_xgboost(d)

    # Hamiltonian attacker (same budget object the Transformer game uses)
    budget   = HamiltonianPerturbationBudget()
    attacker = HamiltonianAttacker(budget, n_steps=10, step_size=0.01)
    surrogate = transformer_for_attack(n_features)   # white-box surrogate

    Xt = torch.tensor(d['X_test'], dtype=torch.float32)
    yt = torch.tensor(d['y_test'], dtype=torch.int64)

    # ---- EVASION transfer: craft on Transformer, evaluate on XGBoost ----
    Xadv = attacker.evasion_attack(surrogate, Xt.to(DEVICE), yt.to(DEVICE)).cpu().numpy()
    p_eva = clf.predict_proba(Xadv)[:,1]
    m_eva = metrics(d['y_test'], p_eva)

    # ---- POISONING transfer at multiple rates ----
    poison = {}
    for r in [0.05, 0.10, 0.20]:
        Xpoi = attacker.poisoning_attack(Xt.to(DEVICE), poison_rate=r).cpu().numpy()
        p_poi = clf.predict_proba(Xpoi)[:,1]
        poison[r] = metrics(d['y_test'], p_poi)

    # ---- Also measure the Transformer itself under the SAME attacks ----
    @torch.no_grad()
    def tf_probs(X):
        return 1/(1+np.exp(-surrogate(torch.tensor(X,dtype=torch.float32).to(DEVICE)).cpu().numpy()))
    tf_clean = metrics(d['y_test'], tf_probs(d['X_test']))
    tf_eva   = metrics(d['y_test'], tf_probs(Xadv))

    print("\n" + "="*70)
    print("ATTACK TRANSFERABILITY  (attacks crafted on Transformer surrogate)")
    print("="*70)
    print(f"{'Model':<22}{'Clean AUC':>12}{'Evasion AUC':>14}{'AUC drop':>12}")
    print("-"*70)
    print(f"{'Transformer':<22}{tf_clean['auc']:>12.4f}{tf_eva['auc']:>14.4f}"
          f"{tf_clean['auc']-tf_eva['auc']:>12.4f}")
    print(f"{'XGBoost (transfer)':<22}{m_clean['auc']:>12.4f}{m_eva['auc']:>14.4f}"
          f"{m_clean['auc']-m_eva['auc']:>12.4f}")
    print("-"*70)
    print(f"{'Model':<22}{'Clean F1':>12}{'5% poison':>14}{'20% poison':>12}")
    print("-"*70)
    print(f"{'XGBoost':<22}{m_clean['f1']:>12.4f}{poison[0.05]['f1']:>14.4f}"
          f"{poison[0.20]['f1']:>12.4f}")

    results = dict(
        xgb_clean=m_clean, xgb_evasion=m_eva,
        xgb_poison={f"poison_{r}": poison[r] for r in poison},
        tf_clean=tf_clean, tf_evasion=tf_eva,
        eps_global=float(budget.eps_global),
    )
    with open(f"{EVAL_DIR}/xgboost_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    # JSON-safe summary
    def js(m):
        return {k:(float(v) if not isinstance(v,np.ndarray) else v.tolist())
                for k,v in m.items()}
    summary = dict(
        xgb_clean=js(m_clean), xgb_evasion=js(m_eva),
        xgb_poison={f"poison_{r}":js(poison[r]) for r in poison},
        tf_clean=js(tf_clean), tf_evasion=js(tf_eva),
        eps_global=float(budget.eps_global))
    with open(f"{EVAL_DIR}/xgboost_results.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] XGBoost + transfer results saved to {EVAL_DIR}/")
    return results


if __name__ == "__main__":
    run()
