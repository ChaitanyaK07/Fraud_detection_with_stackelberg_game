"""
=============================================================================
FILE 4: ROBUSTNESS EVALUATION SUITE
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Transactions
Advisor: Aneesh Chivukula

Evaluation protocol:
  1. Standard metrics: AUC-ROC, F1, Precision, Recall, confusion matrices
  2. Evasion robustness: accuracy under PGA evasion attacks
  3. Poisoning robustness: accuracy under training data poisoning
  4. Statistical significance: paired t-test (baseline vs adversarial)
  5. Empirical results table for paper
=============================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, average_precision_score
)
from scipy import stats
import os, pickle, warnings
warnings.filterwarnings('ignore')

from baseline_model import TabularTransformer, evaluate
from adversarial_game import (
    HamiltonianPerturbationBudget, HamiltonianAttacker, load_processed_data
)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
EVAL_DIR  = "evaluation_results"
os.makedirs(EVAL_DIR, exist_ok=True)

N_TRIALS  = 10    # bootstrap / attack repetitions


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, n_features: int) -> nn.Module:
    model = TabularTransformer(n_features=n_features).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model


def metrics_dict(labels: np.ndarray, probs: np.ndarray,
                 threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        'auc':       roc_auc_score(labels, probs),
        'ap':        average_precision_score(labels, probs),
        'f1':        f1_score(labels, preds, zero_division=0),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall':    recall_score(labels, preds, zero_division=0),
        'specificity': (confusion_matrix(labels, preds)[0, 0] /
                        (confusion_matrix(labels, preds)[0, :].sum() + 1e-9)),
        'cm':        confusion_matrix(labels, preds),
    }


def print_metrics(tag: str, m: dict):
    print(f"\n{'─'*60}")
    print(f"  {tag}")
    print(f"{'─'*60}")
    print(f"  AUC-ROC     : {m['auc']:.4f}")
    print(f"  Avg Prec    : {m['ap']:.4f}")
    print(f"  F1-Score    : {m['f1']:.4f}")
    print(f"  Precision   : {m['precision']:.4f}")
    print(f"  Recall      : {m['recall']:.4f}")
    print(f"  Specificity : {m['specificity']:.4f}")
    print(f"  Confusion Matrix:")
    cm = m['cm']
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")


# ─────────────────────────────────────────────────────────────
# 1. STANDARD EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_standard(model: nn.Module, loader: DataLoader, tag: str) -> dict:
    """Clean data evaluation."""
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in loader:
            all_logits.append(model(Xb.to(DEVICE)).cpu())
            all_labels.append(yb)
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    m = metrics_dict(labels, probs)
    print_metrics(tag, m)
    return {**m, 'probs': probs, 'labels': labels}


# ─────────────────────────────────────────────────────────────
# 2. EVASION ROBUSTNESS
# ─────────────────────────────────────────────────────────────

def evaluate_evasion(model: nn.Module, loader: DataLoader,
                     attacker: HamiltonianAttacker, tag: str) -> dict:
    """Evaluate model under PGA evasion attack."""
    model.eval()
    all_logits, all_labels = [], []

    for Xb, yb in loader:
        Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
        Xb_adv = attacker.evasion_attack(model, Xb, yb)
        with torch.no_grad():
            all_logits.append(model(Xb_adv).cpu())
        all_labels.append(yb.cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = 1 / (1 + np.exp(-logits))
    m = metrics_dict(labels, probs)
    print_metrics(f"{tag} [EVASION ATTACK]", m)
    return {**m, 'probs': probs, 'labels': labels}


# ─────────────────────────────────────────────────────────────
# 3. POISONING ROBUSTNESS
# ─────────────────────────────────────────────────────────────

def evaluate_poisoning(model: nn.Module, test_loader: DataLoader,
                        attacker: HamiltonianAttacker, tag: str,
                        poison_rates: list = [0.05, 0.10, 0.20]) -> dict:
    """
    Evaluate model robustness to feature-level poisoning attacks.
    Tests multiple poison rates and reports degradation curve.
    """
    print(f"\n{'─'*60}")
    print(f"  {tag} — POISONING ATTACK ROBUSTNESS")
    print(f"{'─'*60}")

    clean_m = evaluate_standard(model, test_loader, f"{tag} [CLEAN baseline]")
    results = {'clean': clean_m}

    for rate in poison_rates:
        all_logits, all_labels = [], []
        model.eval()
        for Xb, yb in test_loader:
            Xb_poison = attacker.poisoning_attack(Xb.to(DEVICE), poison_rate=rate)
            with torch.no_grad():
                all_logits.append(model(Xb_poison).cpu())
            all_labels.append(yb)

        logits = torch.cat(all_logits).numpy()
        labels = torch.cat(all_labels).numpy()
        probs  = 1 / (1 + np.exp(-logits))
        m = metrics_dict(labels, probs)
        print(f"  Poison rate {rate*100:4.1f}% | "
              f"AUC: {m['auc']:.4f} | F1: {m['f1']:.4f} | "
              f"ΔF1: {m['f1'] - clean_m['f1']:+.4f}")
        results[f'poison_{rate}'] = m

    return results


# ─────────────────────────────────────────────────────────────
# 4. STATISTICAL SIGNIFICANCE — PAIRED t-TEST
# ─────────────────────────────────────────────────────────────

def statistical_significance_test(baseline_probs: np.ndarray,
                                   adv_probs: np.ndarray,
                                   labels: np.ndarray) -> dict:
    """
    Paired t-test comparing AUC scores across bootstrap resamples.
    H0: baseline AUC == adversarial AUC
    """
    print(f"\n{'─'*60}")
    print("  STATISTICAL SIGNIFICANCE — Paired t-Test (n=1000 bootstraps)")
    print(f"{'─'*60}")

    n_bootstrap = 1000
    baseline_aucs = []
    adv_aucs      = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstrap):
        idx = rng.choice(len(labels), size=len(labels), replace=True)
        if labels[idx].sum() < 2:
            continue
        baseline_aucs.append(roc_auc_score(labels[idx], baseline_probs[idx]))
        adv_aucs.append(roc_auc_score(labels[idx], adv_probs[idx]))

    t_stat, p_value = stats.ttest_rel(adv_aucs, baseline_aucs)

    print(f"  Baseline AUC  : {np.mean(baseline_aucs):.4f} ± {np.std(baseline_aucs):.4f}")
    print(f"  Adversarial AUC: {np.mean(adv_aucs):.4f} ± {np.std(adv_aucs):.4f}")
    print(f"  t-statistic   : {t_stat:.4f}")
    print(f"  p-value       : {p_value:.6f}")
    print(f"  Significant   : {'YES (p < 0.05)' if p_value < 0.05 else 'NO'}")

    return {
        'baseline_mean': np.mean(baseline_aucs),
        'baseline_std':  np.std(baseline_aucs),
        'adv_mean':      np.mean(adv_aucs),
        'adv_std':       np.std(adv_aucs),
        't_stat':        t_stat,
        'p_value':       p_value,
    }


# ─────────────────────────────────────────────────────────────
# 5. EMPIRICAL RESULTS TABLE
# ─────────────────────────────────────────────────────────────

def print_results_table(baseline_clean: dict, baseline_evasion: dict,
                        adv_clean: dict, adv_evasion: dict,
                        ttest: dict):
    """Print LaTeX-ready results table for the paper."""
    print("\n" + "=" * 70)
    print("EMPIRICAL RESULTS TABLE (for IEEE Paper)")
    print("=" * 70)
    header = f"{'Method':<35} {'AUC':>7} {'F1':>7} {'Prec':>7} {'Recall':>7} {'Spec':>7}"
    print(header)
    print("-" * 70)

    rows = [
        ("Baseline (clean)",            baseline_clean),
        ("Baseline (evasion attack)",   baseline_evasion),
        ("Adversarial (clean)",         adv_clean),
        ("Adversarial (evasion attack)", adv_evasion),
    ]
    for name, m in rows:
        print(f"  {name:<33} {m['auc']:>7.4f} {m['f1']:>7.4f} "
              f"{m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['specificity']:>7.4f}")

    print("-" * 70)
    gain = adv_clean['auc'] - baseline_clean['auc']
    rob  = adv_evasion['auc'] - baseline_evasion['auc']
    print(f"\n  AUC gain (adversarial vs baseline, clean) : {gain:+.4f}")
    print(f"  Robustness gain (evasion)                  : {rob:+.4f}")
    print(f"  p-value (paired t-test)                    : {ttest['p_value']:.6f}")

    df = pd.DataFrame([
    {
        "Method": name,
        "AUC": m["auc"],
        "F1": m["f1"],
        "Precision": m["precision"],
        "Recall": m["recall"],
        "Specificity": m["specificity"]
    }
    for name, m in rows
    ])  

    print("\n--- Fraud Detection Performance Comparison ---")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────

def run_evaluation():
    print("=" * 70)
    print("ROBUSTNESS EVALUATION SUITE")
    print("=" * 70)

    train_loader, val_loader, test_loader, n_features = load_processed_data()

    # Load both models
    baseline_ckpt = f"{MODEL_DIR}/baseline_best.pt"
    adv_ckpt      = f"{MODEL_DIR}/adversarial_best.pt"

    print(f"[INFO] Loading models from {MODEL_DIR} …")
    baseline_model = load_model(baseline_ckpt, n_features)
    adv_model      = load_model(adv_ckpt, n_features)

    # Hamiltonian budget for attacks
    budget   = HamiltonianPerturbationBudget()
    attacker = HamiltonianAttacker(budget, n_steps=10, step_size=0.01)

    print("\n" + "=" * 70)
    print("1. STANDARD (CLEAN) EVALUATION")
    print("=" * 70)
    baseline_clean = evaluate_standard(baseline_model, test_loader, "Baseline")
    adv_clean      = evaluate_standard(adv_model,      test_loader, "Adversarial")

    print("\n" + "=" * 70)
    print("2. EVASION ATTACK ROBUSTNESS")
    print("=" * 70)
    baseline_evasion = evaluate_evasion(baseline_model, test_loader, attacker, "Baseline")
    adv_evasion      = evaluate_evasion(adv_model,      test_loader, attacker, "Adversarial")

    print("\n" + "=" * 70)
    print("3. POISONING ATTACK ROBUSTNESS")
    print("=" * 70)
    _ = evaluate_poisoning(baseline_model, test_loader, attacker, "Baseline")
    _ = evaluate_poisoning(adv_model,      test_loader, attacker, "Adversarial")

    print("\n" + "=" * 70)
    print("4. STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    ttest = statistical_significance_test(
        baseline_clean['probs'], adv_clean['probs'], baseline_clean['labels']
    )

    # Results table
    print_results_table(baseline_clean, baseline_evasion,
                        adv_clean, adv_evasion, ttest)

    # Save all results
    all_results = {
        'baseline_clean':    baseline_clean,
        'baseline_evasion':  baseline_evasion,
        'adv_clean':         adv_clean,
        'adv_evasion':       adv_evasion,
        'ttest':             ttest,
    }
    with open(f"{EVAL_DIR}/all_results.pkl", 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n[DONE] All evaluation results saved to {EVAL_DIR}/")
    return all_results


if __name__ == "__main__":
    run_evaluation()
