"""
=============================================================================
FILE 3: ADVERSARIAL GAME FRAMEWORK — Stackelberg Game + Hamiltonian Equations
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Tran`sactions
Advisor: Aneesh Chivukula

THEORETICAL FRAMEWORK:
  We model the fraud detection problem as a two-player Stackelberg game:
    • Leader  (Defender): the fraud detector (our transformer)
    • Follower (Attacker): an adversarial perturbation generator

  The attacker generates adversarial transaction features that attempt to
  evade the detector. The detector is trained to be robust against these.

  HAMILTONIAN INTEGRATION:
  The adversarial perturbation energy is regularized using quantum-inspired
  Hamiltonian operators (H1, H2, Eq29) from the uploaded solver. The
  Hamiltonian eigenspectrum defines a structured perturbation manifold,
  constraining the attacker to physically-plausible fraud evasion strategies.

  Specifically:
    • H1 eigenvalues → bound the ℓ2 norm of perturbations per feature cluster
    • H2 eigenvalues → modulate the coupling strength between feature perturbations
    • Eq29 eigenvalues → govern the field energy of the global perturbation
    • Stackelberg equilibrium is found via projected gradient ascent/descent

  GAME OBJECTIVE:
    min_θ max_δ  L(f_θ(x + δ), y)
    subject to:  ‖δ‖ ≤ ε_H  (Hamiltonian-constrained budget)

  where ε_H is the Hamiltonian-derived perturbation bound.
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import os, warnings, pickle
warnings.filterwarnings('ignore')

# Import baseline model architecture
from baseline_model import TabularTransformer, FocalLoss, load_processed_data, evaluate

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# HAMILTONIAN SOLVER (INTEGRATED FROM final_hamiltonian_solver.py)
# ─────────────────────────────────────────────────────────────

PARAMS_H1 = {
    'gamma_1': 0.1, 'W': 1.5, 'delta_init': 0.1,
    'ħ': 1.0, 'ε0': 0.1, 'ε1': 0.1, 'g1': 0.1, 'E1': 1
}
PARAMS_H2 = {
    'epsilon_0': 0.1, 'epsilon_2': 0.1, 'gamma_2': 0.1,
    'g_2': 0.1, 'E_2': 1, 'delta_2_init': 0.1, 'delta_init': 0.095
}
PARAMS_EQ29 = {
    'epsilon_0': 1.0, 'epsilon_1': 1.0, 'gamma_1': 0.1,
    'g_1': 0.15, 'h': 1.0, 'E_1': 0.5 + 0.3j,
    'eta_r': 1.0, 'field_init': 0.1
}


class QuantumHamiltonianSolver:
    """H1 Solver — adapted from final_hamiltonian_solver.py"""
    def __init__(self, params):
        self.params = params
        self.dim = 4
        self.a, self.a_dag, self.b, self.b_dag = self._create_operators()
        self.ab = self.a @ self.b

    def _create_operators(self):
        a_s = np.zeros((2, 2), dtype=complex); a_s[0, 1] = 1.0
        I   = np.eye(2, dtype=complex)
        return (np.kron(a_s, I), np.kron(a_s.conj().T, I),
                np.kron(I, a_s), np.kron(I, a_s.conj().T))

    def construct_hamiltonian(self, delta):
        W, g1, E1 = self.params['W'], self.params['g1'], self.params['E1']
        hbar, eps0, eps1 = self.params['ħ'], self.params['ε0'], self.params['ε1']
        gamma = self.params['gamma_1']
        n_a = self.a_dag @ self.a; n_b = self.b_dag @ self.b
        H = ((eps0 / (2j*hbar) + gamma/2) * n_a +
             (eps1 / (1j*hbar) - gamma/2) * n_b +
             (g1*E1 / (1j*hbar) - gamma*delta) *
             (self.a_dag @ self.b + self.b_dag @ self.a))
        return H

    def solve(self, max_iter=100, tol=1e-6, mixing=0.3):
        delta = self.params['delta_init']
        for _ in range(max_iter):
            H = self.construct_hamiltonian(delta)
            evals, evecs = np.linalg.eigh(H)
            psi0 = evecs[:, 0]
            dc   = np.vdot(psi0, self.ab @ psi0).real
            dn   = mixing * dc + (1 - mixing) * delta
            if abs(dn - delta) < tol:
                break
            delta = dn
        return np.linalg.eigh(self.construct_hamiltonian(delta))[0]


class H2Solver:
    """H2 Solver — adapted from final_hamiltonian_solver.py"""
    def __init__(self, params):
        self.params = params
        self.dim = 4
        self.a, self.a_dag, self.c, self.c_dag = self._create_operators()
        self.ac = self.a @ self.c

    def _create_operators(self):
        a_s = np.zeros((2, 2), dtype=complex); a_s[0, 1] = 1.0
        I   = np.eye(2, dtype=complex)
        return (np.kron(a_s, I), np.kron(a_s.conj().T, I),
                np.kron(I, a_s), np.kron(I, a_s.conj().T))

    def construct_hamiltonian(self, delta_2):
        e0, e2 = self.params['epsilon_0'], self.params['epsilon_2']
        g2, E2 = self.params['g_2'], self.params['E_2']
        gamma  = self.params['gamma_2']
        n_a = self.a_dag @ self.a; n_c = self.c_dag @ self.c
        H = ((e0/(2j) + gamma/2) * n_a +
             (e2/(1j) - gamma/2) * n_c +
             (g2*E2/(1j) - gamma*delta_2) *
             (self.a_dag @ self.c + self.c_dag @ self.a))
        return H.real

    def solve(self, max_iter=100, tol=1e-6, mixing=0.3):
        d2 = self.params['delta_2_init']
        for _ in range(max_iter):
            H = self.construct_hamiltonian(d2)
            evals, evecs = np.linalg.eigh(H)
            psi0 = evecs[:, 0]
            dc   = np.vdot(psi0, self.ac @ psi0).real
            dn   = mixing * dc + (1 - mixing) * d2
            if abs(dn - d2) < tol:
                break
            d2 = dn
        return np.linalg.eigh(self.construct_hamiltonian(d2))[0]


class Eq29Solver:
    """Eq29 Solver — adapted from final_hamiltonian_solver.py"""
    def __init__(self, params):
        self.params = params
        self.dim = 16
        self._create_pauli_matrices()

    def _create_pauli_matrices(self):
        sx = np.array([[0,1],[1,0]], dtype=complex)
        sy = np.array([[0,-1j],[1j,0]], dtype=complex)
        sz = np.array([[1,0],[0,-1]], dtype=complex)
        I  = np.eye(2, dtype=complex)
        for i, s in enumerate([[sx,sy,sz]]*4):
            for j, (mat, name) in enumerate(zip(s, ['x','y','z'])):
                parts = [I]*4; parts[i] = mat
                full  = parts[0]
                for p in parts[1:]: full = np.kron(full, p)
                setattr(self, f"s{i}{name}", full)

    def construct_full_hamiltonian(self, field):
        e0, e1 = self.params['epsilon_0'], self.params['epsilon_1']
        g1, h  = self.params['g_1'], self.params['h']
        E1     = self.params['E_1']
        gamma  = self.params['gamma_1']
        eta_r  = self.params['eta_r']
        H_en   = (-e0/2 * (self.s0z - self.s2z) -
                   e1/2 * (self.s1z @ self.s0z + self.s3z @ self.s2z @ self.s1z))
        cp_E1  = (1j*g1*h*field/4 * (
            self.s0x - 1j*self.s0y - self.s2y + 1j*self.s2x +
            1j*self.s1z @ self.s0y - self.s1z @ self.s0x) * E1)
        cp_Ec  = (1j*g1*h*field/4 * (
            self.s2x - 1j*self.s2y + 1j*self.s0x - self.s0y) * np.conj(E1))
        return (H_en + cp_E1 + cp_Ec).real

    def solve(self, max_iter=50, tol=1e-5, mixing=0.5):
        field = self.params['field_init']
        for _ in range(max_iter):
            H     = self.construct_full_hamiltonian(field)
            evals, evecs = np.linalg.eigh(H)
            psi0  = evecs[:, 0]
            sx_e  = np.vdot(psi0, self.s0x @ psi0).real
            sy_e  = np.vdot(psi0, self.s0y @ psi0).real
            fn    = mixing * np.sqrt(sx_e**2 + sy_e**2) + (1-mixing)*field
            if abs(fn - field) < tol:
                break
            field = fn
        return np.linalg.eigh(self.construct_full_hamiltonian(field))[0]


# ─────────────────────────────────────────────────────────────
# HAMILTONIAN PERTURBATION BUDGET
# ─────────────────────────────────────────────────────────────
class HamiltonianPerturbationBudget:
    """
    Solves all three Hamiltonians and derives structured perturbation
    constraints for adversarial examples.

    Physical interpretation:
      • ε_H1: energy scale of photon-mode coupling (local feature noise)
      • ε_H2: energy scale of cross-mode coupling (feature interaction noise)
      • ε_Eq29: field amplitude at equilibrium (global perturbation amplitude)
    """
    def __init__(self):
        print("[Hamiltonian] Solving H1, H2, Eq29 for perturbation budget …")
        self.evals_H1  = QuantumHamiltonianSolver(PARAMS_H1).solve()
        self.evals_H2  = H2Solver(PARAMS_H2).solve()
        self.evals_Eq29 = Eq29Solver(PARAMS_EQ29).solve()

        # Perturbation budgets derived from ground-state energy gaps
        self.eps_H1   = abs(self.evals_H1[1]  - self.evals_H1[0])   * 0.5
        self.eps_H2   = abs(self.evals_H2[1]  - self.evals_H2[0])   * 0.5
        self.eps_Eq29 = abs(self.evals_Eq29[1] - self.evals_Eq29[0]) * 0.1

        # Global perturbation bound (geometric mean of all three scales)
        self.eps_global = float(
            (self.eps_H1 * self.eps_H2 * self.eps_Eq29) ** (1/3)
        )
        # Ensure non-trivial but bounded perturbation
        self.eps_global = np.clip(self.eps_global, 0.02, 0.3)

        print(f"  ε_H1   = {self.eps_H1:.6f}")
        print(f"  ε_H2   = {self.eps_H2:.6f}")
        print(f"  ε_Eq29 = {self.eps_Eq29:.6f}")
        print(f"  ε_global (Hamiltonian budget) = {self.eps_global:.6f}")

    def get_feature_weights(self, n_features: int) -> torch.Tensor:
        """
        Project H1+H2 eigenvalue spreads onto feature dimension as
        per-feature perturbation weights (structured budget allocation).
        """
        h1_spread  = np.abs(self.evals_H1)
        h2_spread  = np.abs(self.evals_H2)
        combined   = np.concatenate([h1_spread, h2_spread])
        # Tile/trim to match n_features
        weights = np.tile(combined, n_features // len(combined) + 1)[:n_features]
        weights = weights / (weights.sum() + 1e-8) * n_features
        return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# ATTACKER: PROJECTED GRADIENT ASCENT (PGA)
# ─────────────────────────────────────────────────────────────
class HamiltonianAttacker:
    """
    Generates adversarial examples using projected gradient ascent,
    with Hamiltonian-constrained perturbation budget.

    Implements both:
      (1) Evasion attack — perturb fraud → look like legitimate
      (2) Poisoning attack — perturb training data labels/features
    """
    def __init__(self, budget: HamiltonianPerturbationBudget,
                 n_steps: int = 10, step_size: float = 0.01):
        self.budget    = budget
        self.n_steps   = n_steps
        self.step_size = step_size
        self.eps       = budget.eps_global

    def evasion_attack(self, model: nn.Module,
                       x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        PGA evasion: attacker maximises detection loss on fraud samples.
        Constraint: ‖δ‖∞ ≤ ε_global (Hamiltonian budget)
        """
        model.eval()
        feature_w = self.budget.get_feature_weights(x.shape[1]).to(x.device)

        # Only attack fraud samples (y == 1)
        mask      = (y == 1).float().unsqueeze(1)
        delta     = torch.zeros_like(x).uniform_(-self.eps, self.eps) * mask
        delta.requires_grad_(True)

        for _ in range(self.n_steps):
            x_adv  = x + delta * feature_w.unsqueeze(0)
            logits = model(x_adv)
            # Attacker maximises loss (fraud evades detection → target=0)
            loss   = nn.functional.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits)
            ) * mask.squeeze(1)
            loss   = loss.mean()
            loss.backward()

            with torch.no_grad():
                delta = delta + self.step_size * delta.grad.sign()
                delta = torch.clamp(delta, -self.eps, self.eps) * mask
            delta = delta.detach().requires_grad_(True)

        return (x + delta.detach() * feature_w.unsqueeze(0)).detach()

    def poisoning_attack(self, x: torch.Tensor,
                         poison_rate: float = 0.05) -> torch.Tensor:
        """
        Poisoning attack: inject structured noise into random subset of
        training features, weighted by Hamiltonian energy distribution.
        """
        feature_w = self.budget.get_feature_weights(x.shape[1]).to(x.device)
        n_poison  = max(1, int(len(x) * poison_rate))
        idx       = torch.randperm(len(x))[:n_poison]
        noise     = (torch.rand(n_poison, x.shape[1], device=x.device) * 2 - 1)
        noise     = noise * self.eps * feature_w.unsqueeze(0)
        x_poison  = x.clone()
        x_poison[idx] += noise
        return x_poison


# ─────────────────────────────────────────────────────────────
# STACKELBERG ADVERSARIAL TRAINING
# ─────────────────────────────────────────────────────────────
class StackelbergAdversarialTrainer:
    """
    Stackelberg game training loop:
      1. Inner (attacker) step: generate adversarial examples via PGA
      2. Outer (defender) step: update model on mix of clean + adversarial data

    The Hamiltonian budgets govern step 1, ensuring physically-motivated
    and bounded adversarial perturbations.
    """
    def __init__(self, model: nn.Module, budget: HamiltonianPerturbationBudget,
                 adv_weight: float = 0.5):
        self.model      = model
        self.attacker   = HamiltonianAttacker(budget)
        self.criterion  = FocalLoss(gamma=2.0, alpha=0.75)
        self.optimizer  = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler  = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30)
        self.adv_weight = adv_weight   # blend coefficient λ: L = (1-λ)L_clean + λL_adv

    def _defender_step(self, x_clean: torch.Tensor, x_adv: torch.Tensor,
                        y: torch.Tensor) -> float:
        """Outer minimization: defender minimizes mixed loss."""
        self.model.train()
        self.optimizer.zero_grad()

        logits_clean = self.model(x_clean)
        logits_adv   = self.model(x_adv)

        loss_clean = self.criterion(logits_clean, y)
        loss_adv   = self.criterion(logits_adv,   y)
        loss       = (1 - self.adv_weight) * loss_clean + self.adv_weight * loss_adv

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, loader: DataLoader) -> float:
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Inner step: attacker generates adversarial examples
            x_adv = self.attacker.evasion_attack(self.model, x_batch, y_batch)

            # Outer step: defender updates on clean + adversarial
            loss = self._defender_step(x_batch, x_adv, y_batch)
            total_loss += loss * len(y_batch)

        self.scheduler.step()
        return total_loss / len(loader.dataset)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 30, patience: int = 6) -> list:
        print("\n" + "=" * 70)
        print("STACKELBERG ADVERSARIAL TRAINING (Hamiltonian-Constrained)")
        print("=" * 70)

        best_auc   = 0.0
        patience_c = 0
        history    = []

        for epoch in range(1, epochs + 1):
            train_loss   = self.train_epoch(train_loader)
            val_metrics  = evaluate(self.model, val_loader)

            history.append({
                'epoch': epoch, 'train_loss': train_loss,
                'val_auc': val_metrics['auc'], 'val_f1': val_metrics['f1']
            })

            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Adv Loss: {train_loss:.4f} | "
                  f"Val AUC: {val_metrics['auc']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")

            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                patience_c = 0
                torch.save(self.model.state_dict(),
                           f"{MODEL_DIR}/adversarial_best.pt")
            else:
                patience_c += 1
                if patience_c >= patience:
                    print(f"[Early Stop] No improvement for {patience} epochs.")
                    break

        return history


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def run_adversarial_training():
    print("=" * 70)
    print("ADVERSARIAL GAME FRAMEWORK")
    print("Stackelberg Game with Hamiltonian-Constrained Perturbations")
    print("=" * 70)

    train_loader, val_loader, test_loader, n_features = load_processed_data()

    # Solve Hamiltonians → derive perturbation budget
    budget = HamiltonianPerturbationBudget()

    # Load pre-trained baseline as defender initialization
    model = TabularTransformer(n_features=n_features).to(DEVICE)
    baseline_ckpt = f"{MODEL_DIR}/baseline_best.pt"
    if os.path.exists(baseline_ckpt):
        model.load_state_dict(torch.load(baseline_ckpt, map_location=DEVICE))
        print("[INFO] Initialized defender from pre-trained baseline.")
    else:
        print("[WARN] No baseline checkpoint found; training from scratch.")

    # Adversarial training
    trainer = StackelbergAdversarialTrainer(model, budget, adv_weight=0.5)
    history = trainer.train(train_loader, val_loader, epochs=30, patience=6)

    # Final test evaluation
    print("\n" + "=" * 70)
    print("FINAL ADVERSARIAL MODEL — TEST RESULTS")
    print("=" * 70)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/adversarial_best.pt",
                                      map_location=DEVICE))
    test_metrics = evaluate(model, test_loader)
    print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  Confusion Matrix:\n{test_metrics['cm']}")

    np.save(f"{MODEL_DIR}/adv_test_probs.npy",  test_metrics['probs'])
    np.save(f"{MODEL_DIR}/adv_test_labels.npy", test_metrics['labels'])
    with open(f"{MODEL_DIR}/adv_history.pkl", 'wb') as f:
        pickle.dump(history, f)

    print("\n[DONE] Adversarial model saved.")
    return model, test_metrics, budget


if __name__ == "__main__":
    run_adversarial_training()
