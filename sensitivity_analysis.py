"""
=============================================================================
SENSITIVITY ANALYSIS (Hamiltonian inputs, +/-20%)
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Transactions

Implements the sensitivity-analysis requirement from the new-additions doc:
  - Vary epsilon_0, gamma_1, g_1 by +/-20% about their base values and
    record the resulting epsilon (attacker budget, derived from the H_2
    spectral gap) and lambda (defender L2 regularisation, derived from the
    H_1 ground-state energy).
  - Verify that the game still converges for every perturbation and that
    the derived parameters do not move catastrophically -> robustness to
    hyperparameter mis-specification.

All numbers are produced by the pure-NumPy self-consistent solvers and are
fully reproducible (no randomness).

USAGE
  python sensitivity_analysis.py
INPUTS
  Requires final_hamiltonian_solver.py on the import path.
OUTPUTS
  evaluation_results/sensitivity_analysis.json
=============================================================================
"""
import numpy as np, json, os, copy
from final_hamiltonian_solver import (QuantumHamiltonianSolver, H2Solver,
                                      Eq29Solver, PARAMS_H1, PARAMS_H2,
                                      PARAMS_EQ29)

EVAL_DIR = "evaluation_results"
os.makedirs(EVAL_DIR, exist_ok=True)


# ---- solver wrappers that ALSO return iteration count + ground state -----
def solve_H1(params, max_iter=100, tol=1e-6, mixing=0.3):
    s = QuantumHamiltonianSolver(params)
    delta = params['delta_init']; iters = 0
    for it in range(max_iter):
        H = s.construct_hamiltonian(delta)
        ev, evec = np.linalg.eigh(H)
        psi0 = evec[:, 0]
        dc = np.vdot(psi0, s.ab @ psi0).real
        dn = mixing*dc + (1-mixing)*delta
        iters = it + 1
        if abs(dn - delta) < tol and it > 5:
            break
        delta = dn
    evals = np.linalg.eigh(s.construct_hamiltonian(delta))[0]
    return evals, iters, float(evals[0])


def solve_H2(params, max_iter=100, tol=1e-6, mixing=0.3):
    s = H2Solver(params)
    d2 = params['delta_2_init']; iters = 0
    for it in range(max_iter):
        H = s.construct_hamiltonian(d2)
        ev, evec = np.linalg.eigh(H)
        psi0 = evec[:, 0]
        dc = np.vdot(psi0, s.ac @ psi0).real
        dn = mixing*dc + (1-mixing)*d2
        iters = it + 1
        if abs(dn - d2) < tol and it > 5:
            break
        d2 = dn
    evals = np.linalg.eigh(s.construct_hamiltonian(d2))[0]
    gap = float(evals[-1] - evals[0])
    return evals, iters, gap


def solve_Eq29(params, max_iter=50, tol=1e-5, mixing=0.5):
    s = Eq29Solver(params)
    field = params['field_init']; iters = 0
    for it in range(max_iter):
        H = s.construct_full_hamiltonian(field)
        ev, evec = np.linalg.eigh(H)
        psi0 = evec[:, 0]
        sx = np.vdot(psi0, s.s0x @ psi0).real
        sy = np.vdot(psi0, s.s0y @ psi0).real
        fn = mixing*np.sqrt(sx**2 + sy**2) + (1-mixing)*field
        iters = it + 1
        if abs(fn - field) < tol and it > 3:
            break
        field = fn
    return float(field), iters


# ---- parameter -> game-quantity maps (same clips as the paper / code) ----
def eps_from_gap(gap):   return float(np.clip(gap * 0.05, 0.01, 0.15))
def lambda_from_E0(E0):  return float(np.clip(abs(E0) * 1e-3, 1e-4, 1e-2))


def baseline_point():
    _, itH1, E0  = solve_H1(PARAMS_H1)
    _, itH2, gap = solve_H2(PARAMS_H2)
    phi, itEq    = solve_Eq29(PARAMS_EQ29)
    return dict(E0=E0, gap=gap, phi=phi,
                eps=eps_from_gap(gap), lam=lambda_from_E0(E0),
                iH1=itH1, iH2=itH2, iEq=itEq)


def sensitivity_analysis():
    print("=" * 70)
    print("SENSITIVITY ANALYSIS  --  +/-20% on epsilon_0, gamma_1, g_1")
    print("=" * 70)
    base = baseline_point()
    print(f"  BASELINE: gap(H2)={base['gap']:.6f}  E0(H1)={base['E0']:.6f}  "
          f"phi={base['phi']:.6f}")
    print(f"            eps={base['eps']:.6f}  lambda={base['lam']:.6f}  "
          f"(iters H1/H2/Eq29 = {base['iH1']}/{base['iH2']}/{base['iEq']})")
    print("-" * 70)

    factors = {'-20%': 0.8, 'base': 1.0, '+20%': 1.2}
    rows = []
    targets = ['epsilon_0', 'gamma_1', 'g1']
    for tname in targets:
        for fname, f in factors.items():
            p1 = copy.deepcopy(PARAMS_H1)
            p2 = copy.deepcopy(PARAMS_H2)
            pe = copy.deepcopy(PARAMS_EQ29)
            if tname == 'epsilon_0':
                p1['ε0'] *= f; p1['ε1'] *= f
                p2['epsilon_0'] *= f; p2['epsilon_2'] *= f
                pe['epsilon_0'] *= f; pe['epsilon_1'] *= f
            elif tname == 'gamma_1':
                p1['gamma_1'] *= f
                p2['gamma_2'] *= f          # analogous damping term in H2
                pe['gamma_1'] *= f
            elif tname == 'g1':
                p1['g1'] *= f
                p2['g_2'] *= f              # analogous coupling term in H2
                pe['g_1'] *= f

            _, iH1, E0  = solve_H1(p1)
            _, iH2, gap = solve_H2(p2)
            phi, iEq    = solve_Eq29(pe)
            eps = eps_from_gap(gap); lam = lambda_from_E0(E0)
            converged = (iH1 < 100) and (iH2 < 100) and (iEq < 50)
            d_eps = (eps - base['eps']) / base['eps'] * 100 if base['eps'] else 0.0
            d_lam = (lam - base['lam']) / base['lam'] * 100 if base['lam'] else 0.0
            rows.append(dict(param=tname, factor=fname,
                             E0=E0, gap=gap, phi=phi, eps=eps, lam=lam,
                             d_eps_pct=d_eps, d_lam_pct=d_lam,
                             iH1=iH1, iH2=iH2, iEq=iEq, converged=converged))
            print(f"  {tname:<10} {fname:>5} | gap={gap:.5f} E0={E0:+.5f} "
                  f"phi={phi:.5f} | eps={eps:.5f} lam={lam:.6f} "
                  f"| dEps={d_eps:+6.1f}% dLam={d_lam:+6.1f}% "
                  f"| conv={'Y' if converged else 'N'} "
                  f"({iH1}/{iH2}/{iEq})")
        print("-" * 70)

    eps_all = [r['eps'] for r in rows]
    lam_all = [r['lam'] for r in rows]
    all_conv = all(r['converged'] for r in rows)
    print(f"  eps range over all perturbations : "
          f"[{min(eps_all):.5f}, {max(eps_all):.5f}]  (clip band 0.01-0.15)")
    print(f"  lambda range over all perturbations: "
          f"[{min(lam_all):.6f}, {max(lam_all):.6f}]  (clip band 1e-4-1e-2)")
    print(f"  ALL configurations converged: {all_conv}")
    print(f"  => game is ROBUST to +/-20% hyperparameter mis-specification."
          if all_conv else "  => WARNING: a configuration failed to converge.")

    out = dict(baseline=base, rows=rows,
               eps_min=min(eps_all), eps_max=max(eps_all),
               lam_min=min(lam_all), lam_max=max(lam_all),
               all_converged=all_conv)
    with open(f"{EVAL_DIR}/sensitivity_analysis.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[DONE] Sensitivity results saved to "
          f"{EVAL_DIR}/sensitivity_analysis.json")
    return out


if __name__ == "__main__":
    sensitivity_analysis()
