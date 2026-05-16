"""
=============================================================================
HAMILTONIAN HYPERPARAMETER SEARCH
=============================================================================
Deep Learning-Based Fraud Detection in Banking and UPI Transactions

Implements the hyperparameter-search requirement from the new-additions doc:
  - Grid search over the core Hamiltonian solver parameters:
        epsilon_0 in {0.05, 0.1, 0.2}
        gamma     in {0.05, 0.1, 0.2}
        g         in {0.05, 0.1, 0.15, 0.2}
  - For each configuration we solve all three Hamiltonians (H_1, H_2, H_29)
    self-consistently, record the iterations to convergence, and compute the
    derived game parameters epsilon (attacker budget) and lambda (defender
    regularisation).
  - Selection score balances "useful attack budget" against "small but
    non-trivial regularisation":   score = |eps - 0.05| + 10*|lambda - 1e-3|
  - The best converged operating point is reported.

All numbers are produced by the pure-NumPy self-consistent solvers and are
fully reproducible (no randomness).

USAGE
  python hyperparameter_search.py
INPUTS
  Requires final_hamiltonian_solver.py on the import path.
OUTPUTS
  evaluation_results/hyperparameter_search.json
=============================================================================
"""
import numpy as np, json, itertools, os, copy
from final_hamiltonian_solver import (QuantumHamiltonianSolver, H2Solver,
                                      Eq29Solver, PARAMS_H1, PARAMS_H2,
                                      PARAMS_EQ29)

EVAL_DIR = "evaluation_results"
os.makedirs(EVAL_DIR, exist_ok=True)


# ---- solver wrappers (identical to those used in sensitivity_analysis) ---
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


def eps_from_gap(gap):   return float(np.clip(gap * 0.05, 0.01, 0.15))
def lambda_from_E0(E0):  return float(np.clip(abs(E0) * 1e-3, 1e-4, 1e-2))


def hyperparameter_search():
    print("=" * 70)
    print("HAMILTONIAN HYPERPARAMETER SEARCH "
          "(grid over epsilon_0, gamma, g)")
    print("=" * 70)
    grid_eps0  = [0.05, 0.1, 0.2]
    grid_gamma = [0.05, 0.1, 0.2]
    grid_g     = [0.05, 0.1, 0.15, 0.2]
    results = []
    best = None

    for e0, gm, gg in itertools.product(grid_eps0, grid_gamma, grid_g):
        p1 = copy.deepcopy(PARAMS_H1)
        p2 = copy.deepcopy(PARAMS_H2)
        pe = copy.deepcopy(PARAMS_EQ29)
        p1['ε0'] = e0; p1['ε1'] = e0; p1['gamma_1'] = gm; p1['g1'] = gg
        p2['epsilon_0'] = e0; p2['epsilon_2'] = e0
        p2['gamma_2'] = gm; p2['g_2'] = gg
        pe['epsilon_0'] = e0*10; pe['epsilon_1'] = e0*10
        pe['gamma_1'] = gm; pe['g_1'] = gg

        _, iH1, E0  = solve_H1(p1)
        _, iH2, gap = solve_H2(p2)
        phi, iEq    = solve_Eq29(pe)
        eps = eps_from_gap(gap); lam = lambda_from_E0(E0)
        conv = (iH1 < 100) and (iH2 < 100) and (iEq < 50)
        score = abs(eps - 0.05) + abs(lam - 1e-3) * 10
        rec = dict(eps_0=e0, gamma=gm, g=gg, E0=E0, gap=gap, phi=phi,
                   eps=eps, lam=lam, iH1=iH1, iH2=iH2, iEq=iEq,
                   converged=conv, score=score)
        results.append(rec)
        if conv and (best is None or score < best['score']):
            best = rec

    n_conv = sum(r['converged'] for r in results)
    print(f"  Evaluated {len(results)} configurations; "
          f"{n_conv}/{len(results)} converged.")
    if best is not None:
        print(f"  Best operating point (min selection score):")
        print(f"    eps_0={best['eps_0']}  gamma={best['gamma']}  g={best['g']}")
        print(f"    -> eps={best['eps']:.5f}  lambda={best['lam']:.6f}  "
              f"E0={best['E0']:+.5f}  gap={best['gap']:.5f}")
        print(f"    -> converged in H1/H2/Eq29 = "
              f"{best['iH1']}/{best['iH2']}/{best['iEq']} iterations")
    with open(f"{EVAL_DIR}/hyperparameter_search.json", 'w') as f:
        json.dump(dict(results=results, best=best, n_converged=n_conv,
                       n_total=len(results)), f, indent=2)
    print(f"\n[DONE] Hyperparameter-search results saved to "
          f"{EVAL_DIR}/hyperparameter_search.json")
    return results, best


if __name__ == "__main__":
    hyperparameter_search()
