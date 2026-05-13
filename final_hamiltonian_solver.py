
import numpy as np
import warnings

# Use Agg backend to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# PARAMETERS (Extracted directly from revised.py)
# ==============================================================================

PARAMS_H1 = {
    'gamma_1': 0.1,
    'W': 1.5,
    'delta_init': 0.1,
    'ħ'  : 1.0,
    'ε0' : 0.1,
    'ε1' : 0.1,
    'g1' : 0.1,
    'E1' : 1
}

PARAMS_H2 = {
    'epsilon_0': 0.1,
    'epsilon_2': 0.1,
    'gamma_2': 0.1,
    'g_2': 0.1,
    'E_2': 1,
    'delta_2_init': 0.1,
    'delta_init': 0.095
}

PARAMS_EQ29 = {
    'epsilon_0': 1.0,
    'epsilon_1': 1.0,
    'gamma_1': 0.1,
    'g_1': 0.15,
    'h': 1.0,
    'E_1': 0.5 + 0.3j,
    'eta_r': 1.0,
    'field_init': 0.1
}

# ==============================================================================
# HELPER FUNCTIONS FOR PRINTING
# ==============================================================================

def print_divider(char="-"):
    print(char * 80)

def print_full_matrix(name, matrix):
    print(f"\n{name} MATRIX (Shape: {matrix.shape}):")
    print_divider()
    with np.printoptions(threshold=np.inf, linewidth=200, precision=4, suppress=True):
        print(matrix)
    print_divider()

def print_all_eigenvalues(name, evals):
    print(f"\n{name} EIGENVALUES (Count: {len(evals)}):")
    print_divider()
    for i, e in enumerate(evals):
        print(f"  E_{i:<3} = {e.real:.10f}")
    print_divider()


# ==============================================================================
# SOLVER 1: QuantumHamiltonianSolver (Based on Hamiltonian Solver.py structure)
# Implementing H1 physics from revised.py
# ==============================================================================

class QuantumHamiltonianSolver:
    """
    Corresponds to 'System 1: H1' in revised.py.
    This class matches the structure of the solver in 'Hamiltonian Solver.py'
    but uses the exact physics/equations from 'revised.py'.
    """
    
    def __init__(self, params):
        self.params = params
        self.n_max = 1
        self.dim = 4
        self.a, self.a_dag, self.b, self.b_dag = self._create_operators()
        self.ab = self.a @ self.b
        self.a_dag_b_dag = self.a_dag @ self.b_dag

    def _create_operators(self):
        a_single = np.zeros((2, 2), dtype=complex)
        a_single[0, 1] = 1.0

        a_dag_single = a_single.conj().T
        I = np.eye(2, dtype=complex)

        a = np.kron(a_single, I)
        a_dag = np.kron(a_dag_single, I)

        b = np.kron(I, a_single)
        b_dag = np.kron(I, a_dag_single)

        return a, a_dag, b, b_dag

    def construct_hamiltonian(self, delta):
        # Using exact variable names and equations from revised.py's SelfConsistentH1
        W = self.params['W']
        gamma_1 = self.params['gamma_1']
        hbar = self.params['ħ']
        epsilon0 = self.params['ε0']
        epsilon1 = self.params['ε1']
        g1_param = self.params['g1']
        E1_param = self.params['E1']

        n_a = self.a_dag @ self.a
        n_b = self.b_dag @ self.b

        # Corrected Hamiltonian expression from revised.py
        H = (epsilon0 / (2 * 1j * hbar) + gamma_1 / 2) * n_a + \
            (epsilon1 / (1j * hbar) - gamma_1 / 2) * n_b + \
            (g1_param * E1_param / (1j * hbar) - gamma_1 * delta) *  \
            (self.a_dag @ self.b + self.b_dag @ self.a)

        return H

    def solve(self, max_iter=100, tol=1e-6, mixing=0.3):
        print_divider()
        print("SOLVER 1: H1 (QuantumHamiltonianSolver)")
        print_divider()

        delta = self.params['delta_init']
        
        for iteration in range(max_iter):
            H = self.construct_hamiltonian(delta)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            psi_0 = eigenvectors[:, 0]

            delta_computed = np.vdot(psi_0, self.ab @ psi_0).real
            delta_new = mixing * delta_computed + (1 - mixing) * delta
            
            error = abs(delta_new - delta)

            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: Delta={delta_new:.10f}, Error={error:.2e}")

            if error < tol and iteration > 5:
                print(f"\n[OK] Self-consistency achieved at iteration {iteration+1}")
                print(f"  Converged Delta = {delta_new:.10f}")
                break
            
            delta = delta_new

        H_final = self.construct_hamiltonian(delta_new)
        eigenvalues_final, _ = np.linalg.eigh(H_final)
        
        return H_final, eigenvalues_final


# ==============================================================================
# SOLVER 2: H2Solver (Based on h1_h2_solver.py structure)
# Implementing H2 physics from revised.py
# ==============================================================================

class H2Solver:
    """
    Corresponds to 'System 2: H2' in revised.py.
    This class matches the structure of the solver in 'h1_h2_solver.py'
    but uses the exact physics/equations from 'revised.py'.
    """

    def __init__(self, params):
        self.params = params
        self.n_max = 1
        self.dim = 4
        self.a, self.a_dag, self.c, self.c_dag = self._create_operators()
        self.ac = self.a @ self.c
        self.a_dag_c_dag = self.a_dag @ self.c_dag

    def _create_operators(self):
        a_single = np.zeros((2, 2), dtype=complex)
        a_single[0, 1] = 1.0

        a_dag_single = a_single.conj().T
        I = np.eye(2, dtype=complex)

        # Mode a
        a = np.kron(a_single, I)
        a_dag = np.kron(a_dag_single, I)

        # Mode c
        c = np.kron(I, a_single)
        c_dag = np.kron(I, a_dag_single)

        return a, a_dag, c, c_dag

    def construct_hamiltonian(self, delta_2):
        # Using exact variable names and equations from revised.py's SelfConsistentH2
        eps_0 = self.params['epsilon_0']
        eps_2 = self.params['epsilon_2']
        gamma_2 = self.params['gamma_2']
        g_2 = self.params['g_2']
        E_2 = self.params['E_2']
        delta = self.params['delta_init']
        hbar = 1.0

        n_a = self.a_dag @ self.a
        n_c = self.c_dag @ self.c
        a_dag_c = self.a_dag @ self.c
        c_dag_a = self.c_dag @ self.a

        # EXACT coefficients from PDF Equation 60
        coeff_a = (eps_0 / (2j * hbar) + gamma_2 / 2)
        coeff_c = (eps_2 / (1j * hbar) - gamma_2 / 2)
        coeff_coupling = (g_2 * E_2 / (1j * hbar) - gamma_2 * delta_2 )
        
        H = (coeff_a * n_a + coeff_c * n_c + \
             coeff_coupling * (a_dag_c + c_dag_a))

        return H.real

    def solve(self, max_iter=100, tol=1e-6, mixing=0.3):
        print_divider()
        print("SOLVER 2: H2 (H2Solver)")
        print_divider()

        delta_2 = self.params['delta_2_init']

        for iteration in range(max_iter):
            H = self.construct_hamiltonian(delta_2)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            psi_0 = eigenvectors[:, 0]

            delta_2_computed = np.vdot(psi_0, self.ac @ psi_0).real
            delta_2_new = mixing * delta_2_computed + (1 - mixing) * delta_2

            error = abs(delta_2_new - delta_2)
            
            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: Delta2={delta_2_new:.10f}, Error={error:.2e}")

            if error < tol and iteration > 5:
                print(f"\n[OK] Self-consistency achieved at iteration {iteration+1}")
                print(f"  Converged Delta2 = {delta_2_new:.10f}")
                break

            delta_2 = delta_2_new

        H_final = self.construct_hamiltonian(delta_2_new)
        eigenvalues_final, _ = np.linalg.eigh(H_final)

        return H_final, eigenvalues_final


# ==============================================================================
# SOLVER 3: Eq29Solver (From revised.py - FullEquation29)
# ==============================================================================

class Eq29Solver:
    """
    Corresponds to 'FullEquation29' in revised.py.
    This class strictly follows the implementation in 'revised.py'.
    """

    def __init__(self, params):
        self.params = params
        self.dim = 16  # 2^4
        self._create_pauli_matrices()

    def _create_pauli_matrices(self):
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        # s0
        self.s0x = np.kron(sx, np.kron(I, np.kron(I, I)))
        self.s0y = np.kron(sy, np.kron(I, np.kron(I, I)))
        self.s0z = np.kron(sz, np.kron(I, np.kron(I, I)))

        # s1
        self.s1x = np.kron(I, np.kron(sx, np.kron(I, I)))
        self.s1y = np.kron(I, np.kron(sy, np.kron(I, I)))
        self.s1z = np.kron(I, np.kron(sz, np.kron(I, I)))

        # s2
        self.s2x = np.kron(I, np.kron(I, np.kron(sx, I)))
        self.s2y = np.kron(I, np.kron(I, np.kron(sy, I)))
        self.s2z = np.kron(I, np.kron(I, np.kron(sz, I)))

        # s3
        self.s3x = np.kron(I, np.kron(I, np.kron(I, sx)))
        self.s3y = np.kron(I, np.kron(I, np.kron(I, sy)))
        self.s3z = np.kron(I, np.kron(I, np.kron(I, sz)))

    def construct_full_hamiltonian(self, field):
        # EXACT logic from revised.py FullEquation29.construct_full_hamiltonian
        eps_0 = self.params['epsilon_0']
        eps_1 = self.params['epsilon_1']
        g_1 = self.params['g_1']
        h = self.params['h']
        E_1 = self.params['E_1']
        gamma_1 = self.params['gamma_1']
        eta_r = self.params['eta_r']

        # ENERGY TERMS
        H_energy = (-eps_0/2 * (self.s0z - self.s2z) -
                   eps_1/2 * (self.s1z @ self.s0z + self.s3z @ self.s2z @ self.s1z))

        # COUPLING TERMS WITH E1
        coupling_E1 = (1j*g_1*h*field/4 * (
            self.s0x - 1j*self.s0y - self.s2y + 1j*self.s2x +
            1j*self.s1z @ self.s0y - self.s1z @ self.s0x -
            1j*self.s3z @ self.s2x @ self.s1z + self.s3z @ self.s2y @ self.s1z
        ) * E_1)

        # COUPLING TERMS WITH E1*
        coupling_E1_conj = (1j*g_1*h*field/4 * (
            self.s2x - 1j*self.s2y + 1j*self.s0x - self.s0y -
            self.s1y @ self.s0x - 1j*self.s1y @ self.s0y +
            1j*self.s3y @ self.s2y @ self.s1x - self.s3z @ self.s2y @ self.s1x
        ) * np.conj(E_1))

        # DISSIPATION TERMS WITH eta_r
        diss_eta = (gamma_1/16 * eta_r * (
            (4 - 8j) * np.eye(16, dtype=complex) +
            5*self.s0z - 4*(1+1j)*self.s1z + 4j*self.s1x + 3*self.s1y +
            4j*self.s2z - (5+3j)*self.s1z @ self.s0z + self.s1z @ self.s0z -
            3j*self.s1x @ self.s0z + 2j*self.s2z @ self.s0z +
            2j*self.s2y @ self.s0y + 4j*self.s3z @ self.s1z -
            self.s2y @ self.s1y @ self.s0x - 1j*self.s2y @ self.s1y @ self.s0y +
            1j*self.s2y @ self.s1y @ self.s0z - self.s2y @ self.s1x @ self.s0y -
            1j*self.s2y @ self.s1z @ self.s0x - 1j*self.s2x @ self.s1z @ self.s0y +
            self.s2x @ self.s1z @ self.s0x - self.s2x @ self.s1y @ self.s0y +
            self.s3z @ self.s2y @ self.s0y - self.s3z @ self.s2x @ self.s0x +
            1j*self.s3z @ self.s2z @ self.s0y - self.s3z @ self.s3x @ self.s0x -
            2j*self.s3z @ self.s2y @ self.s0x - 1j*self.s3z @ self.s2x @ self.s0y -
            4j*self.s3z @ self.s2y @ self.s1z +
            (1-3j)*self.s3z @ self.s2y @ self.s1z @ self.s0x -
            1j*self.s3z @ self.s2y @ self.s1x @ self.s0x +
            self.s3z @ self.s2y @ self.s1x @ self.s0y +
            self.s3z @ self.s2x @ self.s1z @ self.s0x -
            1j*self.s3z @ self.s2x @ self.s1z @ self.s0y
        ))

        # DISSIPATION TERMS WITHOUT eta_r
        diss_no_eta = (gamma_1/16 * (
            (4 - 2j) * np.eye(16, dtype=complex) +
            (4+2j)*self.s0z - (4-2j)*self.s1z + 2j*self.s1x + self.s1y -
            (5+1j)*self.s1z @ self.s0z - 2j*self.s1x @ self.s0z - self.s1y @ self.s0z -
            2*self.s2y @ self.s0y + 2*self.s2x @ self.s0x -
            self.s2y @ self.s1y @ self.s0x - 1j*self.s2y @ self.s1y @ self.s0y +
            1j*self.s2y @ self.s1x @ self.s0x - self.s2x @ self.s1y @ self.s0y +
            self.s2y @ self.s1z @ self.s0x + 1j*self.s2y @ self.s1z @ self.s0x +
            self.s2x @ self.s1z @ self.s0y - self.s2x @ self.s1z @ self.s0x -
            self.s3z @ self.s2y @ self.s0y - 1j*self.s3z @ self.s2y @ self.s0x -
            1j*self.s3z @ self.s2x @ self.s0y + self.s3z @ self.s2x @ self.s0x -
            2*self.s3z @ self.s2x @ self.s1z @ self.s0x -
            self.s3z @ self.s2y @ self.s1x @ self.s0x -
            1j*self.s3z @ self.s2x @ self.s1x @ self.s0y +
            (1-1j)*self.s3z @ self.s2y @ self.s1z @ self.s0x +
            2*self.s3z @ self.s2y @ self.s1z @ self.s0y -
            1j*self.s3z @ self.s2y @ self.s1x @ self.s0x +
            self.s3z @ self.s2y @ self.s1x @ self.s0y
        ))

        H_full = (H_energy + coupling_E1 + coupling_E1_conj +
                 diss_eta + diss_no_eta)

        return H_full.real

    def solve(self, max_iter=50, tol=1e-5, mixing=0.5):
        print_divider()
        print("SOLVER 3: EQUATION 29 (Eq29Solver)")
        print_divider()

        field = self.params['field_init']

        for iteration in range(max_iter):
            H = self.construct_full_hamiltonian(field)
            eigenvalues, eigenvectors = np.linalg.eigh(H)
            psi_0 = eigenvectors[:, 0]

            sx_exp = np.vdot(psi_0, self.s0x @ psi_0).real
            sy_exp = np.vdot(psi_0, self.s0y @ psi_0).real
            field_computed = np.sqrt(sx_exp**2 + sy_exp**2)

            field_new = mixing * field_computed + (1 - mixing) * field
            error = abs(field_new - field)

            if iteration % 5 == 0:
                print(f"  Iter {iteration:3d}: Field={field_new:.10f}, Error={error:.2e}")

            if error < tol and iteration > 3:
                print(f"\n[OK] Self-consistency achieved at iteration {iteration+1}")
                print(f"  Converged field = {field_new:.10f}")
                break

            field = field_new

        H_final = self.construct_full_hamiltonian(field_new)
        eigenvalues_final, _ = np.linalg.eigh(H_final)

        return H_final, eigenvalues_final


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    print_divider("=")
    print("STRICT SOLVER EXECUTION")
    print("Using solvers based on Hamiltonian Solver.py and h1_h2_solver.py")
    print("Using exact physics from h1,_h2_and_h3___revised_.py")
    print_divider("=")

    # 1. H1 (QuantumHamiltonianSolver)
    solver1 = QuantumHamiltonianSolver(PARAMS_H1)
    H1_matrix, H1_evals = solver1.solve()
    print_full_matrix("H1", H1_matrix)
    print_all_eigenvalues("H1", H1_evals)

    # 2. H2 (H2Solver)
    solver2 = H2Solver(PARAMS_H2)
    H2_matrix, H2_evals = solver2.solve()
    print_full_matrix("H2", H2_matrix)
    print_all_eigenvalues("H2", H2_evals)

    # 3. Eq29 (Eq29Solver)
    solver3 = Eq29Solver(PARAMS_EQ29)
    H29_matrix, H29_evals = solver3.solve()
    print_full_matrix("Eq29", H29_matrix)
    print_all_eigenvalues("Eq29", H29_evals)
    
    print_divider("=")
    print("[OK] Execution Complete")
    print_divider("=")

if __name__ == "__main__":
    main()
