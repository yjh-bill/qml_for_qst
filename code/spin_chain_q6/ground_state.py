import numpy as np
from math import pi
from scipy.linalg import eigh
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

def getHamiltonian(L, delta=1., h = 1., J=1.):
    XX_terms = [("XX", [i, i+1], J) for i in range(L-1)]
    YY_terms = [("YY", [i, i+1], J) for i in range(L-1)]
    ZZ_terms = [("ZZ", [i, i+1], delta) for i in range(L-1)]
    single_terms = [("Z", [i], h) for i in range(L)]
    H_tot_terms = XX_terms + YY_terms + ZZ_terms + single_terms
    return SparsePauliOp.from_sparse_list(H_tot_terms, num_qubits=L)

def get_state_vector_list(n_qubit):
    return [Statevector.from_int(i, dims=2**n_qubit) for i in range(2**n_qubit)]

def get_state_from_coeffs(basis_list, coeff_list):
    state = basis_list[0] * coeff_list[0]
    for i in range(1, len(coeff_list)):
        state += basis_list[i] * coeff_list[i]
    return state


def get_ground_state(L, delta=1., h = 1., J=1.):
    h_op = getHamiltonian(L, delta=delta, h=h, J=J)
    h_op_mat = h_op.to_matrix()
    eig_vals, eig_vecs = eigh(h_op_mat)
    basis_list = get_state_vector_list(n_qubit=L)
    return get_state_from_coeffs(basis_list, eig_vecs[:, 0])

def state_measure_circ_at_basis(n_qubits, basis):
    circ = QuantumCircuit(n_qubits)
    for i, s in enumerate(basis[::-1]): # here, I am using the same endian convention as qiskit, where the last bit represent the first qubit
        if s == 'Z':
            pass
        elif s == 'X':
            circ.ry(-pi/2, i)
        elif s == 'Y':
            circ.rx(pi/2, i)
        else:
            raise ValueError(f"{s} is invalid basis")

    return circ

def measure_state_at_basis(state, n_qubits, basis, shots):
    measure_circ = state_measure_circ_at_basis(n_qubits, basis)
    state_evolved = state.evolve(measure_circ)
    return state_evolved.sample_counts(shots)

def measure_gs_basis_list(L, basis_list, shots, delta=1., h = 1., J=1., seed = None):
    gs_state = get_ground_state(L, delta=delta, h=h, J=J)
    gs_state.seed(seed)
    results_dic = {}

    for basis in basis_list:
        counts = measure_state_at_basis(gs_state, L, basis, shots=shots)
        results_dic[basis] = counts
    
    return results_dic