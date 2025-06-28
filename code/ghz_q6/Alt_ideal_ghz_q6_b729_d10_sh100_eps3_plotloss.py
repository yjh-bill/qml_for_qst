# %%
import os
from math import pi
import numpy as np
import random
import itertools
import scipy.optimize as opt
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from qiskit_aer.noise import NoiseModel
# from qiskit_ibm_provider import IBMProvider
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms import utils



# %%
shots = 100
epsilon_loss = 1e-3 # loss added if test_counts entry is not found
epsilon_parameters=0.1 # scaling factor for initial random parameters (chosen between 0 to 2pi otherwise)

n_qubits = 6
n_bases = 729
depth = 10

max_iter = 3000

param_name = "theta"

# save checkpoints in case the connection to qpu breaks
# useful for running on real qpu, but not necessary for local simulation
have_checkpoints = False
checkpoint_path = "checkpoints"

# seed = 1000

device = "CPU"
max_parallel_threads = 0

# %%
rng = np.random.default_rng()

# %%
# if have_checkpoints:
#     os.mkdir(checkpoint_path)

# %%
# backend = AerSimulator()
# sampler = Sampler.from_backend(backend)
# pm = generate_preset_pass_manager(backend=backend)

simulator_ideal = AerSimulator(device = device, max_parallel_threads=max_parallel_threads)
#simulator_ideal.set_options(seed_simulator=seed)
# sampler_ideal = Sampler.from_backend(simulator_ideal)
pm_ideal = generate_preset_pass_manager(backend=simulator_ideal, optimization_level=1)

# sampler = sampler_ideal
pm = pm_ideal

# %%
def create_ghz_state(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc

def get_circ_for_basis(qc: QuantumCircuit, basis: str):
    circ = qc.copy()
    for i, s in enumerate(basis[::-1]): # here, I am using the same endian convention as qiskit, where the last bit represent the first qubit
        if s == 'Z':
            pass
        elif s == 'X':
            circ.ry(-pi/2, i)
        elif s == 'Y':
            circ.rx(pi/2, i)
        else:
            raise ValueError(f"{s} is invalid basis")

    circ.measure_all()

    return circ

def get_isa_circ_list_for_bases(qc: QuantumCircuit, basis_list, pass_manager):
    circ_list = [get_circ_for_basis(qc, basis) for basis in basis_list]
    isa_circ_list = pass_manager.run(circ_list)
    return isa_circ_list


def measure_isa_circ_list_fs(isa_circ_list, basis_list, shots=shots):
    sampler = Sampler( )
    # sampler._backend.set_options(max_parallel_threads = max_parallel_threads)
    result_list = sampler.run(isa_circ_list, shots=shots).result()
    counts_list = [result.data.meas.get_counts() for result in result_list]
    results_dic = {basis:counts for (basis, counts) in zip(basis_list, counts_list)}
    return results_dic


def get_basis_list(n_bases,n_qubits):
    bases = ["X", "Y", "Z"]
    basis_list = ["".join(p) for p in itertools.product(bases, repeat=n_qubits)]
    culled_basis_list = random.sample(basis_list, n_bases)
    return culled_basis_list




# def measure_circ_for_all_basis(qc, basis_list, shots=total_shots, sampler=None):
#     results_dic = {}
#     qc_list = [get_circ_for_basis(qc, basis, simulator) for basis in basis_list]
#     job = simulator.run(qc_list, shots=shots)
#     result = job.result()
#     counts_list = result.get_counts()
#     results_dic = {basis:counts for (basis, counts) in zip(basis_list, counts_list)}
    
#     return results_dic

# %%
basis_list = get_basis_list(n_bases, n_qubits)
ghz_circ = create_ghz_state(n_qubits)
ghz_circ_isa_list = get_isa_circ_list_for_bases(ghz_circ, basis_list, pm_ideal)
GHZ_measurement = measure_isa_circ_list_fs(ghz_circ_isa_list, basis_list, shots=shots)

# %%
print(basis_list)

# %%
print(GHZ_measurement)

# %%
def initialize_theta_random(circ_depth=10, num_qbits=3):
    """
    Initialize the theta parameter vector
    :param circ_depth: int, number of parameterized layers in circuit
    :param num_qbits: int, number of qbits
    :return: np.array, values of theta
    """

    theta = rng.random(size=(circ_depth, num_qbits))
    theta = 2*pi*epsilon_parameters*theta
    return theta

# %%
def construct_variational_circ_ansatz(num_qbits, circ_depth, param_name = param_name):
    """
    Generate a parameterized variational quantum circuit
    
    """

    num_parameters = num_qbits * circ_depth
    theta_vec = ParameterVector(param_name, length=num_parameters)
    var_circ = QuantumCircuit(num_qbits)


    for layer in range(circ_depth - 1):
        # Compute if layer is odd or even to apply rx or ry gate to circuit
        is_odd_step = (layer + 1) % 2

        for qbit in range(num_qbits):
            if is_odd_step:
                var_circ.rx(theta_vec[layer * num_qbits + qbit], qbit)
            else:
                var_circ.ry(theta_vec[layer * num_qbits + qbit], qbit)

        # Apply CX gates
        for qbit in range((1-is_odd_step), num_qbits-1, 2):
            # isOddStep may subtract 1 if True, to correctly apply cx gate location
            var_circ.cx(qbit , qbit + 1)
            var_circ.barrier() # for visualization only

    for qbit in range(num_qbits):  # bonus layer at the end only has rx gates and no cx
        var_circ.rx(theta_vec[(circ_depth - 1) * num_qbits + qbit], qbit)

    return var_circ, theta_vec


# %%
ansatz, param_vec = construct_variational_circ_ansatz(n_qubits, depth, param_name=param_name)

# %%
ansatz.draw(output="mpl", style="clifford")

# %%
ansatz_isa_list = get_isa_circ_list_for_bases(ansatz, basis_list, pm)

# %%
def get_ansatz_output(ansatz_isa_list, parameters, basis_list, shots, param_placeholder=param_vec):
    circ_run_list = [circ.assign_parameters({param_placeholder:parameters}) for circ in ansatz_isa_list]
    return measure_isa_circ_list_fs(isa_circ_list=circ_run_list, basis_list=basis_list, shots=shots)

# %%
def KL_divergence(true_data, test_data, epsilon_loss = epsilon_loss):
    loss = 0

    # here, let me assume true data and test data can have different number of shots
    # but the shot in each basis should be the same
    true_data_total_shots = sum(list(true_data.values())[0].values())
    test_data_total_shots = sum(list(test_data.values())[0].values())

    for basis in true_data: # just a note that if there are hallucinated measurements in 
        # bases/states not included in the true data then the kl divergence will not account for these (at least as presented in the paper)
        for state in true_data[basis]:
            true_counts = true_data[basis][state]
            test_counts = test_data[basis].get(state, 0)
            true_prob   = true_counts / true_data_total_shots
            test_prob   = test_counts / test_data_total_shots
            loss += true_prob *(np.log((true_prob)/(test_prob +epsilon_loss)))

    return loss / len(true_data)

# %%
global loss_values
global thetas

loss_values = []
thetas = []

# %%
def compute_kl_loss(theta_vector, ansatz_isa_list,  basis_list, true_data, shots, param_placeholder=param_vec, \
        print_message=True, have_checkpoints=have_checkpoints, checkpoint_path=checkpoint_path):
    # theta = np.reshape(theta_vector, (circ_depth, num_qbits))
    # tempcirc = construct_variational_circ(theta=theta)
    test_data = get_ansatz_output(ansatz_isa_list=ansatz_isa_list, parameters=theta_vector, basis_list=basis_list, \
        shots=shots, param_placeholder=param_placeholder)
    # test_data = measure_circ_for_all_basis(qc = tempcirc, basis_list=basis_list, simulator=simulator)
    loss = KL_divergence(true_data, test_data) + \
        KL_divergence(test_data, true_data) # symmetrizing the loss

    global loss_values
    global thetas 
    loss_values.append(loss)
    thetas.append(theta_vector)

    num_iter = len(loss_values)

    if print_message:
        print(f"Number {num_iter:5}; current loss is {loss}")

    if have_checkpoints:
        # save checkpoints in case the connection to qpu breaks
        # useful for running on real qpu, but not necessary for local simulation
        np.save(os.path.join(checkpoint_path, f"theta_iter{num_iter}.npy"), thetas)
    
    return loss

def compute_kl_loss_filled(theta_vector):
    return compute_kl_loss(theta_vector, ansatz_isa_list, basis_list, GHZ_measurement, shots, param_vec, print_message=True)

def compute_kl_loss_filled_quiet(theta_vector):
    return compute_kl_loss(theta_vector, ansatz_isa_list, basis_list, GHZ_measurement, shots, param_vec, print_message=False)


loss_proper_values = [] # the loss function L(theta)
# by contrast, if we do not use the callback, we get L(theta+-deltaTheta) instead in SPSA algorithm


def callback(nev, x, fx, a, flag):
    print(f"Number of call {nev:5}; current loss is {fx}")
    loss_proper_values.append(fx)

# %%
theta = initialize_theta_random(circ_depth=depth, num_qbits = n_qubits)
theta_vector = np.reshape(theta, theta.size)
circ_depth, num_qbits = theta.shape

spsa = SPSA(maxiter=max_iter, callback=callback)
results = spsa.minimize(compute_kl_loss_filled, theta_vector)

# %%
fig, ax = plt.subplots()
ax.plot(loss_values, label=f"KL Loss")
# plt.ylim(0, 30)
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title(f"Loss Function over Iterations - Alternating Ansatz + KL divergence")
ax.legend()

# fig.savefig("loss_curve.png")

# %%

def simulate_circ(circ):
    """
    Generates our estimate state |phi> via simulation
    :param circ: qiskit.QuantumCircuit object, the variational quantum circuit
    :return: qiskit.Statevector object, represents our estimated quantum state |phi>
    """

    simulator = AerSimulator(method='statevector')
    circ.save_statevector() 
    job = simulator.run(circ)
    result = job.result()
    circ_statevect = Statevector(result.get_statevector(circ))
    
    return circ_statevect

def compute_fidelity(psi, phi):
    """
    Compute the fidelity (a measure of similarity) between the two states
    :param psi: qiskit.Statevector, our target state |psi>
    :param phi: qiskit.Statevector, our estimated state |phi>
    :return: float, fidelity
    """

    fidelity = qiskit.quantum_info.state_fidelity(psi, phi)
    print(f"psi: {psi}, phi: {phi}")
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # plot_bloch_multivector(psi)
    # axes[0].set_title("psi")
    # plot_bloch_multivector(phi)
    # axes[1].set_title("phi")
    return fidelity



ghz_state = Statevector.from_instruction(ghz_circ)
ansatz_state = Statevector.from_instruction(ansatz.assign_parameters({param_vec : results.x}))

final_fidelity = qiskit.quantum_info.state_fidelity(ghz_state, ansatz_state)

# %%
print(f"ghz state: {ghz_state}\nansatz: {ansatz_state}")

# %%
print(final_fidelity)

# %%
print(results.x)


np.save("params_training.npy", thetas)
np.save("loss_training.npy", loss_values)
np.save("loss_training_proper.npy", loss_proper_values)
np.save("final_theta.npy", results.x)
np.save("fidelity.npy", final_fidelity)
