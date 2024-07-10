import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
#from qiskit_aer import AerSimulator
import numpy as np
from BinaryStrings import *
import itertools
import matplotlib.pyplot as plt
import scipy


# a global variable to store the best solution to the string breeding problem computed so far
SbpSolnLibrary = dict()


class BFTAnsatz:
    def __init__(self, n, parity_qubit_id):
        # number of data qubits, including the parity qubit
        self.n = n
        assert isinstance(n, int) and n > 0

        # index of the control qubit
        # note: the data qubits FOLLOW IMMEDIATELY AFTER THE PARITY QUBIT,
        # so that their indexes are parity_qubit_id + 1, ..., parity_qubit_id + n.
        self.parity_qubit_id = parity_qubit_id
        assert isinstance(parity_qubit_id, int) and parity_qubit_id >= 0

        # a solution to the string breeding problem, necessary to construct the phase module
        self.sbp_soln = None

    # load a solution to the string breeding problem
    def solve_sbp(self):
        global SbpSolnLibrary

        # look for solution in the library, if don't find it there then solve
        if self.n in SbpSolnLibrary:
            self.sbp_soln = SbpSolnLibrary[self.n]
        else:
            self.sbp_soln = sbp_solve_greedy(self.n)
            SbpSolnLibrary[self.n] = self.sbp_soln

    # compute the rotation parameters for the phase module on n qubits (which is 2**(n-1) degrees of freedom)
    # so that the goal state specified by u is loaded into the quantum computer
    def compute_theta(self, u):
        assert isinstance(u, np.ndarray)
        assert len(u) == 2 ** (self.n - 1)  # check the number of degrees of freedom

        # compute the angles phi as minus the arcsin of the binary Fourier transform of u
        phi = np.zeros(2 ** (self.n - 1))
        for b in range(2 ** (self.n - 1)):
            s = 0
            for c in range(2 ** (self.n - 1)):
                s += u[c] * (-1) ** binary_dot(b, c)
            s = -s
            phi[b] = np.arcsin(s)

        # for every odd-weight binary string, compute the corresponding rotation angle
        theta = {}
        for B in range(2 ** self.n):
            if binary_weight(B) % 2 == 1:
                b = np.right_shift(B, 1)  # discard the zeroth bit
                s = 0
                for c in range(2 ** (self.n - 1)):
                    s += phi[c] * (-1) ** binary_dot(b, c)
                s /= 2 ** (self.n - 2)
                theta[B] = s

        return theta

    # generate the rotation part of the phase module
    def add_rot_part(self, qc, theta):
        assert isinstance(theta, dict)         # Z rotation angles, computed by self.compute_theta
        assert isinstance(qc, QuantumCircuit)  # the quantum circuit to which gates will be added

        # place an initial layer of Z-rotation gates
        for i in range(self.n):
            qc.rz(theta[2 ** i], i + self.parity_qubit_id)

        # place the remaining Z-rotation gates
        assert not isinstance(self.sbp_soln, type(None))  # prior to calling this function, need to call solve_sbp
        for move in self.sbp_soln:
            if len(move) == 2:  # a CNOT gate
                i, j = move
                qc.cx(i + self.parity_qubit_id, j + self.parity_qubit_id)
            else:  # a rotation gate
                _, B, j = move
                qc.rz(theta[B], j + self.parity_qubit_id)

    # TODO: USE A RETURN METHOD WITH FEWER GATES
    def add_fix_part(self, qc):
        assert not isinstance(self.sbp_soln, type(None))  # prior to calling this function, need to call solve_sbp
        for move in reversed(self.sbp_soln):
            if len(move) == 2:
                i, j = move
                qc.cx(i + self.parity_qubit_id, j + self.parity_qubit_id)
            else:
                pass

    # given a quantum circuit qc, place the desired state u
    # on qubits self.parity_qubit_id + 1, ..., self.parity_qubit_id + self.n - 1,
    # tensored with qubit number self.parity_qubit_id being set to 1.
    def make(self, qc, u):
        theta = self.compute_theta(u)

        # apply the initial layer of Hadamards
        qc.h(range(self.parity_qubit_id, self.n + self.parity_qubit_id))

        # add the phase module
        self.add_rot_part(qc, theta)
        qc.barrier()

        # add the fix module
        self.add_fix_part(qc)
        qc.barrier()

        # add a second layer of Hadamards and a "pine tree" structure
        qc.h(range(self.parity_qubit_id, self.n + self.parity_qubit_id))
        for i in range(self.n - 1):
            qc.cx(self.n + self.parity_qubit_id - 1 - i, self.n + self.parity_qubit_id - 2 - i)
        for i in range(self.n - 2):
            qc.cx(self.parity_qubit_id + 2 + i, self.parity_qubit_id + 1 + i)


# create a quantum circuit which maps |b> to |b+1> if the control qubit is set to 1, where |b> is a
# state of n qubits b_0 ... b_{n-1} such that b_0 + 2 b_1 + ... + 2^{n-1} b_{n-1}.
# here qc is the quantum circuit to which gates will be added,
# b_qubits is an iterable that stores the qubits designated as the "data qubits,"
# ancilla_qubits is an iterable that stores the ancilla qubits,
# and control_qubit is the index of the qubit which controls whether or not incrementing will happen.
def add_controlled_binary_incrementer(qc, n, b_qubits, ancilla_qubits, control_qubit):
    assert len(b_qubits) == n                     # n is the number of data qubits b_0 ... b_{n-1}
    assert n < 3 or len(ancilla_qubits) == n - 2  # need n - 2 ancilla qubits
    assert control_qubit >= 0                     # this variable must represent a valid qubit index
    # assuming: b_qubits, ancilla_qubits, and {control_qubit} are disjoint sets

    if n == 1:
        # for one data qubit, just do the following simple circuit
        qc.cx(control_qubit, b_qubits[0])
    elif n == 2:
        # for two data qubits, just do the following
        qc.ccx(control_qubit, b_qubits[0], b_qubits[1])
        qc.cx(control_qubit, b_qubits[0])
    else:  # n >= 3
        # for three or more data qubits, implement a circuit utilizing ancilla qubits.
        # note: the number of gates is O(n).
        qc.ccx(b_qubits[0], b_qubits[1], ancilla_qubits[0])
        for i in range(n - 3):
            qc.ccx(b_qubits[i + 2], ancilla_qubits[i], ancilla_qubits[i + 1])

        qc.ccx(control_qubit, ancilla_qubits[n - 3], b_qubits[n - 1])

        for i in range(n - 4, -1, -1):
            qc.ccx(b_qubits[i + 2], ancilla_qubits[i], ancilla_qubits[i + 1])
            qc.ccx(control_qubit, ancilla_qubits[i], b_qubits[i + 2])

        qc.ccx(b_qubits[0], b_qubits[1], ancilla_qubits[0])
        qc.ccx(control_qubit, b_qubits[0], b_qubits[1])
        qc.cx(control_qubit, b_qubits[0])


# analogous to add_controlled_binary_incrementer, but maps |b> to |b-1> conditioned on the control qubit.
# TODO: implement this from scratch instead of relying on the incrementer
def add_controlled_binary_decrementer(qc, n, b_qubits, ancilla_qubits, control_qubit):
    qc.x(b_qubits)
    add_controlled_binary_incrementer(qc, n, b_qubits, ancilla_qubits, control_qubit)
    qc.x(b_qubits)


# use a quantum circuit to compute the backwards difference [x_0 - x_{n-1}, x_1 - x_0, ..., x_{n-1} - x_{n-2}],
# of an array arr = [x_0, ..., x_{n-1}], where n is a power of 2.
def q_back_diff(arr):
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 1  # need a one-dimensional array

    # compute the number of data qubits, add one for the parity qubit
    assert (arr.size & (arr.size-1) == 0) and arr.size != 0     # array size must be a power of 2
    n = np.ceil(np.log2(arr.size) + 1)
    n = int(n)

    if n == 1:
        # if n = 1, then arr.size == 1, so arr == [x] and the backward difference is just zero
        return np.array([0])
    elif n == 2:
        # if n == 2, then arr.size == 2, and no ancilla qubits will be used

        # place the differentiation circuit
        qc = QuantumCircuit(3)
        ans = BFTAnsatz(2, 1)
        ans.solve_sbp()
        ans.make(qc, arr)
        qc.barrier()

        # place the differentiation circuit
        qc.h(0)
        add_controlled_binary_incrementer(qc, n - 1, [2], [], 0)
        qc.h(0)

        # read out data
        psi = Statevector.from_instruction(qc)
        data = psi.data

        # compute a slice to pick out 110 and 111
        slc = [3, 7]
        return 2 * data[slc].imag
    else:  # n >= 3
        # load data
        qc = QuantumCircuit(2*n - 2)
        ans = BFTAnsatz(n, n - 2)
        ans.solve_sbp()
        ans.make(qc, arr)
        qc.barrier()

        # place the differentiation circuit
        qc.h(n - 3)
        add_controlled_binary_incrementer(qc, n - 1, range(n - 1, 2*n - 2), range(n - 4, -1, -1), n - 3)
        qc.h(n - 3)

        # read out data
        psi = Statevector.from_instruction(qc)
        data = psi.data

        # compute a slice to pick out 0^{n-3} 11 b_0...b_{n-2}
        slc = 2**(n - 3) * (3 + 4 * np.array(range(2**(n-1))))
        return 2 * data[slc].imag


# use a quantum circuit to compute the forward difference [x_1 - x_0, x_2 - x_1, ..., x_0 - x_{n-1}],
# of an array arr = [x_0, ..., x_{n-1}], where n is a power of 2.
def q_forw_diff(arr):
    assert isinstance(arr, np.ndarray) and len(arr.shape) == 1  # need a one-dimensional array

    # compute the number of data qubits, add one for the parity qubit
    assert (arr.size & (arr.size-1) == 0) and arr.size != 0     # array size must be a power of 2
    n = np.ceil(np.log2(arr.size) + 1)
    n = int(n)

    if n == 1:
        # if n = 1, then arr.size == 1, so arr == [x] and the backward difference is just zero
        return np.array([0])
    elif n == 2:
        # if n == 2, then arr.size == 2, and no ancilla qubits will be used

        # place the differentiation circuit
        qc = QuantumCircuit(3)
        ans = BFTAnsatz(2, 1)
        ans.solve_sbp()
        ans.make(qc, arr)
        qc.barrier()

        # place the differentiation circuit
        qc.h(0)
        add_controlled_binary_decrementer(qc, n - 1, [2], [], 0)
        qc.h(0)

        # read out data
        psi = Statevector.from_instruction(qc)
        data = psi.data

        # compute a slice to pick out 110 and 111
        slc = [3, 7]
        return -2 * data[slc].imag
    else:  # n >= 3
        # load data
        qc = QuantumCircuit(2*n - 2)
        ans = BFTAnsatz(n, n - 2)
        ans.solve_sbp()
        ans.make(qc, arr)
        qc.barrier()

        # place the differentiation circuit
        qc.h(n - 3)
        add_controlled_binary_decrementer(qc, n - 1, range(n - 1, 2*n - 2), range(n - 4, -1, -1), n - 3)
        qc.h(n - 3)

        # read out data
        psi = Statevector.from_instruction(qc)
        data = psi.data

        # compute a slice to pick out 0^{n-3} 11 b_0...b_{n-2}
        slc = 2**(n - 3) * (3 + 4 * np.array(range(2**(n-1))))
        return -2 * data[slc].imag


# given an input {q_0: v_0, ..., q_{k-1}: v_{k-1}}, where q are qubit indexes and v are 0 or 1,
# project onto the subspace where q0 == v0, ..., q_{k-1} == v_{k-1}.
# here n is the total number of qubits in the circuit.
def binary_proj_op(n, proj):
    assert isinstance(proj, dict)

    p0 = np.array([[1, 0], [0, 0]])
    p1 = np.array([[0, 0], [0, 1]])
    id = np.eye(2)

    proj_op = np.array([[1]])
    for i in range(n):
        if i in proj:
            if proj[i] == 0:  # project onto q_i == 0
                proj_op = np.kron(p0, proj_op)
            elif proj[i] == 1:  # project onto q_i == 1
                proj_op = np.kron(p1, proj_op)
            else:
                raise ValueError
        else:  # permit both q_i == 0 and q_i == 1
            proj_op = np.kron(id, proj_op)

    return proj_op
