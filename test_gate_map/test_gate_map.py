# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# WAR: Monkeypatch an internal utility function such that the gate order will remain consistent with user provided circuit
from cuquantum.tensornet._internal import circuit_parser_utils_qiskit
import qiskit
from qiskit.circuit import Measure
from qiskit.circuit.random import random_circuit

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils_circuit import get_rdm_expr_gate_map


def remove_measurements(circuit):
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure):
            raise ValueError('the input circuit can not contain measurements')
    return circuit

circuit_parser_utils_qiskit.remove_measurements = remove_measurements

# create an example circuit
nqubits = 6
depth = 10
circuit = random_circuit(nqubits, depth, max_operands=2)

import cupy
from cuquantum.tensornet import CircuitToEinsum

# NOTE: set decompose_gates=False to ensure the gates specified above do not get decomposed into lower level standard gates
converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy', options={'check_diagonal': False, 'decompose_gates': False})

qubits = circuit.qubits

where = (qubits[0], qubits[2])
fixed = {qubits[1]: '0'}

expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)


def test_gate_map(circuit, num_gates, gate_map, operands):
    """
    A helper function to test the gate map by comparing the gate operands with the raw circuit data.
    Args:
        circuit: A :class:`qiskit.QuantumCircuit` object.
        num_gates: The number of gates in the circuit.
        gate_map: A dictionary mapping the gate index to the ket and bra operand indices.
        operands: A list of tensor operands.
    """
    for i in range(num_gates):
        ket_gate = operands[gate_map[f"ket_gate_{i}"]]
        bra_gate = operands[gate_map[f"bra_gate_{i}"]]
        assert ket_gate.shape == bra_gate.shape
        operation = circuit.data[i].operation
        gate_qubits = circuit.data[i].qubits
        # raw data from qiskit
        tensor = cupy.asarray(qiskit.quantum_info.Operator(operation).data.reshape((2,2)*len(gate_qubits)))
        assert tensor.shape == ket_gate.shape
        assert cupy.allclose(tensor, ket_gate)
        # make sure that the tensor is the same as the ket gate
        assert cupy.allclose(tensor, ket_gate)
        if ket_gate.ndim == 2:
            out = cupy.einsum('ab,bc->ac', ket_gate, bra_gate)
        else:
            out = cupy.einsum('ABab,abCD->ABCD', ket_gate, bra_gate).reshape(4, 4)
        # make sure that U * U^dag = I
        assert abs(out-cupy.eye(out.shape[0])).max() < 1e-10


gate_map = get_rdm_expr_gate_map(circuit, expr, fixed=fixed)
test_gate_map(circuit, len(circuit.data), gate_map, operands)
print('Test complete.')
print('No assertions failed in the testing suite.')
