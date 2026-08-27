# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# WAR: Monkeypatch an internal utility function such that the gate order will remain consistent with user provided circuit
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from cuquantum.tensornet._internal import circuit_parser_utils_qiskit
from qiskit.circuit import Measure

def remove_measurements(circuit):
    for instruction in circuit.data:
        if isinstance(instruction.operation, Measure):
            raise ValueError('the input circuit can not contain measurements')
    return circuit

circuit_parser_utils_qiskit.remove_measurements = remove_measurements

import qiskit
from cuquantum.tensornet import contract, CircuitToEinsum, contract_path, Network
import cupy
from utils_circuit import ket_contract, bra_contract, create_network_and_path


n_qubits = 4

x = cupy.array([[0, 1], [1, 0]], dtype='complex128')
y = cupy.array([[0, -1j], [1j, 0]], dtype='complex128')
z = cupy.array([[1, 0], [0, -1]], dtype='complex128')
iden = cupy.array([[1, 0], [0, -1]], dtype='complex128')

#raw_op_0 = cupy.kron(x, x)
#raw_op_0 = cupy.eye(4)
#raw_op_0[3,3] = -1
raw_op_0 = cupy.kron(x, y)
raw_op_1 = cupy.kron(x, iden)
raw_op_2 = cupy.kron(z, y)


### groundtruth circuit
circuit = qiskit.QuantumCircuit(n_qubits)
circuit.h(0)
circuit.t(1)
circuit.x(2)
circuit.y(3)
circuit.cx(0, 1)
circuit.unitary(raw_op_0.get(), [0, 1], label='xx')
circuit.cz(1, 2)
circuit.unitary(raw_op_1.get(), [1, 2], label='xx')
circuit.cy(2, 3)
circuit.unitary(raw_op_2.get(), [2, 3], label='xx')


converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy', options={'check_diagonal': False})
qubits = circuit.qubits
where = (qubits[0], qubits[1], qubits[2], qubits[3])
#fixed = {qubits[2]: '0', qubits[3]: '0'}
fixed = {}
final_dim = 2**len(where)
expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)
print(expr)

net_result = create_network_and_path(expr, operands)
path, info = net_result['path'], net_result['info']
out = net_result['network'].contract()
separated_operands = out.reshape(final_dim,final_dim)


### test circuit
circuit = qiskit.QuantumCircuit(n_qubits)
circuit.h(0)
circuit.t(1)
circuit.x(2)
circuit.y(3)
circuit.cx(0, 1)
circuit.cz(1, 2)
circuit.cy(2, 3)

converter = CircuitToEinsum(circuit, dtype='complex128', backend='cupy', options={'check_diagonal': False})
qubits = circuit.qubits
where = (qubits[0], qubits[1], qubits[2], qubits[3])
#fixed = {qubits[2]: '0', qubits[3]: '0'}
fixed = {}
expr, operands = converter.reduced_density_matrix(where, fixed=fixed, lightcone=False)
operands[8] = ket_contract(operands[8], raw_op_0)
operands[13] = bra_contract(operands[13], raw_op_0)
operands[9] = ket_contract(operands[9], raw_op_1)
operands[12] = bra_contract(operands[12], raw_op_1)
operands[10] = ket_contract(operands[10], raw_op_2)
operands[11] = bra_contract(operands[11], raw_op_2)

net_result = create_network_and_path(expr, operands)
out = net_result['network'].contract()
merged_operands = out.reshape(final_dim, final_dim)

print(cupy.allclose(separated_operands, merged_operands))
