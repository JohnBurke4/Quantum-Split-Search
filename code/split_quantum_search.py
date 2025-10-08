
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
import numpy as np
from qiskit_aer import AerSimulator



def diffuser_qc(num_qubits):
    qc = QuantumCircuit(num_qubits, name=f"D_{num_qubits}")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def expand_oracle(n, q, r, U_r, oracle_calls):
    diff = diffuser_qc(q)

    qc = QuantumCircuit(n)
    theta = np.arcsin(2 ** (-q / 2))
    t = int(np.round((np.pi / (4 * theta) - 0.5)))
    total_oracle_calls = 0
    if q == 2:
        
        # A different construction for a 2 qubit stage (G^dag U G)
        qc.append(U_r, range(n))
        qc.append(diff, range(r, r + q))
        qc.append(U_r, range(n))
        qc.append(diff, range(r, r + q))
        qc.append(U_r, range(n))
    else:
        
        # Reflection G^t D (G^t)^dag 
        for _ in range(t):
            qc.append(diff, range(r, r + q))
            qc.append(U_r.inverse(), range(n))
          
            total_oracle_calls += oracle_calls
        qc.append(diff, range(r, r + q))
        for _ in range(t):
            
            qc.append(U_r, range(n))
            qc.append(diff, range(r, r + q))
            total_oracle_calls += oracle_calls

    return qc, total_oracle_calls


def initial_oracle(n, control):
    qubits = QuantumRegister(n)
    circuit = QuantumCircuit(qubits)

    for i, bit in enumerate(reversed(control)):
        if bit == "0":
            circuit.x(qubits[i])
    
    circuit.mcp(np.pi, qubits[:-1], qubits[-1])
    for i, bit in enumerate(reversed(control)):
        if bit == "0":
            circuit.x(qubits[i])
    return circuit


# A dummy oracle to keep the correct circuit placement and depth
# The barrier operator keeps the correct operation order in lieu of an oracle
def dummy_oracle(n):
    qubits = QuantumRegister(n)
    circuit = QuantumCircuit(qubits)

    circuit.barrier()
    return circuit

def test_success_chance(n, stages, final_stage_size, target):
    cReg = ClassicalRegister(final_stage_size)
    qReg = QuantumRegister(n)
    qc = QuantumCircuit(qReg, cReg)
    qc.h(range(n))
    processed_qubits = 0
    U = initial_oracle(n, target)
 

    oracle_calls = 1
    for s in stages:
        U, oracle_calls = expand_oracle(n, s, processed_qubits, U, oracle_calls)
        processed_qubits += s
        


    diff = diffuser_qc(final_stage_size)
    
    # Iterations needed for search
    t0 = int(np.floor((np.pi / 4) * np.sqrt(2**final_stage_size)))

    # Final search
    total_oracle_calls = 0
    for _ in range(t0):
        qc.append(U, range(n))
        qc.append(diff, range(n - final_stage_size, n))
        total_oracle_calls += oracle_calls

    qc.measure(range(n - final_stage_size, n), cReg)

    # Testing
    simulator = AerSimulator()
    circ = transpile(qc, simulator)
    shots = 100000
    result = simulator.run(circ, shots=shots).result()

    target_substring = target[n - final_stage_size :]
    counts = result.get_counts(circ)
    success_chance = (counts[target[n - final_stage_size :]] / shots)
    return (total_oracle_calls, target_substring, success_chance)

def evaluate_gate_decomposition_overhead(n, stages, final_stage_size, backend=None):
    
    # Count operators for split search
    
    qReg = QuantumRegister(n)
    qc = QuantumCircuit(qReg)
    qc.h(range(n))
    processed_qubits = 0
    U = dummy_oracle(n)
 

    oracle_calls = 1
    for s in stages:
        U, oracle_calls = expand_oracle(n, s, processed_qubits, U, oracle_calls)
        processed_qubits += s
        


    diff = diffuser_qc(final_stage_size)
    
    # Iterations needed for search
    t0 = int(np.floor((np.pi / 4) * np.sqrt(2**final_stage_size)))

    # Final search
    total_oracle_calls = 0
    for _ in range(t0):
        qc.append(U, range(n))
        qc.append(diff, range(n - final_stage_size, n))
        total_oracle_calls += oracle_calls


    # Evaluate depth and number of CX gates
    if (not backend):
        transpilation = transpile(qc, basis_gates=["u", "cx", "id"])
        
    split_depth = transpilation.depth()
    split_cx_count = transpilation.count_ops().get('cx',0)
    
    # Evaluate Grover search decomp

    qReg = QuantumRegister(n)
    qc = QuantumCircuit(qReg)
    qc.h(range(n))
    U = dummy_oracle(n)

    diff = diffuser_qc(n)
    
    # Iterations needed for search
    t0 = int(np.floor((np.pi / 4) * np.sqrt(2**n)))

    # Final search
    total_oracle_calls = 0
    for _ in range(t0):
        qc.append(U, range(n))
        qc.append(diff, range(n))
        total_oracle_calls += oracle_calls


    # Evaluate depth and number of CX gates
    if (not backend):
        transpilation = transpile(qc, basis_gates=["u", "cx", "id"])
        
    grover_depth = transpilation.depth()
    grover_cx_count = transpilation.count_ops().get('cx',0)
    
    return (split_depth, grover_depth, split_cx_count, grover_cx_count)

    
stages = [3,3,3,3,3]
final_stage = 2
n = sum(stages) + final_stage
target = "1" * n

# Get success prob and oracle calls
(total_oracle_calls, target_substring, success_chance) = test_success_chance(n,stages,final_stage,target)

# Get circuit decomposistion stats
(split_depth, grover_depth, split_cx_count, grover_cx_count) = evaluate_gate_decomposition_overhead(n, stages, final_stage)

# Expected number of oracle calls for Grovers algorithm
expectedOracleCallsGrover = int(np.floor((np.pi / 4) * np.sqrt(2**n)))

print("Qubits:", n)
print("Final stage size (search stage):",final_stage)
print("Target bit string:", target_substring)
print("Success chance split search:", success_chance)
print("")
print("Total oracle calls:",total_oracle_calls)
print("Expected oracle calls Grover:", expectedOracleCallsGrover)
print("")
print("Circuit depth split search:",split_depth)
print("Circuit depth Grover search:",grover_depth)
print("")
print("CX count split search:",split_cx_count)
print("CX coount Grover search:",grover_cx_count)
print("")



