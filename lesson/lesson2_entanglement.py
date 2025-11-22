"""
Lesson 2 Part 8: Starting with Qubit 0 in |1> State
====================================================
This lesson demonstrates:
- Initializing qubit 0 to |1> state
- Creating superposition and entanglement
- Measuring actual values (0 or 1) to show correlation
"""

import pennylane as qml
import numpy as np

print("=" * 60)
print("ENTANGLEMENT: Starting with Qubit 0 in |1> State")
print("=" * 60)

# Create quantum devices
dev = qml.device("default.qubit", wires=2)  # For probabilities
dev_samples = qml.device("default.qubit", wires=2, shots=7)  # For sampling

print(f"\n[OK] Created quantum devices with {len(dev.wires)} qubits")

# ============================================================================
# PART 8: Starting with Qubit 0 in |1> State
# ============================================================================
print("\n" + "-" * 60)
print("SIMULATION: Starting with Qubit 0 in |1> State")
print("-" * 60)

print("""
NEW SIMULATION:
--------------
We'll now:
  1. Initialize qubit 0 to |1> (instead of |0>)
  2. Create superposition on qubit 0
  3. Entangle qubit 0 with qubit 1
  4. Measure both qubits
""")

# Create a separate device with shots for sampling
@qml.qnode(dev_samples)
def entangled_from_one_samples():
    """
    Start with qubit 0 in |1> state, then entangle with qubit 1.
    Returns actual measurement samples (0 or 1).
    """
    # Step 1: Initialize qubit 0 to |1> (flip from |0> to |1>)
    qml.PauliX(wires=0)  # X gate flips |0> -> |1>
    
    # Step 2: Create superposition on qubit 0
    qml.Hadamard(wires=0)  # Now |1> -> (|0> - |1>)/sqrt(2)
    
    # Step 3: Entangle qubit 0 and qubit 1
    qml.CNOT(wires=[0, 1])  # Create entanglement
    
    # Step 4: Measure both qubits (returns actual 0 or 1)
    return qml.sample(qml.PauliZ(0)), qml.sample(qml.PauliZ(1))

# Execute the circuit multiple times to show correlation
print(f"\n[OK] Circuit execution (actual measurements):")
print(f"  Initial state: Qubit 0 = |1>, Qubit 1 = |0>")
print(f"  Running 10 measurements to show correlation...")
print(f"\n  {'Measurement':<15} {'Qubit 0':<15} {'Qubit 1':<15} {'Match':<10}")
print(f"  {'-' * 55}")

# Note: qml.sample() returns eigenvalues: +1 for |0>, -1 for |1>
# We'll convert to 0/1 for easier reading
samples_q0, samples_q1 = entangled_from_one_samples()

# Convert from eigenvalues (+1/-1) to bit values (0/1)
# +1 means |0> (bit 0), -1 means |1> (bit 1)
results_q0 = [(0 if s == 1 else 1) for s in samples_q0]
results_q1 = [(0 if s == 1 else 1) for s in samples_q1]

for i in range(len(results_q0)):
    match = "Yes" if results_q0[i] == results_q1[i] else "No"
    print(f"  {i+1:<15} {results_q0[i]:<15} {results_q1[i]:<15} {match:<10}")

# Summary statistics
matches = sum(1 for i in range(len(results_q0)) if results_q0[i] == results_q1[i])
print(f"\n  Summary:")
print(f"    Total measurements: {len(results_q0)}")
print(f"    Qubits matched: {matches}/{len(results_q0)} ({100*matches/len(results_q0):.0f}%)")
print(f"    This shows perfect correlation in Bell state!")

# ============================================================================
# COMMENTED OUT: Visualization and additional explanations
# ============================================================================
# Visualize the circuit
# print("\n" + "-" * 60)
# print("CIRCUIT VISUALIZATION:")
# print("-" * 60)
#
# print(f"""
# Circuit diagram:
#
# Wire 0: |0> --[X]--[H]--*-- [Measure] --> 0 or 1
#                          |
# Wire 1: |0> -------------X-- [Measure] --> 0 or 1
#          ^     ^   ^     ^      ^
#          |     |   |     |      |
#       Start  Flip Super Entangle Measure
#       |0>    to   position      (actual
#               |1>                0 or 1)
#
# Step-by-step:
#   {'-' * 60}
#   | Step | Operation | State Description                    |
#   {'-' * 60}
#   | 1    | [X] on 0  | Qubit 0: |0> -> |1>                 |
#   |      |           | Qubit 1: |0> (unchanged)              |
#   |      |           | State: |10>                          |
#   {'-' * 60}
#   | 2    | [H] on 0  | Qubit 0: |1> -> (|0> - |1>)/sqrt(2)  |
#   |      |           | Qubit 1: |0> (unchanged)              |
#   |      |           | State: (|00> - |10>)/sqrt(2)         |
#   {'-' * 60}
#   | 3    | CNOT      | Entangle qubits                      |
#   |      |           | State: (|00> - |11>)/sqrt(2)         |
#   |      |           | This is a different Bell state!       |
#   {'-' * 60}
#   | 4    | Measure   | Get actual measurement (0 or 1)      |
#   |      |           | Qubit 0: 0 or 1 (50% each)          |
#   |      |           | Qubit 1: 0 or 1 (50% each)          |
#   |      |           | Always match! (perfect correlation)  |
#   {'-' * 60}
# """)
#
# # Show probabilities
# @qml.qnode(dev)
# def entangled_from_one_probs():
#     \"\"\"Same circuit but return probabilities\"\"\"
#     qml.PauliX(wires=0)  # Initialize qubit 0 to |1>
#     qml.Hadamard(wires=0)  # Create superposition
#     qml.CNOT(wires=[0, 1])  # Entangle
#     return qml.probs(wires=[0, 1])  # Return probabilities
#
# probs = entangled_from_one_probs()
# print("\n[OK] Measurement probabilities:")
# print(f"  |00>: {probs[0]:.4f} (both qubits are |0>)")
# print(f"  |01>: {probs[1]:.4f} (qubit 0=|0>, qubit 1=|1>)")
# print(f"  |10>: {probs[2]:.4f} (qubit 0=|1>, qubit 1=|0>)")
# print(f"  |11>: {probs[3]:.4f} (both qubits are |1>)")
# print(f"  Sum: {sum(probs):.4f} (should be 1.0)")
#
# print("\n" + "-" * 60)
# print("KEY OBSERVATIONS:")
# print("-" * 60)
# print(f"""
# 1. Starting State: |10> (qubit 0 = |1>, qubit 1 = |0>)
#
# 2. After Hadamard on qubit 0:
#    - Qubit 0 becomes: (|0> - |1>)/sqrt(2)
#    - Note the MINUS sign (different from starting with |0>)
#    - State: (|00> - |10>)/sqrt(2)
#
# 3. After CNOT:
#    - Creates Bell state: (|00> - |11>)/sqrt(2)
#    - This is a DIFFERENT Bell state than starting from |00>!
#    - Qubits are still perfectly correlated:
#      * If qubit 0 is |0>, qubit 1 is |0>
#      * If qubit 0 is |1>, qubit 1 is |1>
#    - But the phase is different (minus sign)
#
# 4. Measurement Results:
#    - Each measurement gives actual 0 or 1 (not average)
#    - Qubit 0: 50% chance of 0, 50% chance of 1
#    - Qubit 1: 50% chance of 0, 50% chance of 1
#    - They ALWAYS match! (perfect correlation)
#    - If qubit 0 = 0, then qubit 1 = 0
#    - If qubit 0 = 1, then qubit 1 = 1
#
# 5. Probability Distribution:
#    - |00> probability: {probs[0]:.4f} (50%)
#    - |11> probability: {probs[3]:.4f} (50%)
#    - |01> and |10> probabilities: 0 (perfect correlation!)
# """)
#
# print("\n" + "=" * 60)
# print("Simulation Complete!")
# print("=" * 60)
