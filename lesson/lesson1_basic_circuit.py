"""
Lesson 1: Basic Quantum Circuit Setup
=====================================
This lesson covers:
- Creating a quantum device
- Defining a simple quantum circuit
- Executing the circuit
- Visualizing the circuit
"""

import pennylane as qml
import numpy as np

print("=" * 60)
print("LESSON 1: Basic Quantum Circuit Setup")
print("=" * 60)

# Create a quantum device (simulator)
dev = qml.device("default.qubit", wires=2)

print(f"\n[OK] Created quantum device: {dev}")
print(f"  - Device name: {dev.name}")
print(f"  - Number of wires (qubits): {len(dev.wires)}")

# Define a simple quantum circuit
@qml.qnode(dev)
def simple_circuit(phi):
    """
    A simple quantum circuit that applies a rotation and measures.
    
    Args:
        phi: Rotation angle in radians
        
    Returns:
        Expectation value of Pauli-Z operator on qubit 0
    """
    qml.RY(phi, wires=0)  # Rotate qubit 0 around Y-axis by angle phi
    return qml.expval(qml.PauliZ(0))  # Measure expectation value of Pauli-Z

# Execute the circuit
result = simple_circuit(0.5)
print(f"\n[OK] Executed simple circuit with phi=0.5")
print(f"  Result (expectation value): {result:.4f}")

# Visualize the circuit
print("\n[OK] Circuit visualization:")
print("\n" + "-" * 60)
print("UNDERSTANDING THE CIRCUIT DIAGRAM:")
print("-" * 60)
print(f"""
The visualization shows:
  0: --RY(0.50)-- <Z>

Let's break it down:
  {'-' * 60}
  | Component | Meaning                                      |
  {'-' * 60}
  | 0:        | This is Wire 0 (the first qubit)            |
  | --        | The horizontal line represents the qubit     |
  |           | flowing through time (left to right)         |
  | RY(0.50)  | A rotation gate around Y-axis with angle    |
  |           | 0.50 radians applied to this qubit          |
  | <Z>       | Measurement: we measure the Pauli-Z operator|
  {'-' * 60}

Visual representation:
  
  Wire 0:  |0> -----[RY(0.50)]----- <Z> ----- Result: {result:.4f}
           ^         ^                ^
        Initial   Rotation Gate    Measurement
        State     (Y-axis, 0.5)    (Pauli-Z)
""")

try:
    circuit_str = qml.draw(simple_circuit)(0.5)
    print("\nPennyLane's visualization:")
    print(circuit_str.encode('ascii', 'replace').decode('ascii'))
    print("\nNote: The '??' characters are box-drawing symbols")
    print("      that don't display properly in Windows console.")
except:
    print("  (Circuit visualization skipped due to encoding)")

# Create a clearer ASCII visualization
print("\n" + "-" * 60)
print("CLEARER ASCII VISUALIZATION:")
print("-" * 60)
print(f"""
Wire 0 (Qubit 0):
    |0> ----[RY(0.50)]--- <Z> ----> Output: {result:.4f}
              
    Step-by-step:
    1. Start with qubit in state |0>
    2. Apply RY gate (rotate around Y-axis by 0.5 radians)
    3. Measure using Pauli-Z operator
    4. Get expectation value: {result:.4f}

Wire 1 (Qubit 1):
    |0> ------------------------- (not used in this circuit)

EXPLANATION:
-----------
The circuit diagram shows:
  - Wire 0: The first qubit (index 0)
  - RY(0.50): A rotation gate that rotates the qubit around the Y-axis
             by 0.5 radians (about 28.6 degrees)
  - <Z>: We measure the expectation value of the Pauli-Z operator
         This tells us the qubit's state along the Z-axis

The horizontal line represents time flowing from left to right:
  Start -> Apply Gate -> Measure -> Result
""")

# Try different angles
print("\n[OK] Testing different rotation angles:")
for angle in [0.0, 0.5, 1.0, 1.5, 2.0]:
    result = simple_circuit(angle)
    print(f"  phi = {angle:.1f} -> Result = {result:.4f}")

print("\n" + "=" * 60)
print("Lesson 1 Complete!")
print("=" * 60)
print("\nKey concepts learned:")
print("  - Quantum devices are simulators or hardware backends")
print("  - @qml.qnode decorator converts a function into a quantum circuit")
print("  - qml.RY() applies a rotation gate around the Y-axis")
print("  - qml.expval() measures the expectation value of an operator")
print("  - Wires represent qubits (wires=0 is the first qubit)")

