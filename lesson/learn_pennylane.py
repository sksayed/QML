"""
PennyLane Learning Script
=========================
A comprehensive guide to learning Quantum Machine Learning with PennyLane.
This script covers the fundamentals step by step.
"""

import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from autograd import numpy as anp  # Autograd's numpy for automatic differentiation

print("=" * 60)
print("PENNYLANE LEARNING SCRIPT")
print("=" * 60)
print()

# ============================================================================
# LESSON 1: Basic Quantum Circuit Setup
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 1: Basic Quantum Circuit Setup")
print("=" * 60)

# Create a quantum device (simulator)
dev = qml.device("default.qubit", wires=2)

print(f"[OK] Created quantum device: {dev}")
print(f"  - Device name: {dev.name}")
print(f"  - Number of wires (qubits): {len(dev.wires)}")

# Define a simple quantum circuit
@qml.qnode(dev)
def simple_circuit(phi):
    """
    A simple quantum circuit that applies a rotation and measures.
    """
    qml.RY(phi, wires=0)  # Rotate qubit 0 around Y-axis by angle phi
    return qml.expval(qml.PauliZ(0))  # Measure expectation value of Pauli-Z

# Execute the circuit
result = simple_circuit(0.5)
print(f"\n[OK] Executed simple circuit with phi=0.5")
print(f"  Result (expectation value): {result:.4f}")

# Visualize the circuit
print("\n[OK] Circuit visualization:")
try:
    circuit_str = qml.draw(simple_circuit)(0.5)
    print(circuit_str.encode('ascii', 'replace').decode('ascii'))
except:
    print("  (Circuit visualization skipped due to encoding)")

# ============================================================================
# LESSON 2: Multiple Gates and Entanglement
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 2: Multiple Gates and Entanglement")
print("=" * 60)

@qml.qnode(dev)
def entangled_circuit(theta):
    """
    Create a Bell state (maximally entangled state).
    """
    qml.Hadamard(wires=0)  # Create superposition
    qml.CNOT(wires=[0, 1])  # Create entanglement
    qml.RY(theta, wires=0)  # Apply rotation
    return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))

result1, result2 = entangled_circuit(0.3)
print(f"\n[OK] Entangled circuit results:")
print(f"  Qubit 0 expectation: {result1:.4f}")
print(f"  Qubit 1 expectation: {result2:.4f}")

print("\n[OK] Entangled circuit structure:")
try:
    circuit_str = qml.draw(entangled_circuit)(0.3)
    print(circuit_str.encode('ascii', 'replace').decode('ascii'))
except:
    print("  (Circuit visualization skipped due to encoding)")

# ============================================================================
# LESSON 3: Variational Quantum Circuits (Trainable Parameters)
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 3: Variational Quantum Circuits")
print("=" * 60)

@qml.qnode(dev)
def variational_circuit(params):
    """
    A variational quantum circuit with trainable parameters.
    This is the foundation of quantum machine learning!
    """
    # Encode data (could be classical data)
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    
    # Variational layer (trainable)
    qml.CNOT(wires=[0, 1])
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)
    
    # Measure
    return qml.expval(qml.PauliZ(0))

# Initialize random parameters
params = np.array([0.1, 0.2, 0.3, 0.4])
result = variational_circuit(params)
print(f"\n[OK] Variational circuit with random parameters:")
print(f"  Parameters: {params}")
print(f"  Output: {result:.4f}")

# ============================================================================
# LESSON 4: Automatic Differentiation (Gradients)
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 4: Automatic Differentiation")
print("=" * 60)

# PennyLane can compute gradients automatically!
# Note: For numpy arrays, we need to use autograd's numpy wrapper
grad_fn = qml.grad(variational_circuit)
gradients = grad_fn(params)

print(f"\n[OK] Automatic gradient computation:")
print(f"  Parameters: {params}")
print(f"  Gradients: {gradients}")
print(f"  This is crucial for training quantum models!")
print(f"  (Note: Gradients work best with autograd or ML framework arrays)")

# ============================================================================
# LESSON 5: Optimization (Training a Quantum Circuit)
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 5: Optimizing Quantum Circuits")
print("=" * 60)

# Define a cost function (what we want to minimize)
def cost_function(params):
    """
    Cost function: we want to maximize the expectation value
    (so we minimize its negative)
    """
    return -variational_circuit(params)

# Initialize optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)
# Use autograd's numpy for proper gradient tracking
params = anp.array([0.1, 0.2, 0.3, 0.4])

print(f"\n[OK] Starting optimization...")
print(f"  Initial parameters: {params}")
print(f"  Initial cost: {cost_function(params):.4f}")

# Optimize for a few steps
for i in range(5):
    params, cost = opt.step_and_cost(cost_function, params)
    print(f"  Step {i+1}: Cost = {cost:.4f}, Params = {params}")

print(f"\n[OK] Optimization complete!")
print(f"  Final parameters: {params}")
print(f"  Final cost: {cost_function(params):.4f}")

# ============================================================================
# LESSON 6: Data Encoding (Classical to Quantum)
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 6: Data Encoding")
print("=" * 60)

@qml.qnode(dev)
def data_encoding_circuit(data, weights):
    """
    Encode classical data into quantum state and apply trainable weights.
    """
    # Encode classical data (2 features)
    qml.RY(data[0], wires=0)
    qml.RY(data[1], wires=1)
    
    # Apply trainable weights
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(0))

# Example: encode some data
data_point = np.array([0.5, 0.8])
weights = np.array([0.2, 0.3])
result = data_encoding_circuit(data_point, weights)

print(f"\n[OK] Data encoding example:")
print(f"  Input data: {data_point}")
print(f"  Weights: {weights}")
print(f"  Quantum output: {result:.4f}")

# ============================================================================
# LESSON 7: Multiple Measurements
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 7: Multiple Measurements")
print("=" * 60)

@qml.qnode(dev)
def multi_measurement_circuit(phi):
    """
    Circuit that returns multiple measurements.
    """
    qml.RY(phi, wires=0)
    qml.RY(phi * 2, wires=1)
    qml.CNOT(wires=[0, 1])
    
    # Return multiple expectation values
    return [
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliX(0))
    ]

results = multi_measurement_circuit(0.5)
print(f"\n[OK] Multiple measurements:")
print(f"  <Z_0>: {results[0]:.4f}")
print(f"  <Z_1>: {results[1]:.4f}")
print(f"  <X_0>: {results[2]:.4f}")

# ============================================================================
# LESSON 8: Probability Measurements
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 8: Probability Measurements")
print("=" * 60)

@qml.qnode(dev)
def probability_circuit(phi):
    """
    Circuit that returns measurement probabilities.
    """
    qml.RY(phi, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.probs(wires=[0, 1])

probs = probability_circuit(0.5)
print(f"\n[OK] Measurement probabilities:")
print(f"  |00>: {probs[0]:.4f}")
print(f"  |01>: {probs[1]:.4f}")
print(f"  |10>: {probs[2]:.4f}")
print(f"  |11>: {probs[3]:.4f}")
print(f"  Sum (should be 1.0): {sum(probs):.4f}")

# ============================================================================
# LESSON 9: Visualization
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 9: Circuit Visualization")
print("=" * 60)

@qml.qnode(dev)
def visualization_example(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(params[2], wires=0)
    return qml.expval(qml.PauliZ(0))

print("\n[OK] Circuit visualization (ASCII):")
try:
    circuit_str = qml.draw(visualization_example)([0.1, 0.2, 0.3])
    print(circuit_str.encode('ascii', 'replace').decode('ascii'))
except:
    print("  (Circuit visualization skipped due to encoding)")

# ============================================================================
# LESSON 10: Simple Quantum Machine Learning Example
# ============================================================================
print("\n" + "=" * 60)
print("LESSON 10: Simple QML Classification Example")
print("=" * 60)

# Create a simple dataset
X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
y = np.array([1, 0, 1, 0])  # Binary labels

@qml.qnode(dev)
def qml_classifier(data, weights):
    """
    Simple quantum classifier.
    """
    # Encode data
    qml.RY(data[0], wires=0)
    qml.RY(data[1], wires=1)
    
    # Variational layer
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[2], wires=0)
    
    return qml.expval(qml.PauliZ(0))

def classify(data, weights):
    """Classify data point: > 0 means class 1, <= 0 means class 0"""
    result = qml_classifier(data, weights)
    return 1 if result > 0 else 0

# Initialize weights
weights = anp.array([0.1, 0.2, 0.3])

print(f"\n[OK] Testing QML classifier:")
for i, (data, label) in enumerate(zip(X, y)):
    prediction = classify(data, weights)
    result = qml_classifier(data, weights)
    print(f"  Data {i+1}: {data} -> Prediction: {prediction}, True: {label}, Output: {result:.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("LEARNING SUMMARY")
print("=" * 60)
print("""
You've learned:
  1. [OK] Creating quantum devices and circuits
  2. [OK] Applying quantum gates (RY, CNOT, Hadamard, etc.)
  3. [OK] Variational quantum circuits with trainable parameters
  4. [OK] Automatic differentiation (gradients)
  5. [OK] Optimizing quantum circuits
  6. [OK] Encoding classical data into quantum states
  7. [OK] Multiple types of measurements
  8. [OK] Probability measurements
  9. [OK] Circuit visualization
  10. [OK] Basic quantum machine learning

Next steps:
  - Try modifying the circuits
  - Experiment with different optimizers
  - Explore quantum kernels
  - Build hybrid quantum-classical models
  - Check out PennyLane tutorials: https://github.com/PennyLaneAI/qml
""")

print("=" * 60)
print("Script completed successfully!")
print("=" * 60)

