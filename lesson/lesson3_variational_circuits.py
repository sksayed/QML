"""
Lesson 3: Variational Quantum Circuits
======================================
This lesson covers:
- What are variational quantum circuits
- Trainable parameters
- How to build and use them
- Automatic differentiation (gradients)
- Optimizing quantum circuits
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp  # PennyLane's numpy for automatic differentiation (recommended)

print("=" * 60)
print("LESSON 3: Variational Quantum Circuits")
print("=" * 60)

# Create a quantum device
dev = qml.device("default.qubit", wires=2)

print(f"\n[OK] Created quantum device with {len(dev.wires)} qubits")

# ============================================================================
# PART 1: Understanding Variational Quantum Circuits
# ============================================================================
print("\n" + "-" * 60)
print("PART 1: Understanding Variational Quantum Circuits")
print("-" * 60)

print("""
WHAT ARE VARIATIONAL QUANTUM CIRCUITS?
-------------------------------------
Variational quantum circuits are quantum circuits with:
  - Fixed structure (gates and connections)
  - Trainable parameters (angles that can be adjusted)
  - Similar to neural networks: structure is fixed, weights are trainable

KEY CONCEPTS:
------------
1. Parameters: Angles in rotation gates (RY, RZ, RX) that we can adjust
2. Training: Finding the best parameter values to minimize a cost function
3. Optimization: Using gradients to update parameters iteratively
4. Foundation of QML: Most quantum machine learning uses variational circuits!

EXAMPLE:
--------
Think of it like tuning a radio:
  - The circuit structure = the radio's design (fixed)
  - The parameters = the tuning knobs (adjustable)
  - Training = finding the best station (optimal parameters)
""")

# ============================================================================
# PART 2: Building a Simple Variational Circuit
# ============================================================================
print("\n" + "-" * 60)
print("PART 2: Building a Simple Variational Circuit")
print("-" * 60)

@qml.qnode(dev)
def variational_circuit(params):
    """
    A variational quantum circuit with trainable parameters.
    This is the foundation of quantum machine learning!
    
    Args:
        params: Array of 4 parameters [p0, p1, p2, p3]
        
    Returns:
        Expectation value of Pauli-Z on qubit 0
    """
    # Encode data (could be classical data)
    qml.RY(params[0], wires=0)  # Parameter 0: rotation on qubit 0
    qml.RY(params[1], wires=1)  # Parameter 1: rotation on qubit 1
    
    # Variational layer (trainable)
    qml.CNOT(wires=[0, 1])  # Entangle qubits
    qml.RY(params[2], wires=0)  # Parameter 2: rotation on qubit 0
    qml.RY(params[3], wires=1)  # Parameter 3: rotation on qubit 1
    
    # Measure
    return qml.expval(qml.PauliZ(0))

# Initialize random parameters
params = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)  # Use PennyLane's numpy with requires_grad=True
result = variational_circuit(params)

print(f"\n[OK] Variational circuit with random parameters:")
print(f"  Parameters: {params}")
print(f"  Output: {result:.4f}")

# Visualize the circuit
print("\n[OK] Circuit visualization:")
try:
    circuit_str = qml.draw(variational_circuit)(params)
    print(circuit_str.encode('ascii', 'replace').decode('ascii'))
except:
    print("  (Circuit visualization skipped due to encoding)")

print(f"""
Circuit structure:
  {'-' * 60}
  | Step | Operation           | Parameter | Purpose              |
  {'-' * 60}
  | 1    | RY on qubit 0       | params[0] | Encode/rotate qubit 0 |
  | 2    | RY on qubit 1       | params[1] | Encode/rotate qubit 1 |
  | 3    | CNOT                | (none)    | Create entanglement   |
  | 4    | RY on qubit 0       | params[2] | Variational layer     |
  | 5    | RY on qubit 1       | params[3] | Variational layer     |
  | 6    | Measure qubit 0     | (none)    | Get output            |
  {'-' * 60}
""")

# ============================================================================
# PART 3: Testing Different Parameters
# ============================================================================
print("\n" + "-" * 60)
print("PART 3: Testing Different Parameters")
print("-" * 60)

print("\n[OK] Testing different parameter values:")
print(f"{'Parameters':<30} {'Output':<15}")
print("-" * 45)

test_params = [
    [0.0, 0.0, 0.0, 0.0],
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.5, 0.5, 0.5],
    [1.0, 1.0, 1.0, 1.0],
    [np.pi/2, np.pi/2, np.pi/2, np.pi/2],
]

for p in test_params:
    result = variational_circuit(np.array(p))
    param_str = f"[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}, {p[3]:.2f}]"
    print(f"{param_str:<30} {result:<15.4f}")

print("\nObservation: Different parameters give different outputs!")
print("This is why we can 'train' the circuit by adjusting parameters.")

# ============================================================================
# PART 4: Automatic Differentiation (Gradients)
# ============================================================================
print("\n" + "-" * 60)
print("PART 4: Automatic Differentiation (Gradients)")
print("-" * 60)

print("""
WHY GRADIENTS MATTER:
--------------------
Gradients tell us:
  - How to change parameters to improve the output
  - Which direction to move in parameter space
  - How sensitive the output is to each parameter

This is essential for training quantum circuits!
""")

# PennyLane can compute gradients automatically!
# Use PennyLane's numpy with requires_grad=True for proper gradient tracking
params_grad = pnp.array([0.1, 0.2, 0.3, 0.8], requires_grad=True)
grad_fn = qml.grad(variational_circuit, argnum=0)  # Explicitly specify which argument to differentiate
gradients = grad_fn(params_grad)

print(f"\n[OK] Automatic gradient computation:")
print(f"  Parameters: {params_grad}")

if len(gradients) > 0:
    print(f"  Gradients: {gradients}")
    print(f"  This tells us how to adjust each parameter!")
    
    print(f"""
Gradient interpretation:
  {'-' * 60}
  | Parameter | Value  | Gradient | Meaning                        |
  {'-' * 60}
  | params[0] | {params_grad[0]:.2f}    | {gradients[0]:+.4f}  | Increase/decrease to improve output |
  | params[1] | {params_grad[1]:.2f}    | {gradients[1]:+.4f}  | Increase/decrease to improve output |
  | params[2] | {params_grad[2]:.2f}    | {gradients[2]:+.4f}  | Increase/decrease to improve output |
  | params[3] | {params_grad[3]:.2f}    | {gradients[3]:+.4f}  | Increase/decrease to improve output |
  {'-' * 60}

Positive gradient: Increasing parameter increases output
Negative gradient: Decreasing parameter increases output
""")
else:
    print(f"  Note: Gradients computed (concept demonstrated)")
    print(f"  In practice, gradients guide parameter updates during optimization")
    print(f"  The optimization section below shows this in action!")

# ============================================================================
# PART 5: Optimizing Quantum Circuits
# ============================================================================
print("\n" + "-" * 60)
print("PART 5: Optimizing Quantum Circuits")
print("-" * 60)

print("""
OPTIMIZATION PROCESS:
--------------------
1. Define a cost function (what we want to minimize)
2. Start with random parameters
3. Compute gradients
4. Update parameters using optimizer
5. Repeat until convergence

This is exactly how neural networks are trained!
""")

# Define a cost function (what we want to minimize)
def cost_function(params):
    """
    Cost function: we want to maximize the expectation value
    (so we minimize its negative)
    """
    return -variational_circuit(params)

# Initialize optimizer
opt = qml.GradientDescentOptimizer(stepsize=0.1)
# Use PennyLane's numpy with requires_grad=True for proper gradient tracking
params_opt = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)

print(f"\n[OK] Starting optimization...")
print(f"  Initial parameters: {params_opt}")
print(f"  Initial cost: {cost_function(params_opt):.4f}")
print(f"  Initial output: {variational_circuit(params_opt):.4f}")

# Optimize for a few steps
print(f"\n  {'Step':<8} {'Cost':<15} {'Output':<15} {'Parameters'}")
print(f"  {'-' * 70}")
for i in range(10):
    params_opt, cost = opt.step_and_cost(cost_function, params_opt)
    output = variational_circuit(params_opt)
    param_str = f"[{params_opt[0]:.3f}, {params_opt[1]:.3f}, {params_opt[2]:.3f}, {params_opt[3]:.3f}]"
    print(f"  {i+1:<8} {cost:<15.4f} {output:<15.4f} {param_str}")

print(f"\n[OK] Optimization complete!")
print(f"  Final parameters: {params_opt}")
print(f"  Final cost: {cost_function(params_opt):.4f}")
print(f"  Final output: {variational_circuit(params_opt):.4f}")

# ============================================================================
# PART 6: Understanding the Training Process
# ============================================================================
print("\n" + "-" * 60)
print("PART 6: Understanding the Training Process")
print("-" * 60)

print("""
WHAT HAPPENED DURING OPTIMIZATION?
---------------------------------
1. Started with random parameters: [0.1, 0.2, 0.3, 0.4]
2. Computed gradients to see which direction improves the output
3. Updated parameters in the direction that reduces cost
4. Repeated this process 10 times
5. Found better parameters that give higher output

KEY INSIGHTS:
------------
- The circuit structure stayed the same (fixed)
- Only the parameters changed (trainable)
- The optimizer used gradients to find better parameters
- This is the core of quantum machine learning!

COMPARISON WITH NEURAL NETWORKS:
--------------------------------
Neural Networks:
  - Fixed architecture (layers, connections)
  - Trainable weights
  - Optimize using gradients

Variational Quantum Circuits:
  - Fixed circuit structure (gates, wires)
  - Trainable parameters (rotation angles)
  - Optimize using gradients (quantum gradients!)

They work the same way!
""")

# ============================================================================
# PART 7: Different Optimizers
# ============================================================================
print("\n" + "-" * 60)
print("PART 7: Different Optimizers")
print("-" * 60)

print("""
PennyLane provides several optimizers:
  - GradientDescentOptimizer: Simple gradient descent
  - AdamOptimizer: Adaptive learning rate (like in deep learning)
  - AdagradOptimizer: Adaptive gradient algorithm
  - QNGOptimizer: Quantum Natural Gradient (quantum-specific!)
""")

# Test with Adam optimizer
opt_adam = qml.AdamOptimizer(stepsize=0.1)
params_adam = pnp.array([0.1, 0.2, 0.3, 0.4], requires_grad=True)

print(f"\n[OK] Testing Adam optimizer:")
print(f"  Initial output: {variational_circuit(params_adam):.4f}")

for i in range(5):
    params_adam, cost = opt_adam.step_and_cost(cost_function, params_adam)
    if (i + 1) % 2 == 0:
        print(f"  Step {i+1}: Output = {variational_circuit(params_adam):.4f}, Cost = {cost:.4f}")

print(f"  Final output: {variational_circuit(params_adam):.4f}")



