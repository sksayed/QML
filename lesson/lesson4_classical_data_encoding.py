"""
Lesson 4: Classical Data Encoding in Quantum Circuits
====================================================
This lesson covers:
- What is classical data encoding
- Why we need to encode classical data into quantum states
- Different encoding methods (Angle Encoding, Amplitude Encoding, Basis Encoding)
- How to implement each encoding method
- Practical examples and use cases
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp  # PennyLane's numpy for automatic differentiation

print("=" * 60)
print("LESSON 4: Classical Data Encoding in Quantum Circuits")
print("=" * 60)

# Create a quantum device
dev = qml.device("default.qubit", wires=3)

print(f"\n[OK] Created quantum device with {len(dev.wires)} qubits")

# ============================================================================
# PART 1: Understanding Classical Data Encoding
# ============================================================================
print("\n" + "-" * 60)
print("PART 1: Understanding Classical Data Encoding")
print("-" * 60)

print("""
WHAT IS CLASSICAL DATA ENCODING?
--------------------------------
Classical data encoding is the process of converting regular (classical) data
into quantum states that can be processed by quantum circuits.

WHY DO WE NEED IT?
------------------
- Quantum computers work with quantum states (qubits), not classical numbers
- To use quantum algorithms, we must first encode our data into quantum states
- Different encoding methods have different advantages and use cases

ANALOGY:
--------
Think of it like translating a book:
  - Classical data = text in English
  - Quantum state = text in another language
  - Encoding = translation process
  - Quantum circuit = reader who understands the new language

COMMON ENCODING METHODS:
-----------------------
1. Angle Encoding (Rotation Encoding) - Most common, easy to implement
2. Amplitude Encoding - Efficient for large datasets
3. Basis Encoding - Direct binary representation
4. Qsample Encoding - For probability distributions
""")

# ============================================================================
# PART 2: Angle Encoding (Rotation Encoding)
# ============================================================================
print("\n" + "-" * 60)
print("PART 2: Angle Encoding (Rotation Encoding)")
print("-" * 60)

print("""
ANGLE ENCODING:
---------------
- Each classical data point becomes a rotation angle
- Uses rotation gates: RY, RX, RZ
- Simple and intuitive
- Works well for small to medium datasets
- Each qubit can encode one feature

HOW IT WORKS:
-------------
Data point x → Rotation gate RY(x) → Quantum state
""")

@qml.qnode(dev)
def angle_encoding_circuit(data):
    """
    Encode classical data using angle encoding.
    Each data point becomes a rotation angle.
    
    Args:
        data: Array of classical data points [x0, x1, x2]
    """
    # Encode each data point as a rotation angle
    qml.RY(data[0], wires=0)  # First data point → rotation on qubit 0
    qml.RY(data[1], wires=1)  # Second data point → rotation on qubit 1
    qml.RY(data[2], wires=2)  # Third data point → rotation on qubit 2
    
    return qml.expval(qml.PauliZ(0))  # Measure qubit 0

# Example: Encode some classical data
classical_data = np.array([0.5, 1.0, 1.5])
result = angle_encoding_circuit(classical_data)

print(f"\n[OK] Angle Encoding Example:")
print(f"  Classical data: {classical_data}")
print(f"  Encoded into quantum circuit")
print(f"  Measurement result: {result:.4f}")

# Visualize the circuit
print("\n[OK] Circuit visualization:")
try:
    circuit_str = qml.draw(angle_encoding_circuit)(classical_data)
    print(circuit_str.encode('ascii', 'replace').decode('ascii'))
except:
    print("  (Circuit visualization skipped due to encoding)")

print("""
Angle Encoding Details:
  - Data point 0.5 → RY(0.5) on qubit 0
  - Data point 1.0 → RY(1.0) on qubit 1
  - Data point 1.5 → RY(1.5) on qubit 2
  - Each rotation creates a quantum state based on the data value
""")

# ============================================================================
# PART 3: Normalized Angle Encoding
# ============================================================================
print("\n" + "-" * 60)
print("PART 3: Normalized Angle Encoding")
print("-" * 60)

print("""
WHY NORMALIZE?
-------------
- Rotation angles are typically in range [0, 2π]
- Real-world data can be any value
- Normalization ensures data fits in valid range
- Common normalization: scale to [0, π] or [0, 2π]
""")

def normalize_data(data, target_range=(0, np.pi)):
    """
    Normalize data to a target range (default: [0, π]).
    
    Args:
        data: Array of data points
        target_range: Tuple (min, max) for target range
        
    Returns:
        Normalized data array
    """
    min_val, max_val = target_range
    data_min = np.min(data)
    data_max = np.max(data)
    
    # Handle case where all values are the same
    if data_max == data_min:
        return np.full_like(data, (min_val + max_val) / 2)
    
    # Normalize to [0, 1] then scale to target range
    normalized = (data - data_min) / (data_max - data_min)
    scaled = normalized * (max_val - min_val) + min_val
    
    return scaled

# Example with real-world data
real_world_data = np.array([10, 25, 50, 100])  # Could be temperatures, prices, etc.
normalized_data = normalize_data(real_world_data, target_range=(0, np.pi))

print(f"\n[OK] Data Normalization Example:")
print(f"  Original data: {real_world_data}")
print(f"  Normalized to [0, π]: {normalized_data}")
print(f"  Now ready for angle encoding!")

# ============================================================================
# PART 4: Amplitude Encoding
# ============================================================================
print("\n" + "-" * 60)
print("PART 4: Amplitude Encoding")
print("-" * 60)

print("""
AMPLITUDE ENCODING:
-------------------
- Encodes data into the amplitudes of quantum states
- Very efficient: n qubits can encode 2^n data points!
- Requires data normalization (amplitudes must sum to 1)
- More complex to implement than angle encoding

HOW IT WORKS:
-------------
Data vector [a, b, c, d] → Quantum state: a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩
The amplitudes (a, b, c, d) represent the data values.

ADVANTAGES:
-----------
- Exponential encoding efficiency
- Good for large datasets
- Preserves relationships between data points

DISADVANTAGES:
--------------
- Requires normalization
- More complex to prepare
- Limited to specific data sizes (powers of 2)
""")

@qml.qnode(dev)
def amplitude_encoding_circuit():
    """
    Example of amplitude encoding.
    Note: This is a simplified example. Full amplitude encoding
    requires more sophisticated state preparation.
    """
    # For amplitude encoding, we need to prepare a specific state
    # This is a simplified version - full implementation uses
    # quantum state preparation algorithms
    qml.RY(np.pi/4, wires=0)
    qml.RY(np.pi/3, wires=1)
    qml.CNOT(wires=[0, 1])
    
    return qml.state()

print("\n[OK] Amplitude Encoding Concept:")
print("  - 2 qubits can encode 4 data points")
print("  - 3 qubits can encode 8 data points")
print("  - n qubits can encode 2^n data points!")
print("  - This is exponential efficiency!")

# ============================================================================
# PART 5: Basis Encoding (Binary Encoding)
# ============================================================================
print("\n" + "-" * 60)
print("PART 5: Basis Encoding (Binary Encoding)")
print("-" * 60)

print("""
BASIS ENCODING:
---------------
- Encodes classical bits directly into quantum basis states
- Each classical bit → one qubit
- Simple and direct
- Good for binary data

HOW IT WORKS:
-------------
Classical bit string "101" → Quantum state |101⟩
Each bit directly maps to a qubit state.
""")

@qml.qnode(dev)
def basis_encoding_circuit(bits):
    """
    Encode binary data using basis encoding.
    
    Args:
        bits: Array of bits [b0, b1, b2] where each is 0 or 1
    """
    # Apply X gate (NOT gate) if bit is 1
    if bits[0] == 1:
        qml.PauliX(wires=0)
    if bits[1] == 1:
        qml.PauliX(wires=1)
    if bits[2] == 1:
        qml.PauliX(wires=2)
    
    return qml.expval(qml.PauliZ(0))

# Example: Encode binary data
binary_data = np.array([1, 0, 1])  # Binary: 101
result = basis_encoding_circuit(binary_data)

print(f"\n[OK] Basis Encoding Example:")
print(f"  Binary data: {binary_data} (represents binary number 101)")
print(f"  Encoded as quantum state |101⟩")
print(f"  Measurement result: {result:.4f}")

print("""
Basis Encoding Details:
  - Bit 1 → Apply X gate (flips |0⟩ to |1⟩)
  - Bit 0 → No gate (stays |0⟩)
  - Direct one-to-one mapping
""")

# ============================================================================
# PART 6: Complete Example: Data Encoding + Variational Circuit
# ============================================================================
print("\n" + "-" * 60)
print("PART 6: Complete Example: Data Encoding + Variational Circuit")
print("-" * 60)

print("""
REAL-WORLD USAGE:
-----------------
In quantum machine learning, we typically:
1. Encode classical data into quantum states
2. Apply variational (trainable) layers
3. Measure and get output
4. Use gradients to train the circuit
""")

@qml.qnode(dev)
def complete_qml_circuit(data, weights):
    """
    Complete QML circuit: data encoding + variational layers
    
    Args:
        data: Classical data to encode [x0, x1, x2]
        weights: Trainable parameters [w0, w1, w2]
    """
    # Step 1: Encode classical data (angle encoding)
    qml.RY(data[0], wires=0)
    qml.RY(data[1], wires=1)
    qml.RY(data[2], wires=2)
    
    # Step 2: Apply variational (trainable) layer
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    
    qml.RY(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RY(weights[2], wires=2)
    
    # Step 3: Measure
    return qml.expval(qml.PauliZ(0))

# Example usage
sample_data = np.array([0.5, 1.0, 0.8])
sample_weights = pnp.array([0.2, 0.3, 0.4], requires_grad=True)

result = complete_qml_circuit(sample_data, sample_weights)

print(f"\n[OK] Complete QML Circuit Example:")
print(f"  Input data: {sample_data}")
print(f"  Trainable weights: {sample_weights}")
print(f"  Circuit output: {result:.4f}")

print("""
Circuit Flow:
  1. Encode data → Quantum state
  2. Apply trainable gates → Transform state
  3. Measure → Get classical output
  4. Use output for prediction/classification
""")

# ============================================================================
# PART 7: Different Data Types and Encoding Strategies
# ============================================================================
print("\n" + "-" * 60)
print("PART 7: Different Data Types and Encoding Strategies")
print("-" * 60)

print("""
ENCODING STRATEGIES BY DATA TYPE:
---------------------------------

1. CONTINUOUS DATA (real numbers):
   - Use: Angle Encoding
   - Normalize to [0, π] or [0, 2π]
   - Example: Temperature, price, height

2. BINARY DATA (0s and 1s):
   - Use: Basis Encoding
   - Direct mapping: 0 → |0⟩, 1 → |1⟩
   - Example: Yes/No, True/False

3. CATEGORICAL DATA (categories):
   - Convert to binary first, then basis encoding
   - Or use one-hot encoding → amplitude encoding
   - Example: Colors, types, classes

4. LARGE DATASETS:
   - Use: Amplitude Encoding
   - Exponential efficiency
   - Requires normalization

5. IMAGES:
   - Flatten to 1D array
   - Normalize pixel values
   - Use angle or amplitude encoding
   - Example: MNIST digits, photos
""")

# Example: Encoding different data types
print("\n[OK] Examples of Different Data Types:")

# Continuous data
temperature_data = np.array([20.5, 25.0, 30.2])
temp_normalized = normalize_data(temperature_data, target_range=(0, np.pi))
print(f"  Temperature data: {temperature_data}°C")
print(f"  Normalized: {temp_normalized}")

# Binary data
binary_features = np.array([1, 0, 1, 0, 1])
print(f"  Binary features: {binary_features}")

# Categorical data (one-hot encoded)
categories = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # 3 categories
print(f"  Categorical (one-hot): {categories}")

# ============================================================================
# PART 8: Practical Tips and Best Practices
# ============================================================================
print("\n" + "-" * 60)
print("PART 8: Practical Tips and Best Practices")
print("-" * 60)

print("""
BEST PRACTICES:
---------------

1. DATA PREPROCESSING:
   - Always normalize your data
   - Handle missing values
   - Scale features appropriately

2. CHOOSING ENCODING METHOD:
   - Small datasets (< 10 features): Angle encoding
   - Binary data: Basis encoding
   - Large datasets: Amplitude encoding
   - Images: Flatten + normalize + angle/amplitude encoding

3. FEATURE ENGINEERING:
   - Select relevant features
   - Reduce dimensionality if needed
   - Consider feature interactions

4. VALIDATION:
   - Test encoding with simple examples first
   - Verify data is properly normalized
   - Check that encoding preserves important information

COMMON PITFALLS:
----------------
- Forgetting to normalize data
- Using wrong encoding method for data type
- Encoding too many features (exponential growth!)
- Not preserving data relationships
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("Lesson 4 Complete!")
print("=" * 60)
print("""
Key concepts learned:
  [OK] Classical data encoding converts regular data to quantum states
  [OK] Angle encoding: simple, one feature per qubit
  [OK] Amplitude encoding: efficient, exponential capacity
  [OK] Basis encoding: direct binary mapping
  [OK] Data normalization is crucial
  [OK] Different data types need different encoding strategies
  [OK] Encoding is the first step in quantum machine learning

Applications:
  - Quantum machine learning models
  - Quantum classification
  - Quantum regression
  - Quantum feature maps
  - Quantum neural networks

Next steps:
  - Try encoding your own datasets
  - Experiment with different normalization methods
  - Combine encoding with variational circuits
  - Build a quantum classifier
  - Explore quantum kernels
""")

print("=" * 60)
print("Script completed successfully!")
print("=" * 60)

