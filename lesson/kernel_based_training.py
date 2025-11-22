"""
Kernel-based training of quantum models with scikit-learn

This tutorial demonstrates how to train quantum machine learning models using
kernel-based methods (SVM with quantum kernel) and compares it with variational
training approaches.

Based on: https://pennylane.ai/qml/demos/tutorial_kernel_based_training
"""

import numpy as np
import torch
from torch.nn.functional import relu

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers

import matplotlib.pyplot as plt

np.random.seed(42)

# ============================================================================
# DATA PREPARATION
# ============================================================================

X, y = load_iris(return_X_y=True)
#show me how the value of X and y are
print(f"X: {X}")
print(f"y: {y}")

# pick inputs and labels from the first two classes only,
# corresponding to the first 100 samples
X = X[:100]
y = y[:100]

# scaling the inputs is important since the embedding we use is periodic
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# scaling the labels to -1, 1 is important for the SVM and the
# definition of a hinge loss
y_scaled = 2 * (y - 0.5)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

n_qubits = len(X_train[0])
print(f"Number of qubits: {n_qubits}")

# ============================================================================
# KERNEL-BASED TRAINING
# ============================================================================

dev_kernel = qml.device("lightning.qubit", wires=n_qubits)

projector = np.zeros((2 ** n_qubits, 2 ** n_qubits))
projector[0, 0] = 1

@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

print(f"Kernel value for same input: {kernel(X_train[0], X_train[0])}")

# Compute the kernel matrix for training data
def kernel_matrix(A, B):
    """Compute the kernel matrix between two sets of data points."""
    return np.array([[kernel(a, b) for b in B] for a in A])

print("Computing kernel matrix for training data...")
K_train = kernel_matrix(X_train, X_train)

# Train SVM with quantum kernel
print("\nTraining SVM with quantum kernel...")
svm = SVC(kernel="precomputed").fit(K_train, y_train)

# Compute kernel matrix for test data
K_test = kernel_matrix(X_test, X_train)

# Make predictions
predictions_kernel = svm.predict(K_test)
accuracy_kernel = accuracy_score(y_test, predictions_kernel)
print(f"Accuracy on test set (kernel-based): {accuracy_kernel}")

# Check how many times the device was executed
print(f"Number of circuit evaluations (kernel-based): {dev_kernel.tracker.totals['executions']}")

# Function to compute circuit evaluations for kernel-based training
def circuit_evals_kernel(n_data, split):
    """Compute how many circuit evaluations are needed for
       kernel-based training and prediction."""
    M = int(np.ceil(split * n_data))
    Mpred = n_data - M
    
    # Training: compute kernel matrix M x M
    n_training = M * M
    # Prediction: compute kernel matrix Mpred x M
    n_prediction = Mpred * M
    
    return n_training + n_prediction

# ============================================================================
# VARIATIONAL TRAINING
# ============================================================================

dev_var = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev_var)
def variational_circuit(x, params):
    """The variational quantum circuit."""
    AngleEmbedding(x, wires=range(n_qubits))
    StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))

def variational_model(x, params):
    """The variational model."""
    return variational_circuit(x, params)

def hinge_loss(predictions, targets):
    """Hinge loss function."""
    return torch.mean(relu(1 - predictions * targets))

def cost(params, X_batch, y_batch):
    """Cost function for variational training."""
    predictions = torch.stack([variational_model(x, params) for x in X_batch])
    return hinge_loss(predictions, y_batch)

# Initialize parameters
n_layers = 3
params = np.random.random((n_layers, n_qubits, 3))
trained_params = torch.tensor(params, requires_grad=True)

# Training settings
steps = 100
batch_size = 5
opt = torch.optim.SGD([trained_params], lr=0.1)

print("\nTraining variational circuit...")
for step in range(steps):
    # Select a random batch
    batch_index = np.random.randint(0, len(X_train), (batch_size,))
    X_batch = X_train[batch_index]
    y_batch = torch.tensor(y_train[batch_index])
    
    # Update the weights by one optimizer step
    opt.zero_grad()
    loss = cost(trained_params, X_batch, y_batch)
    loss.backward()
    opt.step()
    
    if step % 10 == 0:
        print(f"step {step} , loss {loss.item()}")

# Make predictions on test set
predictions_var = []
for x in X_test:
    prediction = variational_model(x, trained_params.detach().numpy())
    predictions_var.append(np.sign(prediction))

predictions_var = np.array(predictions_var)
accuracy_var = accuracy_score(y_test, predictions_var)
print(f"Accuracy on test set (variational): {accuracy_var}")

# Check how many times the device was executed
print(f"Number of circuit evaluations (variational): {dev_var.tracker.totals['executions']}")

# Function to compute circuit evaluations for variational training
def circuit_evals_variational(n_data, n_params, n_steps, shift_terms, split, batch_size):
    """Compute how many circuit evaluations are needed for
       variational training and prediction."""
    
    M = int(np.ceil(split * n_data))
    Mpred = n_data - M
    
    n_training = n_params * n_steps * batch_size * shift_terms
    n_prediction = Mpred
    
    return n_training + n_prediction

# Estimate circuit evaluations for variational training
estimated_var_evals = circuit_evals_variational(
    n_data=len(X),
    n_params=len(trained_params.flatten()),
    n_steps=steps,
    shift_terms=2,
    split=len(X_train) / (len(X_train) + len(X_test)),
    batch_size=batch_size,
)
print(f"Estimated circuit evaluations (variational): {estimated_var_evals}")

# Function to compute model evaluations for neural networks
def model_evals_nn(n_data, n_params, n_steps, split, batch_size):
    """Compute how many model evaluations are needed for neural
       network training and prediction."""
    
    M = int(np.ceil(split * n_data))
    Mpred = n_data - M
    
    n_training = n_steps * batch_size
    n_prediction = Mpred
    
    return n_training + n_prediction

# Estimate model evaluations for neural network
estimated_nn_evals = model_evals_nn(
    n_data=len(X),
    n_params=len(trained_params.flatten()),
    n_steps=steps,
    split=len(X_train) / (len(X_train) + len(X_test)),
    batch_size=batch_size,
)
print(f"Estimated model evaluations (neural network): {estimated_nn_evals}")

# ============================================================================
# SCALING COMPARISON
# ============================================================================

print("\nComputing scaling comparison...")
variational_training1 = []
variational_training2 = []
kernelbased_training = []
nn_training = []
x_axis = range(0, 2000, 100)

for M in x_axis:
    var1 = circuit_evals_variational(
        n_data=M, n_params=M, n_steps=M, shift_terms=2, split=0.75, batch_size=1
    )
    variational_training1.append(var1)
    
    var2 = circuit_evals_variational(
        n_data=M, n_params=round(np.sqrt(M)), n_steps=M,
        shift_terms=2, split=0.75, batch_size=1
    )
    variational_training2.append(var2)
    
    kernel = circuit_evals_kernel(n_data=M, split=0.75)
    kernelbased_training.append(kernel)
    
    nn = model_evals_nn(
        n_data=M, n_params=M, n_steps=M, split=0.75, batch_size=1
    )
    nn_training.append(nn)

# Plot the scaling comparison
plt.figure(figsize=(10, 6))
plt.plot(x_axis, nn_training, linestyle='--', label="neural net")
plt.plot(x_axis, variational_training1, label="var. circuit (linear param scaling)")
plt.plot(x_axis, variational_training2, label="var. circuit (sqrt param scaling)")
plt.plot(x_axis, kernelbased_training, label="(quantum) kernel")
plt.xlabel("size of data set")
plt.ylabel("number of evaluations")
plt.legend()
plt.title("Scaling Comparison: Kernel-based vs Variational Training")
plt.tight_layout()
plt.savefig("scaling_comparison.png", dpi=150)
print("Scaling comparison plot saved as 'scaling_comparison.png'")
plt.show()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Kernel-based training accuracy: {accuracy_kernel:.4f}")
print(f"Variational training accuracy: {accuracy_var:.4f}")
print(f"\nCircuit evaluations (kernel-based): {dev_kernel.tracker.totals['executions']}")
print(f"Circuit evaluations (variational): {dev_var.tracker.totals['executions']}")
print(f"\nFor this example with {len(X)} data points:")
print(f"  - Kernel-based: {dev_kernel.tracker.totals['executions']} evaluations")
print(f"  - Variational: {dev_var.tracker.totals['executions']} evaluations")
print(f"  - Ratio: {dev_var.tracker.totals['executions'] / dev_kernel.tracker.totals['executions']:.2f}x more for variational")
