"""
Batched Quantum Attention mechanism for transformer.
Vectorized implementation with automatic batching support.

NOTE: This compares Q[i] with K[i] for each token position i.
It calculates the 'importance' of a token based on its own Q and K,
acting as a weighting mechanism. It does not compute N x N mixing
between different tokens (self-attention per token).
"""
import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

class BatchedQuantumAttention(nn.Module):
    """
    Vectorized Quantum Attention Layer.
    Supports batch processing for high-speed training.
    
    This layer uses quantum circuits to compute attention weights
    by comparing query and key features for each token position.
    """
    
    def __init__(self, n_qubits, n_quantum_layers=2, device='default.qubit'):
        """
        Initialize Batched Quantum Attention.
        
        Args:
            n_qubits: Number of qubits (must match feature dimension)
            n_quantum_layers: Depth of quantum circuit (number of StronglyEntanglingLayers)
            device: PennyLane device name (e.g., 'default.qubit', 'lightning.qubit')
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers

        # Define the Quantum Device
        # Use 'backprop' for simulations. If using real hardware, use 'parameter-shift'
        self.dev = qml.device(device, wires=n_qubits)
        
        # Define weight shapes for the TorchLayer
        # StronglyEntanglingLayers requires shape (n_quantum_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_quantum_layers, n_qubits, 3)}

        # Create the QNode with torch interface
        # We'll process batches manually to ensure correct behavior
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch', diff_method='backprop')
        
        # Store weight shapes for manual batch processing
        self.weight_shapes = weight_shapes
        
        # Initialize weights for the quantum circuit
        # StronglyEntanglingLayers requires shape (n_quantum_layers, n_qubits, 3)
        self.quantum_weights = nn.Parameter(
            torch.randn(n_quantum_layers, n_qubits, 3) * 0.1
        )

        # Classical post-processing
        self.softmax = nn.Softmax(dim=1)  # Softmax over sequence length

    def _circuit(self, inputs, weights):
        """
        Quantum Circuit that compares Query and Key.
        
        Args:
            inputs: Torch Tensor of size (2 * n_qubits) - concatenated [Query, Key]
            weights: Torch Tensor of shape (n_quantum_layers, n_qubits, 3)
        
        Returns:
            Tuple of expectation values for each qubit
        
        Note:
            TorchLayer calls this function for each sample in the batch.
            We use tensor slicing directly - no numpy conversion needed.
            PennyLane's AngleEmbedding works with torch tensors when interface='torch'.
        """
        # CRITICAL: Ensure inputs is 1D
        # TorchLayer should pass 1D tensors, but we flatten to handle edge cases
        if inputs.dim() > 1:
            # If inputs is multi-dimensional, flatten it
            inputs = inputs.flatten()
        
        # Validate total input size
        expected_size = 2 * self.n_qubits
        if inputs.numel() != expected_size:
            raise ValueError(
                f"Input size mismatch: expected {expected_size} elements (2 * n_qubits={self.n_qubits}), "
                f"got {inputs.numel()}. Input shape: {inputs.shape}"
            )
        
        # Split Q and K - direct tensor slicing (now guaranteed to be 1D)
        q_in = inputs[:self.n_qubits]
        k_in = inputs[self.n_qubits:]
        
        # Ensure q_in and k_in are 1D and have correct size
        if q_in.numel() != self.n_qubits or k_in.numel() != self.n_qubits:
            raise ValueError(
                f"Feature extraction failed: q_in has {q_in.numel()} elements, "
                f"k_in has {k_in.numel()} elements, expected {self.n_qubits} each"
            )

        # 1. Encode Query (Rotation Y)
        # AngleEmbedding accepts torch tensors when interface='torch'
        # Multiply by pi to scale from [0,1] to [0,π] for angle encoding
        qml.templates.AngleEmbedding(q_in * np.pi, wires=range(self.n_qubits), rotation='Y')

        # 2. Initial Entanglement (Mixing information)
        # Fixed CNOT entangling layer - creates initial correlations between qubits
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i+1])

        # 3. Encode Key (Rotation X)
        # Encoding on a different axis (X) forces non-linear interaction with Query
        qml.templates.AngleEmbedding(k_in * np.pi, wires=range(self.n_qubits), rotation='X')

        # 4. Trainable Variational Layers (The "Intelligence" of the attention)
        # StronglyEntanglingLayers provides rich entanglement patterns
        qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        
        # 5. Measure Expectation Values
        # Return as tuple for TorchLayer compatibility
        # TorchLayer will stack these into shape (batch_size, n_qubits)
        return tuple([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])

    def forward(self, query, key, value):
        """
        Vectorized Forward Pass
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, n_qubits)
            key: Key tensor of shape (batch_size, seq_len, n_qubits)
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            context: Weighted context vector of shape (batch_size, seq_len, embed_dim)
            attention_weights: Attention weights of shape (batch_size, seq_len, 1)
        
        Note:
            n_qubits must match the dimension of query and key features.
            If your features don't match, use PCA preprocessing.
            TorchLayer handles device transfers automatically (CPU for quantum, GPU for classical).
        """
        batch_size, seq_len, n_features = query.shape

        # Essential validation
        if n_features != self.n_qubits:
            raise ValueError(
                f"Feature dimension ({n_features}) must equal n_qubits ({self.n_qubits}). "
                "Use PCA preprocessing to match dimensions."
            )

        if key.shape != query.shape:
            raise ValueError(f"Key shape {key.shape} must match query shape {query.shape}")

        # --- STEP 1: PREPARE BATCH ---
        # Flatten Batch and Sequence dims: (B * S, n_qubits)
        flat_q = query.reshape(-1, n_features)
        flat_k = key.reshape(-1, n_features)

        # Concatenate Q and K: (B * S, 2 * n_qubits)
        # The circuit expects [Q, K] concatenated
        circuit_in = torch.cat([flat_q, flat_k], dim=1)

        # --- STEP 2: QUANTUM EXECUTION ---
        # Process batch manually to ensure each sample is handled correctly
        # TorchLayer was flattening the entire batch, so we process samples individually
        batch_size = circuit_in.shape[0]
        
        # Verify input shape
        if circuit_in.dim() != 2:
            raise ValueError(f"circuit_in must be 2D (batch_size, input_dim), got shape {circuit_in.shape}")
        
        if circuit_in.shape[1] != 2 * self.n_qubits:
            raise ValueError(
                f"Input feature dimension mismatch: got {circuit_in.shape[1]}, "
                f"expected {2 * self.n_qubits} (2 * n_qubits={self.n_qubits})"
            )
        
        # Process each sample in the batch
        q_out_list = []
        try:
            for i in range(batch_size):
                # Extract single sample (1D tensor)
                sample_input = circuit_in[i]  # Shape: (2 * n_qubits,)
                
                # Call QNode for this sample
                # QNode expects (inputs, weights) where inputs is 1D
                sample_output = self.qnode(sample_input, self.quantum_weights)
                
                # Convert tuple to tensor if needed
                if isinstance(sample_output, tuple):
                    sample_output = torch.stack(sample_output)
                
                q_out_list.append(sample_output)
            
            # Stack results: (batch_size, n_qubits)
            q_out = torch.stack(q_out_list)
            
        except Exception as e:
            raise RuntimeError(
                f"Quantum layer execution failed: {e}. "
                f"Input shape: {circuit_in.shape}, Expected output: ({circuit_in.shape[0]}, {self.n_qubits})"
            )

        # Basic output validation
        if q_out.dim() != 2 or q_out.shape[0] != circuit_in.shape[0] or q_out.shape[1] != self.n_qubits:
            raise ValueError(
                f"Unexpected quantum output shape: {q_out.shape}, "
                f"expected ({circuit_in.shape[0]}, {self.n_qubits})"
            )

        # --- STEP 3: AGGREGATE SCORES ---
        # Average the expectation values of all qubits to get a single attention score per token
        # Shape: (Batch * Seq, 1)
        attention_scores = torch.mean(q_out, dim=1, keepdim=True)

        # --- STEP 4: RESHAPE & APPLY SOFTMAX ---
        # Reshape back to (Batch, Seq, 1)
        attention_scores = attention_scores.reshape(batch_size, seq_len, 1)

        # Apply Softmax (Attention Distribution)
        # This creates a probability distribution over the sequence
        # Each sequence's attention weights sum to 1
        attention_weights = self.softmax(attention_scores)

        # --- STEP 5: COMPUTE WEIGHTED CONTEXT ---
        # Weighted Sum (Context Vector)
        # (Batch, Seq, 1) * (Batch, Seq, Embed_Dim) -> (Batch, Seq, Embed_Dim)
        # Broadcasting: attention_weights broadcast across embedding dimension
        context = attention_weights * value

        return context, attention_weights

# Backward compatibility alias
QuantumAttention = BatchedQuantumAttention
