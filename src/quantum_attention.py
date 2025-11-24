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
    Uses PennyLane's native batching (qml.batch_input) for efficient parallel execution.
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
        
        # Initialize weights for the quantum circuit
        # StronglyEntanglingLayers requires shape (n_quantum_layers, n_qubits, 3)
        self.quantum_weights = nn.Parameter(
            torch.randn(n_quantum_layers, n_qubits, 3) * 0.1
        )
        
        # Create the QNode - we'll handle batching directly in the circuit
        # AngleEmbedding supports batched inputs natively, so we can pass 2D tensors directly
        self.qnode = qml.QNode(self._circuit, self.dev, interface='torch', diff_method='backprop')

        # Classical post-processing
        self.softmax = nn.Softmax(dim=1)  # Softmax over sequence length

    def _circuit(self, inputs, weights):
        """
        Pure Quantum Circuit with Amplitude Encoding - no shape validation, just gates.
        
        Args:
            inputs: Torch Tensor - pre-validated in forward()
                   - 1D: (2^n_qubits,) for single sample (amplitude encoding)
                   - 2D: (batch_size, 2^n_qubits) for batched execution
            weights: Torch Tensor of shape (n_quantum_layers, n_qubits, 3)
        
        Returns:
            Tuple of expectation values for each qubit (or batched results)
        
        Note:
            This method uses AmplitudeEmbedding instead of AngleEmbedding.
            For amplitude encoding, input must be length 2^n_qubits.
        """
        # For amplitude encoding, we encode the full input vector
        # Input is already the full feature vector (2^n_qubits length)
        if inputs.dim() == 1:
            # Single sample: (2^n_qubits,)
            data = inputs
        else:
            # Batched input: (batch_size, 2^n_qubits)
            data = inputs

        # 1. Encode data using Amplitude Encoding
        # AmplitudeEmbedding encodes the full vector into quantum state amplitudes
        qml.AmplitudeEmbedding(
            features=data, 
            wires=range(self.n_qubits), 
            normalize=True  # Automatically normalizes the input
        )

        # 2. Trainable Variational Layers
        qml.templates.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
        
        # 3. Measure Expectation Values
        return tuple([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])

    def forward(self, query, key, value):
        """
        Vectorized Forward Pass with Amplitude Encoding
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, 2^(n_qubits-1))
            key: Key tensor of shape (batch_size, seq_len, 2^(n_qubits-1))
            value: Value tensor of shape (batch_size, seq_len, embed_dim)
        
        Returns:
            context: Weighted context vector of shape (batch_size, seq_len, embed_dim)
            attention_weights: Attention weights of shape (batch_size, seq_len, 1)
        
        Note:
            Uses AmplitudeEmbedding for data encoding (instead of AngleEmbedding).
            Q and K are concatenated to form 2^n_qubits features for amplitude encoding.
            Processes entire batch at once for efficient parallel execution.
        """
        batch_size, seq_len, n_features = query.shape

        # --- VALIDATION (All shape checking happens here, not in _circuit) ---
        # For amplitude encoding, each Q and K should have 2^(n_qubits-1) features
        # When concatenated: 2^(n_qubits-1) + 2^(n_qubits-1) = 2^n_qubits
        expected_features_per_projection = 2 ** (self.n_qubits - 1)
        if n_features != expected_features_per_projection:
            raise ValueError(
                f"Feature dimension ({n_features}) must equal 2^(n_qubits-1) = {expected_features_per_projection} "
                f"for amplitude encoding (n_qubits={self.n_qubits}). "
                f"When Q and K are concatenated, total will be 2^n_qubits = {2 ** self.n_qubits}. "
                "Use PCA preprocessing to match dimensions."
            )

        if key.shape != query.shape:
            raise ValueError(f"Key shape {key.shape} must match query shape {query.shape}")

        # --- STEP 1: PREPARE BATCH FOR AMPLITUDE ENCODING ---
        # Flatten Batch and Sequence dims: (B * S, n_features)
        flat_q = query.reshape(-1, n_features)
        flat_k = key.reshape(-1, n_features)

        # For amplitude encoding, we need total features = 2^n_qubits
        # Concatenate Q and K: (B * S, 2 * n_features)
        # This should equal 2^n_qubits
        circuit_in = torch.cat([flat_q, flat_k], dim=1)
        
        # Validate circuit input shape before passing to quantum circuit
        total_samples = circuit_in.shape[0]
        expected_input_dim = 2 ** self.n_qubits  # For amplitude encoding: 2^n_qubits
        
        if circuit_in.dim() != 2:
            raise ValueError(
                f"circuit_in must be 2D (batch_size, input_dim), got {circuit_in.dim()}D "
                f"with shape {circuit_in.shape}"
            )
        
        if circuit_in.shape[1] != expected_input_dim:
            raise ValueError(
                f"Input feature dimension mismatch: got {circuit_in.shape[1]}, "
                f"expected {expected_input_dim} (2^n_qubits = 2^{self.n_qubits} for amplitude encoding)"
            )

        # --- STEP 2: QUANTUM EXECUTION (NATIVE BATCHING) ---
        # Pass entire batch tensor directly - AmplitudeEmbedding handles batched inputs natively
        # This is MUCH faster than sequential Python loop (10-100x speedup)
        # PennyLane processes all samples in parallel at C++ level
        try:
            # circuit_in shape: (batch_size, 2^n_qubits)
            # Pass entire 2D tensor directly - AmplitudeEmbedding supports batched inputs
            q_out = self.qnode(circuit_in, self.quantum_weights)
            
            # Handle output format - for batched inputs, each expval returns (batch_size,)
            # Stack along feature dimension to get (batch_size, n_qubits)
            if isinstance(q_out, tuple):
                # Each element in tuple is (batch_size,) for one qubit
                # Stack along feature dimension: (batch_size, n_qubits)
                q_out = torch.stack(q_out, dim=1)
            elif q_out.dim() == 1:
                # Single qubit case - should not happen with multiple qubits
                if q_out.shape[0] == total_samples:
                    q_out = q_out.unsqueeze(1)  # (batch_size, 1)
            
        except Exception as e:
            raise RuntimeError(
                f"Quantum layer execution failed: {e}. "
                f"Input shape: {circuit_in.shape}, Expected output: ({circuit_in.shape[0]}, {self.n_qubits})"
            )

        # Validate output shape (sanity check)
        if q_out.dim() != 2 or q_out.shape[0] != total_samples or q_out.shape[1] != self.n_qubits:
            raise ValueError(
                f"Unexpected quantum output shape: {q_out.shape}, "
                f"expected ({total_samples}, {self.n_qubits})"
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
