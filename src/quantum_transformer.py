"""
Quantum Transformer model using PennyLane
Optimized with better naming, documentation, and attention visualization.
"""
import torch
import torch.nn as nn
from .quantum_attention import BatchedQuantumAttention

class QuantumTransformerBlock(nn.Module):
    """
    Single transformer block with Quantum Scalar Dot-Product Attention.
    Structure: Pre-Norm -> Q-Attention -> Residual -> Pre-Norm -> FFN -> Residual
    """
    
    def __init__(self, n_qubits, n_quantum_layers, embed_dim, dropout=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        
        # 1. Quantum Projections
        # Map embedding dimension to qubit dimension
        self.query_proj = nn.Linear(embed_dim, n_qubits)
        self.key_proj = nn.Linear(embed_dim, n_qubits)
        
        # Value remains in embedding dimension to preserve information content
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # 2. Quantum Attention Mechanism
        # This is the "Brain" of the block
        self.quantum_attention = BatchedQuantumAttention(n_qubits, n_quantum_layers)
        
        # 3. Output Projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # 4. Normalization & FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),  # GELU is generally better than ReLU for Transformers
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, return_attention=False):
        """
        Forward pass with batched quantum attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            return_attention: If True, returns attention weights for visualization
        
        Returns:
            If return_attention=False: Output tensor of shape (batch_size, seq_len, embed_dim)
            If return_attention=True: Tuple of (output, attention_weights)
        """
        # --- Sub-layer 1: Quantum Attention ---
        residual = x
        x = self.norm1(x)
        
        # Project to Q/K/V
        q = self.query_proj(x)  # (batch, seq, n_qubits)
        k = self.key_proj(x)    # (batch, seq, n_qubits)
        v = self.value_proj(x)  # (batch, seq, embed_dim)
        
        # Normalize Q and K to [0, 1] for Angle Embedding
        # tanh gives [-1, 1] -> +1 -> [0, 2] -> /2 -> [0, 1]
        q_norm = (torch.tanh(q) + 1.0) / 2.0
        k_norm = (torch.tanh(k) + 1.0) / 2.0
        
        # Apply Quantum Attention with error handling
        try:
            context, attn_weights = self.quantum_attention(q_norm, k_norm, v)
            # Context already has shape (batch, seq, embed_dim)
            # attn_weights has shape (batch, seq, 1) from quantum attention
            attn_output = context
        except Exception as e:
            # Fallback to classical attention if quantum fails
            print(f"Warning: Quantum attention failed, using classical fallback: {e}")
            # Classical attention fallback - must match quantum attention output shape
            # Compute full attention matrix: (batch, seq, seq)
            attention_scores_full = torch.bmm(q, k.transpose(1, 2)) / (self.n_qubits ** 0.5)
            attention_matrix = torch.softmax(attention_scores_full, dim=-1)  # (batch, seq, seq)
            
            # Reduce to match quantum attention shape: (batch, seq, 1)
            # Option 1: Take mean over sequence dimension (average attention received)
            # Option 2: Take diagonal (self-attention weights)
            # Option 3: Take sum (total attention received)
            # We use mean to match the quantum behavior of aggregating attention
            attn_weights = torch.mean(attention_matrix, dim=-1, keepdim=True)  # (batch, seq, 1)
            
            # Apply softmax to normalize the reduced attention weights
            attn_weights = torch.softmax(attn_weights.squeeze(-1), dim=-1).unsqueeze(-1)  # (batch, seq, 1)
            
            # Compute weighted context using the reduced attention weights
            # Broadcast (batch, seq, 1) * (batch, seq, embed_dim) -> (batch, seq, embed_dim)
            attn_output = attn_weights * v
        
        # Ensure float32 dtype for compatibility
        attn_output = attn_output.float()
        
        # Project back and Residual
        x = self.output_proj(attn_output)
        x = x + residual
        
        # --- Sub-layer 2: Feed Forward ---
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        
        if return_attention:
            return x, attn_weights
        return x

class QuantumTransformer(nn.Module):
    """
    Full Quantum Transformer Classifier
    
    Args:
        input_dim: Dimension of input features
        embed_dim: Embedding dimension (default: 64)
        n_transformer_layers: Number of transformer blocks (default: 3)
            NOTE: This was previously called 'n_heads', but that name was misleading.
            This is NOT multi-head attention. Instead, it's the number of stacked
            transformer blocks (layers). Each block contains one quantum attention
            mechanism. The name 'n_heads' was confusing because:
            - In standard transformers, 'heads' refers to parallel attention heads
            - Here, we stack sequential transformer blocks, not parallel heads
            - Example: n_transformer_layers=3 means 3 stacked QuantumTransformerBlocks
        
        n_quantum_layers: Depth of quantum circuit (default: 2)
            NOTE: This was previously called 'n_layers', but that name was ambiguous.
            This parameter controls the depth/complexity of the quantum circuit INSIDE
            each quantum attention mechanism. It determines how many StronglyEntanglingLayers
            are used in the quantum circuit. This is different from n_transformer_layers:
            - n_transformer_layers: How many transformer blocks to stack (classical depth)
            - n_quantum_layers: How deep the quantum circuit is within each block (quantum depth)
            - Example: n_quantum_layers=2 means 2 layers of StronglyEntanglingLayers in the circuit
        
        n_qubits: Number of qubits (default: 4)
        n_classes: Number of output classes (default: 2)
        dropout: Dropout rate (default: 0.1)
        max_seq_len: Maximum sequence length (default: 100)
    """
    
    def __init__(self, 
                 input_dim,
                 embed_dim=64,
                 n_transformer_layers=3,  # Renamed from n_heads for accuracy
                 n_quantum_layers=2,      # Renamed from n_layers for clarity
                 n_qubits=4,
                 n_classes=2,
                 dropout=0.1,
                 max_seq_len=100,
                 # Backward compatibility: support old parameter names
                 n_heads=None,
                 n_layers=None):
        super().__init__()
        
        # Handle backward compatibility: map old names to new names
        if n_heads is not None:
            n_transformer_layers = n_heads
        if n_layers is not None:
            n_quantum_layers = n_layers
        
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        
        # 1. Embedding Layer
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # 2. Learnable Positional Encoding
        # We use a Parameter so the model can learn the best way to represent "time"
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )
        
        # 3. Stack of Transformer Blocks
        # PARAMETER NAMING EXPLANATION:
        # n_transformer_layers determines how many QuantumTransformerBlocks we stack.
        # Each block is a complete transformer layer with:
        #   - Quantum attention mechanism
        #   - Feed-forward network
        #   - Residual connections
        # This is the "classical" depth of the model (how many layers to stack).
        # The old name 'n_heads' was misleading because we're not doing multi-head attention.
        self.blocks = nn.ModuleList([
            QuantumTransformerBlock(
                n_qubits=n_qubits, 
                n_quantum_layers=n_quantum_layers,  # Passed to quantum circuit depth
                embed_dim=embed_dim, 
                dropout=dropout
            )
            for _ in range(n_transformer_layers)  # Number of blocks to stack
        ])
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),  # Added Norm for stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes)
        )
        
    def forward(self, x, return_attention=False):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            return_attention: If True, returns attention weights from last block
        
        Returns:
            If return_attention=False: Logits tensor of shape (batch_size, n_classes)
            If return_attention=True: Tuple of (logits, attention_weights)
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed Input
        x = self.input_embedding(x)
        
        # Add Positional Encoding
        # We slice to the current sequence length
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Pass through Transformer Blocks
        # We collect attention weights from the last block for visualization
        last_attn_weights = None
        
        for i, block in enumerate(self.blocks):
            # Only return attention from last block if requested
            if return_attention and i == len(self.blocks) - 1:
                x, last_attn_weights = block(x, return_attention=True)
            else:
                x = block(x, return_attention=False)
        
        # Global Average Pooling (Sequence -> Vector)
        # Reduces (Batch, Seq, Embed) -> (Batch, Embed)
        x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        if return_attention:
            return logits, last_attn_weights
        return logits
    
    # Backward compatibility: Support old parameter names
    @classmethod
    def from_legacy_params(cls, input_dim, embed_dim=64, n_heads=4, n_layers=3, 
                          n_qubits=4, n_classes=2, dropout=0.1, max_seq_len=100):
        """
        Create model using legacy parameter names for backward compatibility
        
        PARAMETER MAPPING EXPLANATION:
        - n_heads (old) -> n_transformer_layers (new)
          The old name 'n_heads' was misleading. This parameter controls how many
          transformer blocks to stack, not how many attention heads to use.
          In standard transformers, "heads" means parallel attention mechanisms.
          Here, we stack sequential transformer blocks, each with one quantum attention.
        
        - n_layers (old) -> n_quantum_layers (new)
          The old name 'n_layers' was ambiguous. This parameter controls the depth
          of the quantum circuit (number of StronglyEntanglingLayers) within each
          quantum attention mechanism. It's the "quantum depth", not the "model depth".
        
        Args:
            n_heads: Maps to n_transformer_layers (number of stacked transformer blocks)
            n_layers: Maps to n_quantum_layers (depth of quantum circuit)
        """
        return cls(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_transformer_layers=n_heads,      # Old 'n_heads' -> new 'n_transformer_layers'
            n_quantum_layers=n_layers,         # Old 'n_layers' -> new 'n_quantum_layers'
            n_qubits=n_qubits,
            n_classes=n_classes,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
