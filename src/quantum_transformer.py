"""
Quantum Transformer model using PennyLane
"""
import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from .quantum_attention import QuantumAttention

class QuantumTransformerBlock(nn.Module):
    """Single transformer block with quantum attention"""
    
    def __init__(self, n_qubits, n_layers, embed_dim, dropout=0.1):
        super().__init__()
        self.n_qubits = n_qubits
        self.embed_dim = embed_dim
        
        # Classical linear layers for Q, K, V projection
        self.query_proj = nn.Linear(embed_dim, n_qubits)
        self.key_proj = nn.Linear(embed_dim, n_qubits)
        self.value_proj = nn.Linear(embed_dim, n_qubits)
        
        # Quantum attention
        self.quantum_attention = QuantumAttention(n_qubits, n_layers)
        self.qnode = self.quantum_attention.create_qnode()
        
        # Quantum weights (trainable)
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits) * 0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(n_qubits, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """Forward pass"""
        # Self-attention
        residual = x
        x = self.norm1(x)
        
        # Project to quantum space
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        # Apply quantum attention (for each sequence position)
        batch_size, seq_len, _ = x.shape
        quantum_outputs = []
        
        for b in range(batch_size):
            seq_outputs = []
            for s in range(seq_len):
                # Normalize to [0, π] for angle encoding
                q_norm = torch.clamp(q[b, s], -np.pi, np.pi)
                k_norm = torch.clamp(k[b, s], -np.pi, np.pi)
                v_norm = torch.clamp(v[b, s], -np.pi, np.pi)
                
                # Quantum attention
                try:
                    q_out = torch.stack(self.qnode(q_norm, k_norm, v_norm, 
                                                   self.quantum_weights))
                    # Ensure float32 dtype to match model
                    q_out = q_out.float()
                    seq_outputs.append(q_out)
                except Exception as e:
                    # Fallback to classical attention if quantum fails
                    print(f"Warning: Quantum attention failed, using classical fallback: {e}")
                    q_out = (q[b, s] + k[b, s] + v[b, s]) / 3.0
                    q_out = q_out.float()
                    seq_outputs.append(q_out)
            
            quantum_outputs.append(torch.stack(seq_outputs))
        
        attn_output = torch.stack(quantum_outputs)
        # Ensure float32 dtype
        attn_output = attn_output.float()
        attn_output = self.output_proj(attn_output)
        x = residual + attn_output
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x

class QuantumTransformer(nn.Module):
    """Full Quantum Transformer model"""
    
    def __init__(self, 
                 input_dim,
                 embed_dim=64,
                 n_heads=4,
                 n_layers=3,
                 n_qubits=4,
                 n_classes=2,
                 dropout=0.1,
                 max_seq_len=100):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.1
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            QuantumTransformerBlock(n_qubits, n_layers, embed_dim, dropout)
            for _ in range(n_heads)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes)
        )
        
    def forward(self, x):
        """Forward pass"""
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Embedding
        x = self.input_embedding(x)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        
        # Classification
        logits = self.classifier(x)
        return logits

