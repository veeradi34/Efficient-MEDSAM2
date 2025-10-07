"""
Efficient Memory Attention Module for EfficientMedSAM2

This module implements a memory-efficient version of the memory attention mechanism
used in MedSAM2 for handling temporal context in 3D medical images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from torch.utils.checkpoint import checkpoint


class EfficientMemoryAttentionModule(nn.Module):
    """
    Memory-efficient implementation of the memory attention mechanism for 
    handling temporal context in 3D medical images.
    
    This module uses:
    1. Fewer attention layers
    2. Reduced hidden dimensions
    3. Optional gradient checkpointing
    4. Flash attention implementation when available
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,  # Reduced from 8 in the original implementation
        dim_feedforward: int = 512,  # Reduced from 2048 in the original
        dropout: float = 0.1,
        num_layers: int = 1,  # Reduced from 4 in the original implementation
        use_checkpoint: bool = True,
    ):
        """
        Initialize the efficient memory attention module.
        
        Args:
            d_model: Hidden dimension of the attention module
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
            num_layers: Number of transformer layers
            use_checkpoint: Whether to use gradient checkpointing
        """
        super().__init__()
        
        # Save parameters
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            EfficientMemoryAttentionLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        query: torch.Tensor,  # (B, L_q, E)
        key: torch.Tensor,    # (B, L_k, E)
        value: torch.Tensor,  # (B, L_v, E)
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the efficient memory attention module.
        
        Args:
            query: Query tensor of shape (B, L_q, E)
            key: Key tensor of shape (B, L_k, E)
            value: Value tensor of shape (B, L_v, E)
            key_padding_mask: Optional mask for padding tokens
            
        Returns:
            Output tensor of shape (B, L_q, E)
        """
        output = query
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if self.use_checkpoint and self.training:
                # Use gradient checkpointing to save memory
                output = checkpoint(
                    layer,
                    output, key, value, key_padding_mask
                )
            else:
                output = layer(
                    output, key, value, key_padding_mask
                )
        
        # Apply layer norm
        output = self.norm(output)
        
        return output


class EfficientMemoryAttentionLayer(nn.Module):
    """
    Efficient implementation of a single memory attention layer.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        """
        Initialize the efficient memory attention layer.
        
        Args:
            d_model: Hidden dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of the feedforward network
            dropout: Dropout rate
        """
        super().__init__()
        
        # Save parameters
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        
        # Self-attention module
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention module
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        
        # Implementation of feedforward model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _sa_block(
        self, 
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention block"""
        # Try to use Flash Attention if available
        try:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = x + self._self_attn(
                    x, x, x,
                    key_padding_mask=key_padding_mask,
                )[0]
        except:
            # Fall back to standard attention if Flash Attention is not available
            x = x + self.self_attn(
                x, x, x,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return x
    
    def _ca_block(
        self, 
        x: torch.Tensor, 
        mem_k: torch.Tensor,
        mem_v: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attention block"""
        # Try to use Flash Attention if available
        try:
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = x + self._cross_attn(
                    x, mem_k, mem_v,
                    key_padding_mask=key_padding_mask,
                )[0]
        except:
            # Fall back to standard attention
            x = x + self.cross_attn(
                query=x,
                key=mem_k,
                value=mem_v,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return x
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feedforward block"""
        return x + self.ffn(x)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the efficient memory attention layer.
        
        Args:
            query: Query tensor of shape (B, L_q, E)
            key: Key tensor of shape (B, L_k, E)
            value: Value tensor of shape (B, L_v, E)
            key_padding_mask: Optional mask for padding tokens
            
        Returns:
            Output tensor of shape (B, L_q, E)
        """
        # Self-attention
        x = self.norm1(query)
        x = self._sa_block(x, key_padding_mask)
        query = query + self.dropout(x)
        
        # Cross-attention (memory attention)
        x = self.norm2(query)
        x = self._ca_block(x, key, value, key_padding_mask)
        query = query + self.dropout(x)
        
        # Feed-forward
        x = self.norm3(query)
        x = self._ff_block(x)
        query = query + self.dropout(x)
        
        return query
