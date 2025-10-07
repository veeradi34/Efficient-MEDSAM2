"""
Efficient Memory Encoder for EfficientMedSAM2

This module implements a memory-efficient version of the memory encoder used in MedSAM2
for encoding mask and feature information from previous frames/slices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from torch.utils.checkpoint import checkpoint


class EfficientMemoryEncoder(nn.Module):
    """
    Memory-efficient implementation of the memory encoder for handling
    previous frame/slice information in 3D medical images.
    
    This encoder uses fewer parameters and operations compared to the original implementation.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        mask_in_chans: int = 1,
        activation: nn.Module = nn.GELU,
        stride: int = 4,
        total_stride: int = 16,
    ):
        """
        Initialize the efficient memory encoder.
        
        Args:
            embed_dim: Embedding dimension
            mask_in_chans: Number of channels in the input mask
            activation: Activation function
            stride: Stride for downsampling
            total_stride: Total downsampling factor
        """
        super().__init__()
        
        # Save parameters
        self.embed_dim = embed_dim
        self.mask_in_chans = mask_in_chans
        
        # MaskDownSampler reduces spatial dimensions of mask and increases channels
        self.mask_downsampler = EfficientMaskDownSampler(
            embed_dim=embed_dim,
            stride=stride,
            total_stride=total_stride,
            activation=activation,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Image feature encoder for memory entries
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            activation(),
        )
        
        # Output projection
        self.output_proj = nn.Linear(2 * embed_dim, embed_dim)
        
    def forward(
        self,
        memory_feats: List[torch.Tensor],
        memory_masks: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass for the efficient memory encoder.
        
        Args:
            memory_feats: List of memory feature tensors of shape (B, C, H, W)
            memory_masks: Optional list of memory mask tensors of shape (B, 1, H, W)
            
        Returns:
            Memory embeddings of shape (B, N, E) where N is the total number of memory tokens
        """
        if not memory_feats:
            # No memory, return empty tensor
            return torch.empty(0, 0, self.embed_dim, device=memory_feats[0].device)
            
        batch_size = memory_feats[0].shape[0]
        device = memory_feats[0].device
        
        # Process each memory entry
        mem_embeddings = []
        
        for i, feat in enumerate(memory_feats):
            # Get corresponding mask if available
            mask = memory_masks[i] if memory_masks and i < len(memory_masks) else None
            
            # Process feature through feature encoder
            processed_feat = self.feature_encoder(feat)
            
            # Flatten spatial dimensions
            B, C, H, W = processed_feat.shape
            processed_feat = processed_feat.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
            
            if mask is not None:
                # Process mask through mask downsampler
                processed_mask = self.mask_downsampler(mask)  # (B, C, H', W')
                
                # Flatten spatial dimensions
                B, C, H, W = processed_mask.shape
                processed_mask = processed_mask.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
                
                # Concatenate feature and mask embeddings
                combined = torch.cat([processed_feat, processed_mask], dim=-1)
                
                # Project to embedding dimension
                mem_embed = self.output_proj(combined)
            else:
                # If no mask, just use feature embeddings with zero padding for mask part
                zero_mask = torch.zeros_like(processed_feat)
                combined = torch.cat([processed_feat, zero_mask], dim=-1)
                mem_embed = self.output_proj(combined)
            
            # Apply layer norm
            mem_embed = self.norm(mem_embed)
            
            mem_embeddings.append(mem_embed)
        
        # Stack along sequence dimension
        memory_tokens = torch.cat(mem_embeddings, dim=1)  # (B, sum(H*W), C)
        
        return memory_tokens


class EfficientMaskDownSampler(nn.Module):
    """
    Efficient implementation of the mask downsampler that progressively
    downsamples a mask and increases its channel capacity.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        kernel_size: int = 4,
        stride: int = 4,
        padding: int = 0,
        total_stride: int = 16,
        activation: nn.Module = nn.GELU,
    ):
        """
        Initialize the efficient mask downsampler.
        
        Args:
            embed_dim: Output embedding dimension
            kernel_size: Convolution kernel size
            stride: Stride for downsampling
            padding: Padding for convolution
            total_stride: Total downsampling factor
            activation: Activation function
        """
        super().__init__()
        
        import math
        
        # Compute number of downsampling layers needed
        num_layers = int(math.log2(total_stride) // math.log2(stride))
        assert stride**num_layers == total_stride, "total_stride must be a power of stride"
        
        # Create encoder layers
        self.encoder = nn.Sequential()
        mask_in_chans, mask_out_chans = 1, 1
        
        for i in range(num_layers):
            mask_out_chans = mask_in_chans * (stride**2)
            self.encoder.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    mask_in_chans,
                    mask_out_chans,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            
            # Add activation except after the last layer
            if i < num_layers - 1:
                self.encoder.add_module(f"act_{i}", activation())
            
            mask_in_chans = mask_out_chans
        
        # Final projection to embed_dim
        self.proj = nn.Sequential(
            nn.Conv2d(mask_out_chans, embed_dim, kernel_size=1),
            activation(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the efficient mask downsampler.
        
        Args:
            x: Input mask tensor of shape (B, 1, H, W)
            
        Returns:
            Downsampled mask embedding of shape (B, embed_dim, H', W')
        """
        # Apply encoder
        x = self.encoder(x)
        
        # Apply final projection
        x = self.proj(x)
        
        return x
