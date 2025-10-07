"""
Utility functions for EfficientMedSAM2

This module provides utility functions for the EfficientMedSAM2 model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional


def get_1d_sine_pe(
    embed_dim: int,
    num_positions: int,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Create 1D sine positional embeddings.
    
    Args:
        embed_dim: Embedding dimension
        num_positions: Number of positions
        scale: Scale factor
        
    Returns:
        Positional encoding of shape (1, num_positions, embed_dim)
    """
    # Half of the embedding uses sine, half uses cosine
    half_dim = embed_dim // 2
    
    # Create position indices
    position = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    
    # Create frequency tensor with logarithmic spacing
    div_term = torch.exp(
        torch.arange(0, half_dim, dtype=torch.float32) * (-math.log(10000.0) / half_dim)
    )
    
    # Compute sine and cosine components
    pe = torch.zeros(1, num_positions, embed_dim)
    pe[0, :, :half_dim] = torch.sin(position * div_term)
    pe[0, :, half_dim:2*half_dim] = torch.cos(position * div_term)
    
    # If embed_dim is odd, repeat the last element
    if embed_dim % 2 == 1:
        pe = torch.cat([pe, pe[:, :, -1:]], dim=2)
    
    return pe


def select_closest_cond_frames(
    curr_idx: int,
    cond_indices: List[int],
    max_frames: int,
) -> List[int]:
    """
    Select the temporally closest conditioning frames.
    
    Args:
        curr_idx: Current frame index
        cond_indices: List of conditioning frame indices
        max_frames: Maximum number of frames to select
        
    Returns:
        List of selected frame indices
    """
    # If there are fewer frames than the maximum, return all of them
    if len(cond_indices) <= max_frames:
        return cond_indices
    
    # Calculate temporal distance from current frame
    distances = [abs(curr_idx - idx) for idx in cond_indices]
    
    # Sort indices by distance
    sorted_indices = [idx for _, idx in sorted(zip(distances, cond_indices))]
    
    # Return the closest frames
    return sorted_indices[:max_frames]


class MLP(nn.Module):
    """Multi-layer perceptron implementation."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: nn.Module = nn.GELU(),
        dropout_rate: float = 0.1,
    ):
        """Initialize MLP."""
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(activation)
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(activation)
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers:
            x = layer(x)
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    This is a regularization technique used in vision transformers.
    """
    def __init__(self, drop_prob: float = 0.0):
        """Initialize DropPath."""
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def measure_memory_usage(func):
    """
    Decorator to measure GPU memory usage of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function that measures memory usage
    """
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Record memory before execution
            mem_before = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            
            # Run the function
            result = func(*args, **kwargs)
            
            # Record memory after execution
            mem_after = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            
            print(f"Function {func.__name__} used {mem_after - mem_before:.2f} MB of GPU memory")
            
            return result
        else:
            return func(*args, **kwargs)
    
    return wrapper


def apply_quantization(model, bit_width=8):
    """
    Apply post-training quantization to model weights.
    
    Args:
        model: PyTorch model to quantize
        bit_width: Bit width for quantization (default: 8)
        
    Returns:
        Quantized model
    """
    # This is a simplified implementation for demonstration
    # In practice, use PyTorch's quantization APIs
    
    import copy
    quantized_model = copy.deepcopy(model)
    
    # Get min and max for each parameter
    for name, param in quantized_model.named_parameters():
        if param.requires_grad:
            # Calculate quantization scale
            data_min = param.data.min()
            data_max = param.data.max()
            
            # Skip if min == max (constant tensor)
            if data_min == data_max:
                continue
            
            # Calculate scale and zero point
            scale = (data_max - data_min) / (2 ** bit_width - 1)
            zero_point = -round(data_min / scale)
            
            # Quantize
            param.data = torch.round(param.data / scale + zero_point) * scale - zero_point * scale
    
    return quantized_model


def calculate_param_count(model):
    """
    Calculate parameter count for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Tuple of (total_params, trainable_params) in millions
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params / 1e6, trainable_params / 1e6  # Convert to millions


class LayerNorm2d(nn.Module):
    """
    Layer normalization for 2D data (BatchNorm without tracking stats).
    """
    def __init__(self, num_channels: int, eps: float = 1e-6):
        """Initialize LayerNorm2d."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + self.eps
        x = (x - mean) / std
        x = x * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)
        return x
