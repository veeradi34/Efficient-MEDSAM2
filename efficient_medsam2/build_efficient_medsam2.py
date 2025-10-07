"""
EfficientMedSAM2 Model Builder

This module provides functions for building and loading EfficientMedSAM2 models.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os

from .efficient_medsam2_base import EfficientMedSAM2
from .modeling.efficient_image_encoder import EfficientImageEncoder, EfficientNetEncoder
from .modeling.efficient_memory_attention import EfficientMemoryAttentionModule
from .modeling.efficient_memory_encoder import EfficientMemoryEncoder
from .modeling.efficient_prompt_encoder import EfficientPromptEncoder
from .modeling.efficient_mask_decoder import EfficientMaskDecoder


def build_efficient_medsam2_model(
    encoder_type: str = "mobilenet",  # "mobilenet" or "efficientnet"
    checkpoint: Optional[str] = None,  # Path to checkpoint file
    image_size: int = 512,
    embed_dim: int = 256,
    use_half_precision: bool = True,
    max_num_memory_frames: int = 3,  # Maximum number of frames to store in memory
    device: Optional[str] = None,  # Device to load the model to
) -> EfficientMedSAM2:
    """
    Build an EfficientMedSAM2 model.
    
    Args:
        encoder_type: Type of image encoder to use ("mobilenet" or "efficientnet")
        checkpoint: Path to checkpoint file
        image_size: Input image size
        embed_dim: Embedding dimension
        use_half_precision: Whether to use half precision (FP16)
        max_num_memory_frames: Maximum number of frames to store in memory
        device: Device to load the model to
        
    Returns:
        EfficientMedSAM2 model
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Create image encoder
    if encoder_type == "mobilenet":
        image_encoder = EfficientImageEncoder(
            image_size=image_size,
            embed_dim=embed_dim,
            pretrained=True,
            use_checkpointing=True,
        )
    elif encoder_type == "efficientnet":
        image_encoder = EfficientNetEncoder(
            image_size=image_size,
            embed_dim=embed_dim,
            pretrained=True,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    # Create memory attention module
    memory_attention = EfficientMemoryAttentionModule(
        d_model=embed_dim,
        nhead=4,  # Reduced from 8
        dim_feedforward=512,  # Reduced from 2048
        dropout=0.1,
        num_layers=1,  # Reduced from 4
        use_checkpoint=True,
    )
    
    # Create memory encoder
    memory_encoder = EfficientMemoryEncoder(
        embed_dim=embed_dim,
        mask_in_chans=1,
        activation=nn.GELU,
        stride=4,
        total_stride=16,
    )
    
    # Create prompt encoder
    image_embedding_size = (image_size // 16, image_size // 16)  # For stride=16
    prompt_encoder = EfficientPromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=(image_size, image_size),
        mask_in_chans=16,  # Reduced from 64
        activation=nn.GELU,
    )
    
    # Create mask decoder
    mask_decoder = EfficientMaskDecoder(
        transformer_dim=embed_dim,
        transformer_depth=2,  # Reduced from 3-4
        transformer_nhead=4,  # Reduced from 8
        transformer_mlp_dim=512,  # Reduced from 2048
        transformer_dropout=0.1,
        activation=nn.GELU,
        output_upscale=4,
        iou_head_depth=2,  # Reduced from 3
        iou_head_hidden_dim=128,  # Reduced from 256
        use_checkpointing=True,
        multimask_output=False,  # Set to False for memory efficiency
    )
    
    # Create the model
    model = EfficientMedSAM2(
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,
        prompt_encoder=prompt_encoder,
        mask_decoder=mask_decoder,
        num_maskmem=max_num_memory_frames,
        image_size=image_size,
        backbone_stride=16,
        use_half_precision=use_half_precision,
    )
    
    # Load checkpoint if provided
    if checkpoint is not None:
        if os.path.exists(checkpoint):
            print(f"Loading checkpoint from {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if isinstance(state_dict, dict) and "model_state" in state_dict:
                model.load_state_dict(state_dict["model_state"])
            else:
                model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint file {checkpoint} not found")
    
    # Move model to device
    model.to(device)
    
    # Use half precision if enabled
    if use_half_precision and device.type == "cuda":
        model.half()
    
    return model


def save_efficient_medsam2_model(
    model: EfficientMedSAM2,
    path: str,
    save_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    additional_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save an EfficientMedSAM2 model checkpoint.
    
    Args:
        model: EfficientMedSAM2 model to save
        path: Path to save the checkpoint
        save_optimizer: Whether to save optimizer state
        optimizer: Optimizer to save
        epoch: Current epoch number
        loss: Current loss value
        additional_data: Additional data to save
    """
    # Create checkpoint dictionary
    checkpoint = {
        "model_state": model.state_dict(),
    }
    
    # Add optimizer state if requested
    if save_optimizer and optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    
    # Add other metadata
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if loss is not None:
        checkpoint["loss"] = loss
    
    # Add additional data
    if additional_data is not None:
        checkpoint.update(additional_data)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Save the checkpoint
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def compare_model_sizes(original_model, efficient_model):
    """
    Compare the size and memory usage of the original and efficient models.
    
    Args:
        original_model: Original MedSAM2 model
        efficient_model: EfficientMedSAM2 model
        
    Returns:
        Dictionary with comparison metrics
    """
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    original_params = count_parameters(original_model)
    efficient_params = count_parameters(efficient_model)
    
    # Calculate parameter reduction
    param_reduction = (original_params - efficient_params) / original_params
    
    # Measure memory usage (if on CUDA)
    memory_reduction = None
    if torch.cuda.is_available():
        # Original model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        original_model.cuda()
        
        # Run a dummy forward pass to measure memory
        batch_size, time_dim, channels, height, width = 1, 1, 3, 512, 512
        dummy_input = torch.randn(batch_size, time_dim, channels, height, width).cuda()
        
        with torch.no_grad():
            original_model({"images": dummy_input})
        
        original_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        # Efficient model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        efficient_model.cuda()
        
        with torch.no_grad():
            efficient_model({"images": dummy_input})
        
        efficient_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        # Calculate memory reduction
        memory_reduction = (original_memory - efficient_memory) / original_memory
    
    # Return comparison metrics
    return {
        "original_params": original_params,
        "efficient_params": efficient_params,
        "param_reduction": param_reduction,
        "param_reduction_percentage": param_reduction * 100,
        "memory_reduction": memory_reduction,
        "memory_reduction_percentage": memory_reduction * 100 if memory_reduction is not None else None,
    }
