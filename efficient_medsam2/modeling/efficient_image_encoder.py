"""
Efficient Image Encoder for EfficientMedSAM2

This module implements a memory-efficient image encoder using MobileNetV3 as the backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple


class EfficientImageEncoder(nn.Module):
    """
    Memory-efficient image encoder using MobileNetV3 as the backbone.
    Reduces parameter count by ~10x compared to the original SAM ViT encoder.
    """
    def __init__(
        self,
        image_size: int = 512,
        embed_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_checkpointing: bool = True,
    ):
        """
        Initialize the efficient image encoder.
        
        Args:
            image_size: Input image size (default: 512)
            embed_dim: Output embedding dimension (default: 256)
            pretrained: Whether to use pretrained weights for MobileNetV3
            freeze_backbone: Whether to freeze the backbone weights
            use_checkpointing: Whether to use gradient checkpointing to save memory
        """
        super().__init__()
        
        # Save parameters
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.use_checkpointing = use_checkpointing
        
        # Initialize MobileNetV3 backbone
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Extract feature dimensions from backbone
        backbone_dim = self.backbone.classifier[0].in_features
        
        # Remove classifier to use as feature extractor
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add projection to match expected embedding dimension
        self.proj = nn.Sequential(
            nn.Conv2d(backbone_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Output stride of MobileNetV3 is 32 by default
        self.output_stride = 32
                
        # Feature size after encoding
        self.feature_size = image_size // self.output_stride
        
        # Use FP16 for model parameters to save memory
        self.half_precision = True
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using the backbone"""
        # Apply gradient checkpointing if enabled
        if self.use_checkpointing and self.training:
            # Break the backbone into smaller chunks for checkpointing
            for module in self.backbone:
                if isinstance(module, nn.Sequential):
                    x = torch.utils.checkpoint.checkpoint(module, x)
                else:
                    x = module(x)
        else:
            x = self.backbone(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the efficient image encoder.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Image embeddings of shape (B, embed_dim, H', W')
        """
        # Save original precision
        original_dtype = x.dtype
        
        # Use FP16 if half precision is enabled
        if self.half_precision and x.dtype != torch.float16:
            x = x.half()
            
        # Get backbone features
        features = self.forward_features(x)
        
        # Reshape from [B, C, 1, 1] to [B, C, H', W']
        features = features.view(features.shape[0], features.shape[1], 
                                self.feature_size, self.feature_size)
        
        # Project to embedding dimension
        embeddings = self.proj(features)
        
        # Restore original precision if needed
        if original_dtype != torch.float16 and self.half_precision:
            embeddings = embeddings.to(original_dtype)
            
        return embeddings

# Alternative implementation using EfficientNet
class EfficientNetEncoder(nn.Module):
    """Alternative implementation using EfficientNet-B0 as backbone"""
    def __init__(
        self,
        image_size: int = 512,
        embed_dim: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.image_size = image_size
        self.embed_dim = embed_dim
        
        # Initialize EfficientNet backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Extract feature dimensions
        backbone_dim = self.backbone.classifier[1].in_features
        
        # Remove classifier
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add projection
        self.proj = nn.Conv2d(backbone_dim, embed_dim, kernel_size=1)
        
        # Output stride
        self.output_stride = 32
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.backbone(x)
        
        # Project to embedding dimension
        x = self.proj(x)
        
        return x
