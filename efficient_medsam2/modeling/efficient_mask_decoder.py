"""
Efficient Mask Decoder for EfficientMedSAM2

This module implements a memory-efficient version of the mask decoder
used in SAM/MedSAM2 for generating segmentation masks from embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Type, Optional

from torch.utils.checkpoint import checkpoint


class EfficientMaskDecoder(nn.Module):
    """
    Memory-efficient implementation of the mask decoder for generating
    segmentation masks from image embeddings and prompt embeddings.
    
    This decoder uses a simplified transformer and MLP architecture
    to reduce parameter count and memory usage.
    """
    def __init__(
        self,
        transformer_dim: int = 256,
        transformer_depth: int = 2,  # Reduced from original (typically 2-4 in SAM)
        transformer_nhead: int = 4,  # Reduced from original (typically 8 in SAM)
        transformer_mlp_dim: int = 512,  # Reduced from original (typically 2048 in SAM)
        transformer_dropout: float = 0.1,
        activation: Type[nn.Module] = nn.GELU,
        output_upscale: int = 4,  # Scale factor for upscaling masks (e.g. from 128x128 to 512x512)
        iou_head_depth: int = 2,  # Reduced from original (typically 3 in SAM)
        iou_head_hidden_dim: int = 128,  # Reduced from original (typically 256 in SAM)
        use_checkpointing: bool = True,
        multimask_output: bool = False,  # Set to False for memory efficiency
    ):
        """
        Initialize the efficient mask decoder.
        
        Args:
            transformer_dim: Transformer hidden dimension
            transformer_depth: Number of transformer layers
            transformer_nhead: Number of transformer attention heads
            transformer_mlp_dim: Transformer MLP dimension
            transformer_dropout: Transformer dropout rate
            activation: Activation function
            output_upscale: Scale factor for upscaling output masks
            iou_head_depth: Depth of the IoU prediction head
            iou_head_hidden_dim: Hidden dimension of the IoU prediction head
            use_checkpointing: Whether to use gradient checkpointing
            multimask_output: Whether to output multiple masks
        """
        super().__init__()
        
        # Save parameters
        self.transformer_dim = transformer_dim
        self.use_checkpointing = use_checkpointing
        self.multimask_output = multimask_output
        self.num_mask_outputs = 3 if multimask_output else 1
        
        # Create transformer decoder
        self.transformer = EfficientTwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=transformer_dim,
            mlp_dim=transformer_mlp_dim,
            num_heads=transformer_nhead,
            dropout=transformer_dropout,
        )
        
        # Create output token (learns to query the transformer to generate a mask)
        self.output_tokens = nn.Parameter(torch.zeros(1, self.num_mask_outputs, transformer_dim))
        nn.init.normal_(self.output_tokens, std=0.02)
        
        # Create upscaling layers for the mask
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 2),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 2, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Conv2d(transformer_dim // 4, 1, kernel_size=1),
        )
        
        # Create IoU prediction head (predicts mask quality)
        self.iou_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        nn.init.normal_(self.iou_token, std=0.02)
        
        self.iou_prediction_head = MLP(
            input_dim=transformer_dim,
            hidden_dim=iou_head_hidden_dim,
            output_dim=self.num_mask_outputs,
            num_layers=iou_head_depth,
            activation=activation,
        )
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the efficient mask decoder.
        
        Args:
            image_embeddings: Image embeddings from the image encoder (B, C, H, W)
            image_pe: Positional encodings for the image embeddings (B, N, C) or (1, N, C)
            sparse_prompt_embeddings: Sparse prompt embeddings (points, boxes) (B, P, C)
            dense_prompt_embeddings: Dense prompt embeddings (masks) (B, N, C)
            multimask_output: Whether to return multiple mask predictions
            
        Returns:
            - Predicted masks (B, num_masks, H, W)
            - IoU predictions (B, num_masks)
        """
        # Force multimask_output to False if the model doesn't support it
        multimask_output = multimask_output and self.multimask_output
        num_masks = 3 if multimask_output else 1
        
        # Get batch size and embedding dimensions
        batch_size = image_embeddings.shape[0]
        h, w = image_embeddings.shape[-2:]
        
        # Prepare tokens for transformer
        output_tokens = self.output_tokens.expand(batch_size, -1, -1)
        iou_token = self.iou_token.expand(batch_size, -1, -1)
        
        # Add IoU token to output tokens
        tokens = torch.cat([output_tokens, iou_token], dim=1)
        
        # Prepare image embeddings for transformer
        image_embeddings_flattened = image_embeddings.flatten(2).transpose(1, 2)
        
        # Apply transformer
        if self.use_checkpointing and self.training:
            # Use gradient checkpointing to save memory
            tokens = checkpoint(
                self.transformer,
                tokens, 
                image_embeddings_flattened,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                image_pe,
            )
        else:
            tokens = self.transformer(
                tokens, 
                image_embeddings_flattened,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                image_pe,
            )
        
        # Split output and iou tokens
        output_tokens, iou_token = tokens[:, :self.num_mask_outputs], tokens[:, -1:]
        
        # Generate IoU predictions
        iou_pred = self.iou_prediction_head(iou_token)
        
        # Select output tokens based on multimask_output
        if multimask_output:
            # Use all output tokens
            selected_tokens = output_tokens
        else:
            # Use only the first output token
            selected_tokens = output_tokens[:, :1]
        
        # Reshape and expand tokens to match spatial dimensions
        selected_tokens = selected_tokens.reshape(batch_size * num_masks, self.transformer_dim, 1, 1)
        selected_tokens = selected_tokens.expand(-1, -1, h, w)
        
        # Generate mask logits
        mask_logits = self.output_upscaling(selected_tokens)
        mask_logits = mask_logits.reshape(batch_size, num_masks, mask_logits.shape[-2], mask_logits.shape[-1])
        
        # Reshape IoU predictions
        iou_pred = iou_pred.reshape(batch_size, num_masks)
        
        return mask_logits, iou_pred


class EfficientTwoWayTransformer(nn.Module):
    """
    Efficient implementation of the two-way transformer used in the mask decoder.
    """
    def __init__(
        self,
        depth: int = 2,
        embedding_dim: int = 256,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize the efficient two-way transformer.
        
        Args:
            depth: Number of transformer layers
            embedding_dim: Transformer hidden dimension
            num_heads: Number of attention heads
            mlp_dim: MLP dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Save parameters
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        
        # Create transformer layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                EfficientTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                )
            )
        
        # Final layer normalization
        self.norm_tokens = nn.LayerNorm(embedding_dim)
        self.norm_image = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        tokens: torch.Tensor,
        image_embedding: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the efficient two-way transformer.
        
        Args:
            tokens: Output tokens of shape (B, N_t, C)
            image_embedding: Image embeddings of shape (B, N_i, C)
            sparse_prompt_embeddings: Sparse prompt embeddings (B, N_s, C)
            dense_prompt_embeddings: Dense prompt embeddings (B, N_d, C)
            image_pe: Positional encodings for the image embeddings (B, N_i, C) or (1, N_i, C)
            
        Returns:
            Updated tokens of shape (B, N_t, C)
        """
        # If no sparse prompt embeddings are provided, create empty tensor
        if sparse_prompt_embeddings.shape[1] == 0:
            sparse_prompt_embeddings = torch.zeros(
                (image_embedding.shape[0], 0, self.embedding_dim),
                device=image_embedding.device,
            )
        
        # Extract batch size
        batch_size = image_embedding.shape[0]
        
        # Expand image_pe if needed
        if image_pe.shape[0] != batch_size:
            image_pe = image_pe.expand(batch_size, -1, -1)
        
        # Add positional encoding to image embedding
        image_embedding = image_embedding + image_pe
        
        # Apply transformer layers
        for layer in self.layers:
            tokens, image_embedding = layer(
                tokens=tokens,
                image_embedding=image_embedding,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )
        
        # Apply layer normalization
        tokens = self.norm_tokens(tokens)
        image_embedding = self.norm_image(image_embedding)
        
        return tokens


class EfficientTwoWayAttentionBlock(nn.Module):
    """
    Efficient implementation of the two-way attention block used in the transformer.
    """
    def __init__(
        self,
        embedding_dim: int = 256,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.1,
    ):
        """
        Initialize the efficient two-way attention block.
        
        Args:
            embedding_dim: Transformer hidden dimension
            num_heads: Number of attention heads
            mlp_dim: MLP dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Save parameters
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        # Self-attention for tokens
        self.self_attn_tokens = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Self-attention for image embeddings
        self.self_attn_image = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention from tokens to image
        self.cross_attn_tokens_to_image = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention from image to tokens
        self.cross_attn_image_to_tokens = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # MLP for tokens
        self.mlp_tokens = MLP(
            input_dim=embedding_dim,
            hidden_dim=mlp_dim,
            output_dim=embedding_dim,
            num_layers=2,
            activation=nn.GELU,
        )
        
        # MLP for image embeddings
        self.mlp_image = MLP(
            input_dim=embedding_dim,
            hidden_dim=mlp_dim,
            output_dim=embedding_dim,
            num_layers=2,
            activation=nn.GELU,
        )
        
        # Layer normalization
        self.norm1_tokens = nn.LayerNorm(embedding_dim)
        self.norm2_tokens = nn.LayerNorm(embedding_dim)
        self.norm3_tokens = nn.LayerNorm(embedding_dim)
        self.norm4_tokens = nn.LayerNorm(embedding_dim)
        
        self.norm1_image = nn.LayerNorm(embedding_dim)
        self.norm2_image = nn.LayerNorm(embedding_dim)
        self.norm3_image = nn.LayerNorm(embedding_dim)
        self.norm4_image = nn.LayerNorm(embedding_dim)
    
    def forward(
        self,
        tokens: torch.Tensor,
        image_embedding: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the efficient two-way attention block.
        
        Args:
            tokens: Output tokens of shape (B, N_t, C)
            image_embedding: Image embeddings of shape (B, N_i, C)
            sparse_prompt_embeddings: Sparse prompt embeddings (B, N_s, C)
            dense_prompt_embeddings: Dense prompt embeddings (B, N_d, C)
            
        Returns:
            - Updated tokens of shape (B, N_t, C)
            - Updated image embeddings of shape (B, N_i, C)
        """
        # Self-attention for tokens
        tokens_norm = self.norm1_tokens(tokens)
        tokens = tokens + self.self_attn_tokens(
            query=tokens_norm,
            key=tokens_norm,
            value=tokens_norm,
            need_weights=False,
        )[0]
        
        # Self-attention for image embeddings
        image_norm = self.norm1_image(image_embedding)
        image_embedding = image_embedding + self.self_attn_image(
            query=image_norm,
            key=image_norm,
            value=image_norm,
            need_weights=False,
        )[0]
        
        # Cross-attention from tokens to image
        tokens_norm = self.norm2_tokens(tokens)
        tokens = tokens + self.cross_attn_tokens_to_image(
            query=tokens_norm,
            key=torch.cat([image_embedding, sparse_prompt_embeddings, dense_prompt_embeddings], dim=1),
            value=torch.cat([image_embedding, sparse_prompt_embeddings, dense_prompt_embeddings], dim=1),
            need_weights=False,
        )[0]
        
        # Cross-attention from image to tokens
        image_norm = self.norm2_image(image_embedding)
        image_embedding = image_embedding + self.cross_attn_image_to_tokens(
            query=image_norm,
            key=tokens,
            value=tokens,
            need_weights=False,
        )[0]
        
        # MLP for tokens
        tokens_norm = self.norm3_tokens(tokens)
        tokens = tokens + self.mlp_tokens(tokens_norm)
        
        # MLP for image embeddings
        image_norm = self.norm3_image(image_embedding)
        image_embedding = image_embedding + self.mlp_image(image_norm)
        
        return tokens, image_embedding


class MLP(nn.Module):
    """
    Simple multi-layer perceptron implementation.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: Type[nn.Module] = nn.GELU,
        dropout: float = 0.1,
    ):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
            activation: Activation function
            dropout: Dropout rate
        """
        super().__init__()
        
        # Create layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim -> hidden_dim
                self.layers.append(nn.Linear(input_dim, hidden_dim))
                self.layers.append(activation())
                self.layers.append(nn.Dropout(dropout))
            elif i == num_layers - 1:
                # Last layer: hidden_dim -> output_dim
                self.layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                # Hidden layer: hidden_dim -> hidden_dim
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(activation())
                self.layers.append(nn.Dropout(dropout))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP."""
        for layer in self.layers:
            x = layer(x)
        return x


class LayerNorm2d(nn.Module):
    """Layer normalization for 2D data."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        """
        Initialize the 2D layer normalization.
        
        Args:
            num_channels: Number of input channels
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        # Create parameters for normalization
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for 2D layer normalization."""
        # x: (B, C, H, W)
        
        # Calculate mean and variance along spatial dimensions
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], unbiased=False, keepdim=True)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply scale and shift
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        
        return x
