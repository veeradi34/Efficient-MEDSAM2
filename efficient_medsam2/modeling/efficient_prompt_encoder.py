"""
Efficient Prompt Encoder for EfficientMedSAM2

This module implements a memory-efficient version of the prompt encoder
used in SAM/MedSAM2 for encoding user prompts (points, boxes, masks).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Type, List

import numpy as np


class EfficientPromptEncoder(nn.Module):
    """
    Memory-efficient implementation of the prompt encoder for handling
    user prompts (points, boxes, masks) in medical image segmentation.
    
    The prompt encoder is already relatively small in the original SAM architecture,
    so we maintain most of its structure while making some efficiency improvements.
    """
    def __init__(
        self,
        embed_dim: int = 256,
        image_embedding_size: Tuple[int, int] = (32, 32),  # For 512x512 input with stride=16
        input_image_size: Tuple[int, int] = (512, 512),
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
    ):
        """
        Initialize the efficient prompt encoder.
        
        Args:
            embed_dim: Embedding dimension
            image_embedding_size: Size of the image embedding (H, W)
            input_image_size: Size of the input image (H, W)
            mask_in_chans: Number of hidden channels for mask encoding
            activation: Activation function
        """
        super().__init__()
        
        # Save parameters
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        self.input_image_size = input_image_size
        self.mask_in_chans = mask_in_chans
        
        # Position embedding for encoding point prompts
        self.pe_layer = EfficientPositionEmbedding(embed_dim // 2)
        
        # Number of point embeddings: positive point, negative point, 2 box corners
        self.num_point_embeddings: int = 4
        
        # Create point embeddings (these are learned)
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        
        # Mask encoder for mask prompts
        self.mask_encoder = EfficientMaskEncoder(
            embed_dim=embed_dim,
            mask_in_chans=mask_in_chans,
            activation=activation,
        )
        
        # No mask token is used when no mask is provided
        self.no_mask_embed = nn.Parameter(torch.randn(1, embed_dim))
        
        # Calculate position encoding grid for dense positional embeddings
        self._register_positional_grid()
    
    def _register_positional_grid(self):
        """Pre-compute the positional encoding grid"""
        # Create a grid of (x, y) coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1, self.image_embedding_size[0], dtype=torch.float32),
            torch.linspace(0, 1, self.image_embedding_size[1], dtype=torch.float32),
            indexing="ij",
        )
        
        # Reshape to [H*W, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        
        # Register as a buffer (not a parameter)
        self.register_buffer("grid", grid)
        
        # Initialize dense_pe as None (will be computed on first use)
        self.dense_pe = None
    
    def get_dense_pe(self) -> torch.Tensor:
        """
        Return the dense positional encoding used to encode the positions in the image.
        
        Returns:
            Positional encoding with shape (1, H*W, embed_dim)
        """
        if self.dense_pe is None or self.dense_pe.device != self.grid.device:
            # Compute dense positional embedding
            self.dense_pe = self.pe_layer(self.grid.unsqueeze(0))  # (1, H*W, embed_dim//2)
            
            # Duplicate to match embed_dim
            self.dense_pe = self.dense_pe.repeat_interleave(2, dim=-1)  # (1, H*W, embed_dim)
        
        return self.dense_pe
    
    def _embed_points(
        self,
        points: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        pad: bool = False,
    ) -> torch.Tensor:
        """
        Embed point prompts.
        
        Args:
            points: Point coordinates of shape (B, N, 2)
            labels: Point labels of shape (B, N), where 1 is a foreground point and 0 is a background point
            pad: Whether to pad the points to a fixed number
            
        Returns:
            Point embeddings of shape (B, N, embed_dim)
        """
        if points is None:
            return torch.zeros(0, self.embed_dim, device=self.no_mask_embed.device)
        
        # Get batch and point dimensions
        batch_size, num_points, _ = points.shape
        
        # Normalize point coordinates to [0, 1]
        points = points / torch.tensor(
            self.input_image_size, device=points.device
        ).reshape(1, 1, 2)
        
        # Compute point embeddings using the PE layer
        point_embedding = self.pe_layer(points)
        
        # Double the point embedding dimension (matching embed_dim)
        point_embedding = point_embedding.repeat_interleave(2, dim=-1)
        
        # Add learned embedding based on point label (foreground/background)
        if labels is not None:
            # Convert labels to indices (0 or 1)
            label_indices = (labels > 0).long()
            
            # Get the appropriate point embedding (pos or neg)
            for i in range(batch_size):
                for j in range(num_points):
                    # Add the appropriate embedding based on the label
                    point_embedding[i, j] += self.point_embeddings[label_indices[i, j]](
                        torch.zeros(1, dtype=torch.long, device=points.device)
                    )
        
        return point_embedding
    
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embed box prompts.
        
        Args:
            boxes: Box coordinates of shape (B, 4) in format [x1, y1, x2, y2]
            
        Returns:
            Box embeddings of shape (B, 2, embed_dim)
        """
        # Get batch size
        batch_size = boxes.shape[0]
        
        # Normalize box coordinates to [0, 1]
        boxes = boxes / torch.tensor(
            self.input_image_size + self.input_image_size, device=boxes.device
        ).reshape(1, 4)
        
        # Extract corner points
        corner_points = torch.zeros(batch_size, 2, 2, device=boxes.device)
        corner_points[:, 0] = boxes[:, :2]  # Top-left corner
        corner_points[:, 1] = boxes[:, 2:]  # Bottom-right corner
        
        # Compute corner embeddings using the PE layer
        corner_embedding = self.pe_layer(corner_points)
        
        # Double the embedding dimension (matching embed_dim)
        corner_embedding = corner_embedding.repeat_interleave(2, dim=-1)
        
        # Add learned embedding for box corners
        for i in range(batch_size):
            corner_embedding[i, 0] += self.point_embeddings[2](
                torch.zeros(1, dtype=torch.long, device=boxes.device)
            )
            corner_embedding[i, 1] += self.point_embeddings[3](
                torch.zeros(1, dtype=torch.long, device=boxes.device)
            )
        
        return corner_embedding
    
    def forward(
        self,
        points: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the efficient prompt encoder.
        
        Args:
            points: Point coordinates of shape (B, N, 2) and their labels (B, N)
            boxes: Box coordinates of shape (B, 4) in format [x1, y1, x2, y2]
            masks: Mask inputs of shape (B, 1, H, W)
            
        Returns:
            - sparse_embeddings: Embeddings for points and boxes (B, N, embed_dim)
            - dense_embeddings: Embeddings for masks (B, H*W, embed_dim)
        """
        # Process points if provided
        if points is not None:
            point_coords, point_labels = points
            point_embeddings = self._embed_points(point_coords, point_labels)
        else:
            point_embeddings = torch.empty(0, self.embed_dim, device=self.no_mask_embed.device)
        
        # Process boxes if provided
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
        else:
            box_embeddings = torch.empty(0, self.embed_dim, device=self.no_mask_embed.device)
        
        # Combine point and box embeddings (sparse embeddings)
        if point_embeddings.shape[0] > 0 and box_embeddings.shape[0] > 0:
            # Make sure batch dimensions match
            assert point_embeddings.shape[0] == box_embeddings.shape[0], "Batch sizes must match"
            sparse_embeddings = torch.cat([point_embeddings, box_embeddings], dim=1)
        elif point_embeddings.shape[0] > 0:
            sparse_embeddings = point_embeddings
        elif box_embeddings.shape[0] > 0:
            sparse_embeddings = box_embeddings
        else:
            sparse_embeddings = torch.empty(0, 0, self.embed_dim, device=self.no_mask_embed.device)
        
        # Process masks if provided
        if masks is not None:
            dense_embeddings = self.mask_encoder(masks)
        else:
            # If no mask is provided, use the no-mask embedding
            batch_size = (
                point_embeddings.shape[0] if point_embeddings.shape[0] > 0 
                else box_embeddings.shape[0] if box_embeddings.shape[0] > 0 
                else 1
            )
            
            # Expand no_mask_embed to match batch size and spatial dimensions
            dense_embeddings = self.no_mask_embed.expand(
                batch_size, self.image_embedding_size[0] * self.image_embedding_size[1], -1
            )
        
        return sparse_embeddings, dense_embeddings


class EfficientMaskEncoder(nn.Module):
    """Efficient encoder for mask inputs."""
    def __init__(
        self,
        embed_dim: int = 256,
        mask_in_chans: int = 16,
        activation: Type[nn.Module] = nn.GELU,
    ):
        """
        Initialize the efficient mask encoder.
        
        Args:
            embed_dim: Output embedding dimension
            mask_in_chans: Number of hidden channels
            activation: Activation function
        """
        super().__init__()
        
        # Create a small convolutional network for mask encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 2, kernel_size=3, padding=1),
            activation(),
            nn.Conv2d(mask_in_chans // 2, mask_in_chans, kernel_size=3, padding=1),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
    
    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the mask encoder.
        
        Args:
            masks: Input masks of shape (B, 1, H, W)
            
        Returns:
            Mask embeddings of shape (B, H*W, embed_dim)
        """
        # Encode the masks
        encoded_masks = self.encoder(masks)  # (B, embed_dim, H, W)
        
        # Reshape to (B, embed_dim, H*W) and transpose to (B, H*W, embed_dim)
        B, E, H, W = encoded_masks.shape
        encoded_masks = encoded_masks.reshape(B, E, H * W).transpose(1, 2)
        
        return encoded_masks


class EfficientPositionEmbedding(nn.Module):
    """
    Efficient implementation of random Fourier position embeddings.
    """
    def __init__(self, embed_dim: int = 128, scale: float = 1.0):
        """
        Initialize the efficient position embedding.
        
        Args:
            embed_dim: Embedding dimension (half of the final dimension)
            scale: Scale factor for the embedding
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.scale = scale
        
        # Initialize random frequencies
        freqs = torch.randn(embed_dim // 2, 2) * scale
        
        # Register as a buffer (not a parameter)
        self.register_buffer("freqs", freqs)
        
        # Pre-compute scaling for more efficient forward pass
        self.register_buffer("scale_factor", torch.tensor(2 * np.pi))
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute position embeddings.
        
        Args:
            positions: Input positions of shape (B, N, 2)
            
        Returns:
            Position embeddings of shape (B, N, embed_dim)
        """
        # Compute the inner product between positions and frequencies
        # positions: (B, N, 2), freqs: (embed_dim/2, 2)
        # -> (B, N, embed_dim/2)
        projection = torch.einsum("bnd,md->bnm", positions, self.freqs)
        
        # Scale the projection
        projection = projection * self.scale_factor
        
        # Compute sine and cosine components
        sin_embedding = torch.sin(projection)  # (B, N, embed_dim/2)
        cos_embedding = torch.cos(projection)  # (B, N, embed_dim/2)
        
        # Concatenate sine and cosine components
        embedding = torch.cat([sin_embedding, cos_embedding], dim=-1)  # (B, N, embed_dim)
        
        return embedding
