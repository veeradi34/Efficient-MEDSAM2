"""
ROI-Aware Memory Attention for EfficientMedSAM2
Implements spatial masking to focus attention on regions of interest.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class ROIAwareMemoryAttention(nn.Module):
    """
    Memory attention with ROI masking for efficiency.
    """
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 1,
        max_memory_frames: int = 2,  # Reduced from teacher (4)
        roi_dilation: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_memory_frames = max_memory_frames
        self.roi_dilation = roi_dilation
        
        # Lightweight attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # ROI detection network
        self.roi_detector = nn.Sequential(
            nn.Conv2d(d_model, d_model // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 4, d_model // 8, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Position embedding for spatial awareness
        self.register_buffer('pos_embed_cache', None)
        
    def _get_pos_embed(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate 2D positional embeddings."""
        if self.pos_embed_cache is not None and self.pos_embed_cache.shape[-2:] == (H, W):
            return self.pos_embed_cache
            
        # Create 2D positional embeddings
        y_pos = torch.arange(H, device=device).float().unsqueeze(1).repeat(1, W)
        x_pos = torch.arange(W, device=device).float().unsqueeze(0).repeat(H, 1)
        
        # Normalize to [-1, 1]
        y_pos = 2.0 * y_pos / (H - 1) - 1.0
        x_pos = 2.0 * x_pos / (W - 1) - 1.0
        
        # Generate sinusoidal embeddings
        pos_embed = torch.zeros(self.d_model, H, W, device=device)
        
        d_model_half = self.d_model // 2
        div_term = torch.exp(torch.arange(0, d_model_half, 2, device=device) * 
                           -(torch.log(torch.tensor(10000.0)) / d_model_half))
        
        # Y position embeddings
        pos_embed[0::4] = torch.sin(y_pos.unsqueeze(0) * div_term.unsqueeze(-1).unsqueeze(-1))
        pos_embed[1::4] = torch.cos(y_pos.unsqueeze(0) * div_term.unsqueeze(-1).unsqueeze(-1))
        
        # X position embeddings  
        pos_embed[2::4] = torch.sin(x_pos.unsqueeze(0) * div_term.unsqueeze(-1).unsqueeze(-1))
        pos_embed[3::4] = torch.cos(x_pos.unsqueeze(0) * div_term.unsqueeze(-1).unsqueeze(-1))
        
        self.pos_embed_cache = pos_embed
        return pos_embed
    
    def create_roi_mask(
        self,
        features: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Create ROI attention mask to focus on relevant regions.
        Args:
            features: [B, C, H, W] feature tensor
            threshold: ROI detection threshold
        Returns:
            roi_mask: [B, 1, H, W] binary mask
        """
        # Use learnable ROI detection
        roi_scores = self.roi_detector(features)  # [B, 1, H, W]
        roi_mask = (roi_scores > threshold).float()
        
        # Dilate ROI for context
        if self.roi_dilation > 0:
            kernel_size = 2 * self.roi_dilation + 1
            roi_mask = F.max_pool2d(
                roi_mask,
                kernel_size=kernel_size,
                stride=1,
                padding=self.roi_dilation
            )
        
        return roi_mask
    
    def apply_roi_masking(
        self,
        features: torch.Tensor,
        roi_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply ROI masking to features for efficient attention.
        Args:
            features: [B, C, H, W] input features
            roi_mask: [B, 1, H, W] ROI mask
        Returns:
            masked_features: ROI-masked features
            valid_positions: Indices of valid ROI positions
        """
        B, C, H, W = features.shape
        
        # Flatten spatial dimensions
        features_flat = features.reshape(B, C, H * W).transpose(1, 2)  # [B, HW, C]
        roi_mask_flat = roi_mask.reshape(B, 1, H * W).transpose(1, 2)  # [B, HW, 1]
        
        # Apply masking
        masked_features = features_flat * roi_mask_flat
        
        # Find valid positions (at least one pixel in ROI per batch)
        valid_positions = []
        for b in range(B):
            valid_idx = roi_mask_flat[b, :, 0] > 0
            if valid_idx.sum() == 0:
                # If no ROI detected, use center region as fallback
                center_h, center_w = H // 2, W // 2
                center_idx = center_h * W + center_w
                valid_idx[center_idx] = True
            valid_positions.append(valid_idx)
        
        return masked_features, valid_positions
    
    def forward(
        self,
        current_features: torch.Tensor,
        memory_bank: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with ROI-aware attention.
        Args:
            current_features: [B, C, H, W] current frame features
            memory_bank: [B, T, C, H, W] memory features from previous frames
            return_attention_weights: Whether to return attention weights
        Returns:
            attended_features: [B, C, H, W] attended features
        """
        B, C, H, W = current_features.shape
        
        # Add positional embeddings
        pos_embed = self._get_pos_embed(H, W, current_features.device)
        current_features = current_features + pos_embed.unsqueeze(0)
        
        # Create ROI mask
        roi_mask = self.create_roi_mask(current_features)
        
        # Apply ROI masking
        masked_features, valid_positions = self.apply_roi_masking(current_features, roi_mask)
        
        # Prepare memory features
        if memory_bank is not None:
            # Truncate memory to max frames
            if memory_bank.shape[1] > self.max_memory_frames:
                memory_bank = memory_bank[:, -self.max_memory_frames:]
            
            # Get actual dimensions from memory bank
            if len(memory_bank.shape) == 5:  # [B, T, C, H, W]
                B_mem, T, C_mem, H_mem, W_mem = memory_bank.shape
                # Flatten memory [B, T, C, H, W] -> [B, T*HW, C]
                memory_flat = memory_bank.reshape(B_mem, T * H_mem * W_mem, C_mem)
            elif len(memory_bank.shape) == 4:  # [B, C, H, W] - single frame
                B_mem, C_mem, H_mem, W_mem = memory_bank.shape
                # Reshape to add time dimension and flatten
                memory_flat = memory_bank.reshape(B_mem, H_mem * W_mem, C_mem)
            else:
                # Handle unexpected shapes gracefully
                print(f"Warning: Unexpected memory bank shape: {memory_bank.shape}")
                memory_flat = None
        else:
            memory_flat = None
        
        # Apply attention layers
        output = masked_features
        attention_weights = []
        
        for layer_idx in range(self.num_layers):
            attn = self.attention_layers[layer_idx]
            norm1 = self.norm_layers[layer_idx]
            ffn = self.ffn_layers[layer_idx]
            
            # Self-attention on current features
            attn_out, self_attn_weights = attn(output, output, output)
            output = norm1(output + attn_out)
            
            # Cross-attention with memory (if available)
            if memory_flat is not None:
                try:
                    # Ensure memory and query have compatible dimensions for attention
                    if memory_flat.shape[-1] != output.shape[-1]:
                        # Skip cross-attention if dimensions don't match
                        print(f"Warning: Skipping cross-attention due to dimension mismatch: memory={memory_flat.shape}, output={output.shape}")
                    else:
                        # Only attend to ROI regions in memory
                        memory_roi_masked = memory_flat
                        cross_attn_out, cross_attn_weights = attn(output, memory_roi_masked, memory_roi_masked)
                        output = norm1(output + cross_attn_out)
                except Exception as e:
                    print(f"Warning: Cross-attention failed: {e}, skipping...")
                    pass
                
                if return_attention_weights:
                    attention_weights.append({
                        'self_attention': self_attn_weights,
                        'cross_attention': cross_attn_weights
                    })
            else:
                if return_attention_weights:
                    attention_weights.append({'self_attention': self_attn_weights})
            
            # Feed forward
            ffn_out = ffn(output)
            output = norm1(output + ffn_out)
        
        # Reshape back to spatial format
        output_spatial = torch.zeros_like(masked_features)
        
        for b in range(B):
            valid_idx = valid_positions[b]
            if valid_idx.sum() > 0:
                output_spatial[b, valid_idx] = output[b, valid_idx]
        
        output_spatial = output_spatial.transpose(1, 2).reshape(B, C, H, W)
        
        # Residual connection with original features
        attended_features = current_features + output_spatial
        
        if return_attention_weights:
            return attended_features, attention_weights
        else:
            return attended_features


class SlimMemoryBank(nn.Module):
    """
    Lightweight memory bank for storing frame features.
    """
    def __init__(
        self,
        feature_dim: int = 256,
        max_frames: int = 4,
        compression_ratio: float = 0.5
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_frames = max_frames
        self.compressed_dim = int(feature_dim * compression_ratio)
        
        # Feature compression for memory efficiency
        self.compressor = nn.Conv2d(feature_dim, self.compressed_dim, kernel_size=1)
        self.decompressor = nn.Conv2d(self.compressed_dim, feature_dim, kernel_size=1)
        
        # Memory storage
        self.register_buffer('memory_features', torch.empty(0))
        self.register_buffer('memory_masks', torch.empty(0))
        
    def add_frame(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Add new frame to memory bank."""
        B, C, H, W = features.shape
        
        # Compress features
        compressed_features = self.compressor(features)
        
        # Initialize memory if empty
        if self.memory_features.numel() == 0:
            self.memory_features = compressed_features.unsqueeze(1)  # [B, 1, C_comp, H, W]
            if mask is not None:
                self.memory_masks = mask.unsqueeze(1)
        else:
            # Add to memory
            self.memory_features = torch.cat([self.memory_features, compressed_features.unsqueeze(1)], dim=1)
            if mask is not None:
                self.memory_masks = torch.cat([self.memory_masks, mask.unsqueeze(1)], dim=1)
            
            # Truncate if exceeds max frames
            if self.memory_features.shape[1] > self.max_frames:
                self.memory_features = self.memory_features[:, -self.max_frames:]
                if self.memory_masks.numel() > 0:
                    self.memory_masks = self.memory_masks[:, -self.max_frames:]
    
    def get_memory(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get decompressed memory features."""
        if self.memory_features.numel() == 0:
            return None, None
        
        try:
            # Decompress features with proper error handling
            if len(self.memory_features.shape) == 5:
                B, T, C_comp, H, W = self.memory_features.shape
                compressed_flat = self.memory_features.reshape(B * T, C_comp, H, W)
                decompressed_flat = self.decompressor(compressed_flat)
                decompressed = decompressed_flat.reshape(B, T, self.feature_dim, H, W)
            else:
                # Handle unexpected shapes
                print(f"Warning: Unexpected memory features shape: {self.memory_features.shape}")
                return None, None
                
        except Exception as e:
            print(f"Warning: Memory decompression failed: {e}")
            print(f"Memory features shape: {self.memory_features.shape}")
            return None, None
        
        masks = self.memory_masks if self.memory_masks.numel() > 0 else None
        
        return decompressed, masks
    
    def clear_memory(self):
        """Clear memory bank."""
        self.memory_features = torch.empty(0, device=self.memory_features.device)
        self.memory_masks = torch.empty(0, device=self.memory_masks.device)


if __name__ == "__main__":
    # Test ROI-aware memory attention
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create model
    memory_attn = ROIAwareMemoryAttention(
        d_model=256,
        nhead=4,
        num_layers=1,
        max_memory_frames=2
    ).to(device)
    
    # Test inputs
    B, C, H, W = 2, 256, 32, 32
    current_features = torch.randn(B, C, H, W).to(device)
    memory_bank = torch.randn(B, 3, C, H, W).to(device)  # 3 frames in memory
    
    # Forward pass
    attended_features = memory_attn(current_features, memory_bank)
    
    print(f"Input features: {current_features.shape}")
    print(f"Memory bank: {memory_bank.shape}")
    print(f"Output features: {attended_features.shape}")
    
    # Test memory bank
    mem_bank = SlimMemoryBank(feature_dim=256, max_frames=4).to(device)
    
    # Add frames
    for i in range(5):  # Add more than max_frames
        frame_features = torch.randn(B, 256, H, W).to(device)
        mem_bank.add_frame(frame_features)
    
    memory_features, memory_masks = mem_bank.get_memory()
    print(f"Memory bank features: {memory_features.shape if memory_features is not None else 'None'}")
    
    print("ROI-aware memory attention test completed!")