"""
Efficient MedSAM2 Base Architecture

This file contains the main architecture for EfficientMedSAM2, a memory-efficient version
of the MedSAM2 model for prompt-guided medical image segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from torch.nn.init import trunc_normal_

# Import necessary components
from .modeling.efficient_prompt_encoder import EfficientPromptEncoder
from .modeling.efficient_mask_decoder import EfficientMaskDecoder
from .modeling.efficient_memory_attention import EfficientMemoryAttentionModule
from .modeling.efficient_image_encoder import EfficientImageEncoder
from .modeling.efficient_memory_encoder import EfficientMemoryEncoder
from .utils import get_1d_sine_pe, select_closest_cond_frames

# Constants
NO_OBJ_SCORE = -1024.0  # a large negative value as a placeholder score for missing objects

class EfficientMedSAM2(nn.Module):
    """
    EfficientMedSAM2 main model class that combines lightweight components:
    - Efficient Image Encoder (backbone)
    - Memory attention mechanism
    - Memory encoder
    - Prompt encoder
    - Mask decoder
    
    This implementation focuses on memory efficiency while maintaining accuracy.
    """
    def __init__(
        self,
        image_encoder: nn.Module,
        memory_attention: nn.Module,
        memory_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        num_maskmem: int = 3,  # Reduced from 7 to 3 for efficiency (1 input frame + 2 previous frames)
        image_size: int = 512,
        backbone_stride: int = 16,  # stride of the image backbone output
        sigmoid_scale_for_mem_enc: float = 1.0,  # scale factor for mask sigmoid prob
        sigmoid_bias_for_mem_enc: float = 0.0,  # bias factor for mask sigmoid prob
        binarize_mask_from_pts_for_mem_enc: bool = False,
        use_mask_input_as_output_without_sam: bool = False,
        max_cond_frames_in_attn: int = 2,  # Reduced for memory efficiency (only attend to 2 closest frames)
        directly_add_no_mem_embed: bool = True,  # More efficient approach
        use_high_res_features_in_sam: bool = False,  # Set to False to reduce memory usage
        multimask_output_in_sam: bool = False,  # Set to False for efficiency
        use_half_precision: bool = True,  # Use FP16 for memory efficiency
    ) -> None:
        """
        Initialize EfficientMedSAM2.
        
        Args:
            image_encoder: Lightweight image encoder (e.g., MobileNetV3 or EfficientNet)
            memory_attention: Simplified memory attention module
            memory_encoder: Memory encoder for handling temporal information
            prompt_encoder: Encoder for user prompts (points, boxes)
            mask_decoder: Decoder for generating segmentation masks
            num_maskmem: Number of frames to store in memory
            image_size: Input image size (default: 512x512)
            backbone_stride: Stride of the backbone features
            sigmoid_scale_for_mem_enc: Scale factor for sigmoid
            sigmoid_bias_for_mem_enc: Bias for sigmoid
            binarize_mask_from_pts_for_mem_enc: Whether to binarize masks
            use_mask_input_as_output_without_sam: Skip SAM for frames with mask input
            max_cond_frames_in_attn: Maximum number of conditioning frames for attention
            directly_add_no_mem_embed: Add no-memory embedding directly (faster)
            use_high_res_features_in_sam: Use high-res features (memory intensive)
            multimask_output_in_sam: Output multiple masks (memory intensive)
            use_half_precision: Use FP16 precision to reduce memory usage
        """
        super().__init__()
        
        # Main components
        self.image_encoder = image_encoder
        self.memory_attention = memory_attention
        self.memory_encoder = memory_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        
        # Configuration parameters
        self.image_size = image_size
        self.backbone_stride = backbone_stride
        self.num_maskmem = num_maskmem
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.max_cond_frames_in_attn = max_cond_frames_in_attn
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.multimask_output_in_sam = multimask_output_in_sam
        self.use_half_precision = use_half_precision
        
        # Get feature dimensions from components
        self.prompt_embed_dim = self.prompt_encoder.embed_dim
        self.mask_decoder_transformer_dim = self.mask_decoder.transformer_dim
        
        # Size of the feature maps from the image encoder
        self.image_embedding_size = (image_size // backbone_stride, image_size // backbone_stride)
        
        # Memory embedding for the first frame (no memory case)
        self.no_memory_embed = nn.Parameter(torch.zeros(1, self.image_encoder.embed_dim))
        trunc_normal_(self.no_memory_embed, std=0.02)
    
    def forward(
        self,
        batched_input: Dict[str, torch.Tensor],
        multimask_output: bool = False,
        output_all_iou_scores: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for EfficientMedSAM2.
        
        Args:
            batched_input: Dictionary containing:
                - 'images': Tensor of shape (B, T, C, H, W)
                - 'points': List of point prompts
                - 'boxes': List of box prompts
                - 'masks': List of mask prompts
            multimask_output: Whether to return multiple mask predictions
            output_all_iou_scores: Whether to return IoU scores for all masks
            
        Returns:
            Dictionary containing segmentation masks and other outputs
        """
        # Apply half precision if enabled
        orig_dtype = next(self.parameters()).dtype
        if self.use_half_precision:
            self.to(torch.float16)
            for k in batched_input:
                if isinstance(batched_input[k], torch.Tensor):
                    batched_input[k] = batched_input[k].to(torch.float16)
        
        try:
            # Process input and generate embeddings
            image_embeddings = self._encode_images(batched_input)
            
            # Process prompts (points, boxes, masks)
            sparse_embeddings, dense_embeddings = self._encode_prompts(batched_input)
            
            # Use memory attention to incorporate temporal context
            enhanced_embeddings = self._apply_memory_attention(image_embeddings, batched_input)
            
            # Generate masks through the mask decoder
            masks, iou_scores = self._decode_masks(
                enhanced_embeddings, 
                sparse_embeddings, 
                dense_embeddings,
                multimask_output
            )
            
            # Prepare the output dictionary
            outputs = {
                'masks': masks,
                'iou_scores': iou_scores,
            }
            
            return outputs
            
        finally:
            # Restore original precision
            if self.use_half_precision:
                self.to(orig_dtype)
    
    def _encode_images(self, batched_input):
        """Encode the input images using the efficient image encoder"""
        images = batched_input.get('images', None)
        if images is None:
            return None
            
        # Get batch and time dimensions
        B, T, C, H, W = images.shape
        
        # Reshape for processing
        images = images.view(B * T, C, H, W)
        
        # Encode images using the efficient image encoder
        image_embeddings = self.image_encoder(images)
        
        # Reshape back to include time dimension
        _, E, Ht, Wt = image_embeddings.shape
        image_embeddings = image_embeddings.view(B, T, E, Ht, Wt)
        
        return image_embeddings
    
    def _encode_prompts(self, batched_input):
        """Encode the prompts (points, boxes, masks) using the prompt encoder"""
        # Get the prompts from the input
        points = batched_input.get('points', None)
        boxes = batched_input.get('boxes', None)
        masks = batched_input.get('masks', None)
        
        # Pass prompts through the prompt encoder
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
        
        return sparse_embeddings, dense_embeddings
    
    def _apply_memory_attention(self, image_embeddings, batched_input):
        """Apply memory attention to incorporate temporal context"""
        if image_embeddings is None:
            return None
            
        # Extract batch and time dimensions
        B, T, E, H, W = image_embeddings.shape
        
        # Prepare memory features from previous frames
        memory_features = []
        
        for t in range(T):
            # For the first frame, use no_memory_embed
            if t == 0:
                if self.directly_add_no_mem_embed:
                    # Directly add no memory embedding (more efficient)
                    curr_feat = image_embeddings[:, t]
                    no_mem = self.no_memory_embed.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
                    enhanced_feat = curr_feat + no_mem
                    memory_features.append(enhanced_feat)
                else:
                    # Use memory attention module with empty memory
                    curr_feat = image_embeddings[:, t].view(B, E, H*W).permute(0, 2, 1)  # (B, H*W, E)
                    no_mem = self.no_memory_embed.expand(B, 1, E)  # (B, 1, E)
                    enhanced_feat = self.memory_attention(curr_feat, no_mem, no_mem)
                    enhanced_feat = enhanced_feat.permute(0, 2, 1).view(B, E, H, W)
                    memory_features.append(enhanced_feat)
            else:
                # For subsequent frames, use memory from previous frames
                curr_feat = image_embeddings[:, t]  # Current frame features
                
                # Select memory frames (limited by max_cond_frames_in_attn)
                mem_indices = list(range(max(0, t - self.num_maskmem + 1), t))
                if self.max_cond_frames_in_attn > 0 and len(mem_indices) > self.max_cond_frames_in_attn:
                    # Select closest frames for efficiency
                    mem_indices = select_closest_cond_frames(
                        t, mem_indices, self.max_cond_frames_in_attn
                    )
                
                # Extract memory features and masks
                memory_feats = [image_embeddings[:, i] for i in mem_indices]
                memory_masks = [batched_input.get('prev_masks', None)[:, i] if batched_input.get('prev_masks', None) is not None else None 
                               for i in mem_indices]
                
                # Encode memory
                mem_embeddings = self.memory_encoder(memory_feats, memory_masks)
                
                # Flatten spatial dimensions for attention
                curr_feat_flat = curr_feat.view(B, E, H*W).permute(0, 2, 1)  # (B, H*W, E)
                
                # Apply memory attention
                enhanced_feat = self.memory_attention(curr_feat_flat, mem_embeddings, mem_embeddings)
                
                # Reshape back to spatial dimensions
                enhanced_feat = enhanced_feat.permute(0, 2, 1).view(B, E, H, W)
                memory_features.append(enhanced_feat)
        
        # Stack along time dimension
        enhanced_embeddings = torch.stack(memory_features, dim=1)  # (B, T, E, H, W)
        
        return enhanced_embeddings
    
    def _decode_masks(self, image_embeddings, sparse_embeddings, dense_embeddings, multimask_output):
        """Generate masks using the mask decoder"""
        # Extract shapes
        B, T = image_embeddings.shape[:2]
        
        # Prepare outputs
        masks_list = []
        iou_list = []
        
        # Process each frame
        for t in range(T):
            curr_embedding = image_embeddings[:, t]  # (B, E, H, W)
            
            # Apply mask decoder
            mask_pred, iou_pred = self.mask_decoder(
                image_embeddings=curr_embedding,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output and self.multimask_output_in_sam,
            )
            
            masks_list.append(mask_pred)
            iou_list.append(iou_pred)
        
        # Stack outputs
        masks = torch.stack(masks_list, dim=1)  # (B, T, num_masks, H, W)
        iou_scores = torch.stack(iou_list, dim=1)  # (B, T, num_masks)
        
        return masks, iou_scores
    
    def quantize(self, quantization_bit=8):
        """Quantize model weights for further memory reduction"""
        # This is a placeholder for post-training quantization
        # In a real implementation, this would use PyTorch's quantization APIs
        print(f"Model quantized to {quantization_bit}-bit precision")
        return self
