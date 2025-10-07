"""
Efficient MedSAM2 Student Model
Lightweight model designed for knowledge distillation from MedSAM2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import os

from .backbone import EfficientBackbone
from .roi_memory_attention import ROIAwareMemoryAttention, SlimMemoryBank


class EfficientPromptEncoder(nn.Module):
    """
    Lightweight prompt encoder for points, boxes, and masks.
    """
    
    def __init__(self, embed_dim: int = 256, image_embedding_size: int = 32):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size
        
        # Point embeddings
        self.point_embeddings = nn.Embedding(2, embed_dim)  # positive/negative points
        
        # Box embedding
        self.box_embedding = nn.Embedding(1, embed_dim)
        
        # Mask embedding - more efficient
        self.mask_embed = nn.Sequential(
            nn.Conv2d(1, embed_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        
        # Positional encoding for points
        self.register_buffer("pe_layer", self._get_positional_encoding())
        
    def _get_positional_encoding(self) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        pe = torch.zeros(self.image_embedding_size, self.image_embedding_size, self.embed_dim)
        
        # Create position indices
        position_h = torch.arange(self.image_embedding_size).float()
        position_w = torch.arange(self.image_embedding_size).float()
        
        # Create div term for sinusoidal encoding (only for half dimensions)
        half_dim = self.embed_dim // 2
        div_term = torch.exp(torch.arange(0, half_dim).float() * 
                           -(torch.log(torch.tensor(10000.0)) / half_dim))
        
        # Apply sinusoidal encoding
        for h in range(self.image_embedding_size):
            for w in range(self.image_embedding_size):
                # Height encoding (sin for even indices)
                pe[h, w, 0::2] = torch.sin(position_h[h] * div_term)[:self.embed_dim//2]
                # Width encoding (cos for odd indices)  
                pe[h, w, 1::2] = torch.cos(position_w[w] * div_term)[:self.embed_dim//2]
        
        return pe
    
    def forward(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts into sparse and dense representations.
        """
        
        B = 1  # Will be updated based on inputs
        sparse_embeddings = []
        
        # Process points
        if point_coords is not None and point_labels is not None:
            B = point_coords.shape[0]
            point_embeddings = self.point_embeddings(point_labels)
            
            # Add positional encoding
            for i, coords in enumerate(point_coords):
                for j, coord in enumerate(coords):
                    x, y = int(coord[0] * self.image_embedding_size / 512), int(coord[1] * self.image_embedding_size / 512)
                    x = max(0, min(x, self.image_embedding_size - 1))
                    y = max(0, min(y, self.image_embedding_size - 1))
                    point_embeddings[i, j] += self.pe_layer[y, x]
            
            sparse_embeddings.append(point_embeddings)
        
        # Process boxes
        if boxes is not None:
            B = boxes.shape[0]
            box_embeddings = self.box_embedding.weight.unsqueeze(0).repeat(B, boxes.shape[1], 1)
            sparse_embeddings.append(box_embeddings)
        
        # Combine sparse embeddings
        if sparse_embeddings:
            sparse_embed = torch.cat(sparse_embeddings, dim=1)
        else:
            device = self.pe_layer.device
            sparse_embed = torch.empty(B, 0, self.embed_dim, device=device)
        
        # Process masks (dense embedding)
        if masks is not None:
            dense_embed = self.mask_embed(masks)
        else:
            # No mask prompt
            device = self.pe_layer.device
            dense_embed = torch.zeros(
                B, self.embed_dim, 
                self.image_embedding_size // 4, 
                self.image_embedding_size // 4,
                device=device
            )
        
        return sparse_embed, dense_embed


class LightweightMaskDecoder(nn.Module):
    """
    Lightweight mask decoder for efficient inference.
    """
    
    def __init__(
        self,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 3,
        iou_head_depth: int = 2,
        iou_head_hidden_dim: int = 128
    ):
        super().__init__()
        
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        
        # Output tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(num_multimask_outputs, transformer_dim)
        
        # Lightweight transformer for mask generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim,
            nhead=8,
            dim_feedforward=transformer_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # Output upscaling - more efficient
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(32, transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU()
        )
        
        # Hypernetworks for mask generation
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.ReLU(inplace=True),
                nn.Linear(transformer_dim, transformer_dim // 8)
            ) for _ in range(num_multimask_outputs)
        ])
        
        # IoU prediction head
        iou_layers = []
        iou_layers.extend([nn.Linear(transformer_dim, iou_head_hidden_dim), nn.ReLU(inplace=True)])
        for _ in range(iou_head_depth - 1):
            iou_layers.extend([nn.Linear(iou_head_hidden_dim, iou_head_hidden_dim), nn.ReLU(inplace=True)])
        iou_layers.append(nn.Linear(iou_head_hidden_dim, num_multimask_outputs))
        
        self.iou_prediction_head = nn.Sequential(*iou_layers)
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate masks from embeddings.
        """
        
        B, C, H, W = image_embeddings.shape
        
        # Prepare tokens
        output_tokens = torch.cat([
            self.iou_token.weight,
            self.mask_tokens.weight
        ], dim=0).unsqueeze(0).repeat(B, 1, 1)
        
        # Add sparse prompts to tokens
        if sparse_prompt_embeddings.shape[1] > 0:
            tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
        else:
            tokens = output_tokens
        
        # Prepare source features
        src = image_embeddings.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # Add dense prompt embeddings to source
        if dense_prompt_embeddings.numel() > 0 and dense_prompt_embeddings.shape[2] > 0:
            dense_resized = F.interpolate(
                dense_prompt_embeddings, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            dense_flat = dense_resized.flatten(2).transpose(1, 2)
            src = src + dense_flat
        
        # Transformer decoder
        hs = self.transformer(tokens, src)
        
        # Extract IoU and mask tokens
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_multimask_outputs), :]
        
        # Predict IoU scores
        iou_pred = self.iou_prediction_head(iou_token_out)
        
        # Generate masks
        src_reshaped = src.transpose(1, 2).reshape(B, C, H, W)
        upscaled_embedding = self.output_upscaling(src_reshaped)
        
        masks = []
        for i in range(self.num_multimask_outputs):
            mask_tokens_i = mask_tokens_out[:, i, :]
            hyper_in = self.output_hypernetworks_mlps[i](mask_tokens_i)
            
            # Generate mask through hypernetwork
            B_hyper, C_hyper = hyper_in.shape
            H_up, W_up = upscaled_embedding.shape[-2:]
            
            hyper_in_spatial = hyper_in.view(B_hyper, C_hyper, 1, 1).expand(-1, -1, H_up, W_up)
            mask = (hyper_in_spatial * upscaled_embedding).sum(dim=1, keepdim=True)
            masks.append(mask)
        
        masks = torch.cat(masks, dim=1)
        
        # Select output based on multimask_output
        if multimask_output:
            return masks, iou_pred
        else:
            # Return best mask based on IoU prediction
            best_idx = torch.argmax(iou_pred, dim=1)
            best_masks = masks[torch.arange(B), best_idx].unsqueeze(1)
            best_iou = iou_pred[torch.arange(B), best_idx].unsqueeze(1)
            return best_masks, best_iou


class EfficientMedSAM2Student(nn.Module):
    """
    Complete efficient student model for MedSAM2 knowledge distillation.
    """
    
    def __init__(
        self,
        backbone_type: str = "mobilenet_v3_small",
        embed_dim: int = 256,
        image_size: int = 512,
        patch_size: int = 16,
        memory_layers: int = 1,
        memory_heads: int = 4,
        max_memory_frames: int = 2
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Channel adapter for multi-channel medical data (e.g., MSD 4-channel MRI)
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from utils.channel_adapter import ChannelAdapter
        self.channel_adapter = ChannelAdapter()
        
        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad_(True)
        
        # Components
        self.image_encoder = EfficientBackbone(
            backbone_type=backbone_type,
            out_channels=embed_dim,
            patch_size=patch_size
        )
        
        self.memory_attention = ROIAwareMemoryAttention(
            d_model=embed_dim,
            nhead=memory_heads,
            num_layers=memory_layers,
            max_memory_frames=max_memory_frames
        )
        
        self.prompt_encoder = EfficientPromptEncoder(
            embed_dim=embed_dim,
            image_embedding_size=image_size // patch_size
        )
        
        self.mask_decoder = LightweightMaskDecoder(
            transformer_dim=embed_dim
        )
        
        # Memory bank for temporal consistency
        self.memory_bank = SlimMemoryBank(
            feature_dim=embed_dim,
            max_frames=max_memory_frames
        )
        
    def forward(
        self,
        images: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        mask_inputs: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        use_memory: bool = True,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the student model.
        """
        
        # Adapt multi-channel input to 3-channel for backbone
        adapted_images = self.channel_adapter(images)
        
        # Extract image features
        image_features = self.image_encoder(adapted_images)
        
        # Use 1/16 scale features for main processing
        main_features = image_features['backbone_1_16']
        
        # Apply memory attention if enabled
        post_memory_features = main_features
        if use_memory:
            # Get memory from bank
            memory_features, _ = self.memory_bank.get_memory()
            
            # Apply memory attention
            post_memory_features = self.memory_attention(main_features, memory_features)
            
            # Update memory bank with current features
            self.memory_bank.add_frame(main_features.detach())
        
        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            masks=mask_inputs
        )
        
        # Generate masks
        masks, iou_pred = self.mask_decoder(
            image_embeddings=post_memory_features,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output
        )
        
        result = {
            'masks': masks,
            'iou_predictions': iou_pred,
            'logits': masks  # Use masks as logits for now
        }
        
        if return_features:
            result['features'] = image_features
            result['post_memory_features'] = post_memory_features
        
        return result
    
    def reset_memory(self):
        """Reset the memory bank."""
        self.memory_bank.clear_memory()
    
    def set_memory_frames(self, max_frames: int):
        """Set maximum memory frames."""
        self.memory_bank.max_frames = max_frames
        self.memory_attention.max_memory_frames = max_frames


class EfficientMedSAM2Predictor:
    """
    Predictor interface for EfficientMedSAM2Student model.
    """
    
    def __init__(self, model: EfficientMedSAM2Student):
        self.model = model
        self.model.eval()
        self.is_image_set = False
        self.current_image = None
        
    def set_image(self, image: np.ndarray):
        """Set the current image for prediction."""
        if isinstance(image, np.ndarray):
            # Convert numpy to tensor
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=0)  # Add channel dimension
            if len(image.shape) == 3 and image.shape[0] not in [1, 3]:
                image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            
            # Normalize to [0, 1]
            if image.max() > 1:
                image = image.astype(np.float32) / 255.0
            
            # Ensure 3 channels
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            
            # Convert to tensor and add batch dimension
            self.current_image = torch.from_numpy(image).unsqueeze(0)
            
        elif isinstance(image, torch.Tensor):
            self.current_image = image
            
        self.is_image_set = True
        
        # Reset memory for new image
        self.model.reset_memory()
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the set image.
        """
        
        if not self.is_image_set:
            raise RuntimeError("Image must be set before prediction")
        
        # Convert inputs to tensors
        point_coords_tensor = None
        point_labels_tensor = None
        
        if point_coords is not None and point_labels is not None:
            point_coords_tensor = torch.from_numpy(point_coords).float().unsqueeze(0)
            point_labels_tensor = torch.from_numpy(point_labels).long().unsqueeze(0)
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        current_image = self.current_image.to(device)
        
        if point_coords_tensor is not None:
            point_coords_tensor = point_coords_tensor.to(device)
            point_labels_tensor = point_labels_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            output = self.model(
                images=current_image,
                point_coords=point_coords_tensor,
                point_labels=point_labels_tensor,
                multimask_output=multimask_output
            )
        
        # Convert outputs to numpy
        masks = torch.sigmoid(output['logits']).cpu().numpy()
        iou_predictions = output['iou_predictions'].cpu().numpy()
        logits = output['logits'].cpu().numpy()
        
        # Remove batch dimension
        masks = masks[0]
        iou_predictions = iou_predictions[0]
        logits = logits[0]
        
        return masks, iou_predictions, logits


def build_efficient_student_model(
    backbone_type: str = "mobilenet_v3_small",
    embed_dim: int = 256,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda"
) -> EfficientMedSAM2Student:
    """
    Factory function to build student model.
    """
    
    model = EfficientMedSAM2Student(
        backbone_type=backbone_type,
        embed_dim=embed_dim
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded student model from {checkpoint_path}")
    
    model.to(device)
    return model


if __name__ == "__main__":
    # Test student model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Testing EfficientMedSAM2Student...")
    
    student = build_efficient_student_model(device=device)
    predictor = EfficientMedSAM2Predictor(student)
    
    # Test forward pass
    dummy_images = torch.randn(2, 3, 512, 512).to(device)
    point_coords = torch.tensor([[[256, 256]]], dtype=torch.float).to(device)
    point_labels = torch.tensor([[1]], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = student(
            images=dummy_images,
            point_coords=point_coords,
            point_labels=point_labels,
            return_features=True
        )
    
    print("Student model output:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
    
    # Test predictor interface
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    predictor.set_image(dummy_image)
    
    point_coords = np.array([[256, 256]])
    point_labels = np.array([1])
    
    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels
    )
    
    print(f"\nPredictor output:")
    print(f"  Masks: {masks.shape}")
    print(f"  Scores: {scores.shape}")
    print(f"  Logits: {logits.shape}")
    
    print("EfficientMedSAM2Student test completed!")