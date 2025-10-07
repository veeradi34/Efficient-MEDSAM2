"""
Teacher Model: Full MedSAM2 Implementation
This wraps the original MedSAM2 model for knowledge distillation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add paths to import SAM2 and MedSAM2 from local directory
sys.path.append(os.path.join(os.path.dirname(__file__), '../sam2'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
except ImportError:
    print("Warning: Could not import SAM2 modules. Make sure SAM2 is in the local directory.")
    # Define dummy classes for development
    class SAM2ImagePredictor:
        def __init__(self, model): pass
    def build_sam2_video_predictor(config, checkpoint): return None


class MedSAM2Teacher(nn.Module):
    """
    Teacher model wrapper for MedSAM2.
    Extracts intermediate features for knowledge distillation.
    """
    
    def __init__(
        self,
        config_file: str = "configs/sam2.1_hiera_t512.yaml",
        checkpoint_path: str = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.config_file = config_file
        
        # Build the original SAM2 model
        try:
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.sam2_model = build_sam2_video_predictor(config_file, checkpoint_path)
            else:
                # Load from default location
                self.sam2_model = build_sam2_video_predictor(config_file)
                
            self.sam2_model.to(device)
            
            # Create predictor wrapper
            self.predictor = SAM2ImagePredictor(self.sam2_model)
            
        except Exception as e:
            print(f"Warning: Could not load SAM2 model: {e}")
            self.sam2_model = None
            self.predictor = None
        
        # Hook storage for intermediate features
        self.intermediate_features = {}
        if self.sam2_model is not None:
            self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture intermediate features."""
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    self.intermediate_features[name] = output.clone()
                elif isinstance(output, (list, tuple)) and len(output) > 0:
                    self.intermediate_features[name] = output[0].clone()
            return hook
        
        # Register hooks for different scales
        if hasattr(self.sam2_model, 'image_encoder'):
            # Hook the backbone at different scales
            if hasattr(self.sam2_model.image_encoder, 'trunk'):
                # Hiera backbone
                trunk = self.sam2_model.image_encoder.trunk
                if hasattr(trunk, 'stages'):
                    # Hook different stages for multi-scale features
                    if len(trunk.stages) >= 2:
                        trunk.stages[1].register_forward_hook(get_activation('backbone_1_4'))
                    if len(trunk.stages) >= 4:
                        trunk.stages[3].register_forward_hook(get_activation('backbone_1_16'))
                        
            # Hook memory encoder if available
            if hasattr(self.sam2_model, 'memory_encoder'):
                self.sam2_model.memory_encoder.register_forward_hook(get_activation('memory_encoder'))
                
            # Hook memory attention
            if hasattr(self.sam2_model, 'memory_attention'):
                self.sam2_model.memory_attention.register_forward_hook(get_activation('memory_attention'))
    
    def extract_backbone_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale backbone features."""
        if self.sam2_model is None:
            # Return dummy features for development
            B, C, H, W = images.shape
            return {
                'backbone_1_4': torch.randn(B, 256, H//4, W//4, device=images.device),
                'backbone_1_16': torch.randn(B, 256, H//16, W//16, device=images.device),
                'final': torch.randn(B, 256, H//16, W//16, device=images.device)
            }
            
        self.intermediate_features.clear()
        
        with torch.no_grad():
            # Forward through image encoder
            features = self.sam2_model.image_encoder(images)
            
        # Organize features by scale
        extracted_features = {}
        
        # Add the final encoder output
        if isinstance(features, torch.Tensor):
            extracted_features['final'] = features
        elif isinstance(features, dict):
            extracted_features.update(features)
            
        # Add intermediate features from hooks
        extracted_features.update(self.intermediate_features)
        
        return extracted_features
    
    def forward_with_memory(
        self, 
        images: torch.Tensor, 
        memory_bank: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through memory attention."""
        
        # Extract backbone features first
        backbone_features = self.extract_backbone_features(images)
        
        # If we have memory bank, process through memory attention
        if memory_bank is not None and hasattr(self.sam2_model, 'memory_attention'):
            # This is a simplified memory attention - adapt based on actual SAM2 implementation
            memory_features = self.sam2_model.memory_attention(
                backbone_features.get('final', list(backbone_features.values())[0]),
                memory_bank
            )
            backbone_features['post_memory'] = memory_features
            
        return backbone_features.get('final', list(backbone_features.values())[0]), backbone_features
    
    def predict_mask(
        self,
        images: torch.Tensor,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks using the teacher model.
        Returns logits for knowledge distillation.
        """
        
        if self.predictor is None:
            # Return dummy outputs for development
            B = images.shape[0] if images.dim() == 4 else 1
            H, W = images.shape[-2:]
            masks = torch.zeros(B, 1, H, W, device=images.device)
            scores = torch.ones(B, 1, device=images.device)
            logits = torch.zeros(B, 1, H, W, device=images.device)
            return masks, scores, logits
        
        # Set image for predictor
        if images.dim() == 4:
            # Remove batch dimension for SAM2ImagePredictor
            image_np = images[0].cpu().numpy()
        else:
            image_np = images.cpu().numpy()
            
        self.predictor.set_image(image_np)
        
        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output
        )
        
        # Convert to tensors
        masks_tensor = torch.from_numpy(masks).to(self.device)
        scores_tensor = torch.from_numpy(scores).to(self.device)  
        logits_tensor = torch.from_numpy(logits).to(self.device)
        
        return masks_tensor, scores_tensor, logits_tensor
    
    def get_feature_dimensions(self) -> Dict[str, Tuple[int, ...]]:
        """Get the dimensions of different feature maps for student model design."""
        # Create a dummy input to get feature dimensions
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        
        with torch.no_grad():
            features = self.extract_backbone_features(dummy_input)
            
        feature_dims = {}
        for name, feat in features.items():
            if isinstance(feat, torch.Tensor):
                feature_dims[name] = feat.shape
                
        return feature_dims
    
    def freeze_model(self):
        """Freeze all parameters of the teacher model."""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_model(self):
        """Unfreeze all parameters (not typically needed for teacher)."""
        for param in self.parameters():
            param.requires_grad = True


def load_medsam2_teacher(
    config_file: str = "configs/sam2.1_hiera_t512.yaml",
    checkpoint_path: str = None,
    device: str = "cuda"
) -> MedSAM2Teacher:
    """
    Factory function to load MedSAM2 teacher model.
    """
    teacher = MedSAM2Teacher(
        config_file=config_file,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # Freeze by default (teacher should not be trained)
    teacher.freeze_model()
    teacher.eval()
    
    return teacher


if __name__ == "__main__":
    # Test the teacher model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load teacher
    teacher = load_medsam2_teacher(device=device)
    
    # Test with dummy input
    dummy_image = torch.randn(1, 3, 512, 512).to(device)
    
    # Test feature extraction
    features = teacher.extract_backbone_features(dummy_image)
    print("Extracted features:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    # Test prediction
    point_coords = np.array([[256, 256]])
    point_labels = np.array([1])
    
    masks, scores, logits = teacher.predict_mask(
        dummy_image,
        point_coords=point_coords,
        point_labels=point_labels
    )
    
    print(f"\nPrediction results:")
    print(f"  Masks: {masks.shape}")
    print(f"  Scores: {scores.shape}")
    print(f"  Logits: {logits.shape}")