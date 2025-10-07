"""
Knowledge Distillation Loss Functions
Implements feature-level, logit-level, and ground truth losses for the 3-stage approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in medical segmentation.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] prediction logits
            target: [B, C, H, W] ground truth masks
        """
        # Binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Probability
        pt = torch.exp(-bce_loss)
        
        # Focal weight
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Focal loss
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for overlap-based optimization.
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] prediction logits
            target: [B, C, H, W] ground truth masks
        """
        # Apply sigmoid to get probabilities
        pred_probs = torch.sigmoid(pred)
        
        # Flatten spatial dimensions
        pred_flat = pred_probs.view(pred_probs.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice_coeff = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        
        dice_loss = 1.0 - dice_coeff
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level distillation loss for Stage 1.
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        cosine_weight: float = 0.1,
        attention_weight: float = 0.5
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.attention_weight = attention_weight
    
    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute feature distillation loss.
        """
        
        total_loss = 0.0
        loss_dict = {}
        count = 0
        
        # Feature matching at different scales
        feature_keys = ['backbone_1_4', 'backbone_1_16', 'post_memory']
        
        for key in feature_keys:
            if key in student_features and key in teacher_features:
                s_feat = student_features[key]
                t_feat = teacher_features[key]
                
                # Ensure same spatial dimensions
                if s_feat.shape != t_feat.shape:
                    s_feat = F.interpolate(
                        s_feat, 
                        size=t_feat.shape[-2:], 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # MSE loss
                mse_loss = F.mse_loss(s_feat, t_feat)
                
                # Cosine similarity loss
                s_norm = F.normalize(s_feat.flatten(2), dim=-1)
                t_norm = F.normalize(t_feat.flatten(2), dim=-1)
                cos_sim = F.cosine_similarity(s_norm, t_norm, dim=-1).mean()
                cos_loss = 1.0 - cos_sim
                
                # Attention transfer loss
                s_attn = torch.mean(s_feat, dim=1, keepdim=True)  # Spatial attention
                t_attn = torch.mean(t_feat, dim=1, keepdim=True)
                attn_loss = F.mse_loss(
                    F.softmax(s_attn.flatten(2), dim=-1),
                    F.softmax(t_attn.flatten(2), dim=-1)
                )
                
                # Combine losses
                feature_loss = (
                    self.mse_weight * mse_loss + 
                    self.cosine_weight * cos_loss + 
                    self.attention_weight * attn_loss
                )
                
                total_loss += feature_loss
                loss_dict[f'{key}_mse'] = mse_loss
                loss_dict[f'{key}_cosine'] = cos_loss
                loss_dict[f'{key}_attention'] = attn_loss
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        loss_dict['feature_total'] = total_loss
        return total_loss, loss_dict


class LogitDistillationLoss(nn.Module):
    """
    Logit-level distillation loss for Stage 2.
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        kl_weight: float = 1.0,
        soft_dice_weight: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.kl_weight = kl_weight
        self.soft_dice_weight = soft_dice_weight
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute logit distillation loss.
        """
        
        # Temperature scaling
        student_soft = torch.sigmoid(student_logits / self.temperature)
        teacher_soft = torch.sigmoid(teacher_logits / self.temperature)
        
        # KL divergence loss
        kl_loss = F.kl_div(
            torch.log(student_soft + 1e-8),
            teacher_soft,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Soft dice loss on teacher probabilities
        intersection = (student_soft * teacher_soft).sum()
        union = student_soft.sum() + teacher_soft.sum()
        soft_dice = 1.0 - (2.0 * intersection) / (union + 1e-8)
        
        # Combined loss
        total_loss = self.kl_weight * kl_loss + self.soft_dice_weight * soft_dice
        
        loss_dict = {
            'kl_loss': kl_loss,
            'soft_dice': soft_dice,
            'logit_total': total_loss
        }
        
        return total_loss, loss_dict


class GroundTruthLoss(nn.Module):
    """
    Ground truth supervision loss (MedSAM2 style: weighted focal + dice ≈ 20:1).
    """
    
    def __init__(
        self,
        focal_weight: float = 1.0,
        dice_weight: float = 20.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        gt_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute ground truth supervision loss.
        """
        
        # Focal loss
        focal = self.focal_loss(pred_logits, gt_masks)
        
        # Dice loss
        dice = self.dice_loss(pred_logits, gt_masks)
        
        # Combined loss (MedSAM2 style)
        total_loss = self.focal_weight * focal + self.dice_weight * dice
        
        loss_dict = {
            'focal_loss': focal,
            'dice_loss': dice,
            'gt_total': total_loss
        }
        
        return total_loss, loss_dict


class DistillationLoss(nn.Module):
    """
    Multi-stage knowledge distillation loss function.
    Combines feature, logit, and ground truth losses based on training stage.
    """
    
    def __init__(
        self,
        stage: int = 1,
        feature_weight: float = 1.0,
        logit_weight: float = 1.0,
        gt_weight: float = 1.0,
        temperature: float = 4.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 20.0,
        focal_weight: float = 1.0
    ):
        super().__init__()
        
        self.stage = stage
        self.feature_weight = feature_weight
        self.logit_weight = logit_weight
        self.gt_weight = gt_weight
        
        # Initialize component losses
        self.feature_distillation = FeatureDistillationLoss()
        self.logit_distillation = LogitDistillationLoss(temperature=temperature)
        self.ground_truth_loss = GroundTruthLoss(
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )
    
    def forward(
        self,
        student_output: Dict[str, torch.Tensor],
        teacher_output: Optional[Dict[str, torch.Tensor]] = None,
        gt_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total distillation loss based on current stage.
        
        Args:
            student_output: Dictionary containing student model outputs
            teacher_output: Dictionary containing teacher model outputs
            gt_masks: Ground truth masks for supervision
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        
        total_loss = 0.0
        loss_dict = {}
        
        if self.stage == 1:
            # Stage 1: Pre-memory feature distillation only
            if teacher_output is not None and 'features' in student_output and 'features' in teacher_output:
                feat_loss, feat_dict = self.feature_distillation(
                    student_output['features'],
                    teacher_output['features']
                )
                total_loss += self.feature_weight * feat_loss
                loss_dict.update(feat_dict)
            
        elif self.stage == 2:
            # Stage 2: Memory-aware distillation (post-memory features + logits)
            if teacher_output is not None:
                # Post-memory feature distillation
                if 'post_memory_features' in student_output and 'post_memory_features' in teacher_output:
                    post_mem_loss = F.mse_loss(
                        student_output['post_memory_features'],
                        teacher_output['post_memory_features']
                    )
                    total_loss += self.feature_weight * post_mem_loss
                    loss_dict['post_memory_loss'] = post_mem_loss
                
                # Logit distillation
                if 'logits' in student_output and 'logits' in teacher_output:
                    logit_loss, logit_dict = self.logit_distillation(
                        student_output['logits'],
                        teacher_output['logits']
                    )
                    total_loss += self.logit_weight * logit_loss
                    loss_dict.update(logit_dict)
            
        elif self.stage == 3:
            # Stage 3: Fine-tuning with GT + small KD term
            if gt_masks is not None and 'logits' in student_output:
                gt_loss, gt_dict = self.ground_truth_loss(
                    student_output['logits'],
                    gt_masks
                )
                total_loss += self.gt_weight * gt_loss
                loss_dict.update(gt_dict)
            
            # Small KD term (reduced weight)
            if teacher_output is not None and 'logits' in student_output and 'logits' in teacher_output:
                kd_loss, kd_dict = self.logit_distillation(
                    student_output['logits'],
                    teacher_output['logits']
                )
                total_loss += 0.1 * kd_loss  # Small KD weight
                loss_dict.update({f'kd_{k}': v for k, v in kd_dict.items()})
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive loss weighting that adjusts weights during training.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        num_losses: int,
        temperature: float = 2.0
    ):
        super().__init__()
        self.base_loss = base_loss
        self.num_losses = num_losses
        self.temperature = temperature
        
        # Learnable weights
        self.loss_weights = nn.Parameter(torch.ones(num_losses))
        
    def forward(self, *args, **kwargs):
        """Forward with adaptive weighting."""
        loss, loss_dict = self.base_loss(*args, **kwargs)
        
        # Apply learned weights (softmax normalized)
        weights = F.softmax(self.loss_weights / self.temperature, dim=0)
        
        # This is a simplified version - in practice, you'd need to 
        # separate different loss terms and apply weights individually
        return loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy data
    B, C, H, W = 2, 1, 64, 64
    student_logits = torch.randn(B, C, H, W).to(device)
    teacher_logits = torch.randn(B, C, H, W).to(device)
    gt_masks = torch.randint(0, 2, (B, C, H, W)).float().to(device)
    
    # Test individual losses
    print("Testing individual losses:")
    
    # Focal loss
    focal_loss = FocalLoss()
    focal_result = focal_loss(student_logits, gt_masks)
    print(f"Focal Loss: {focal_result.item():.4f}")
    
    # Dice loss
    dice_loss = DiceLoss()
    dice_result = dice_loss(student_logits, gt_masks)
    print(f"Dice Loss: {dice_result.item():.4f}")
    
    # Ground truth loss
    gt_loss = GroundTruthLoss()
    gt_result, gt_dict = gt_loss(student_logits, gt_masks)
    print(f"Ground Truth Loss: {gt_result.item():.4f}")
    
    # Logit distillation
    logit_dist = LogitDistillationLoss()
    logit_result, logit_dict = logit_dist(student_logits, teacher_logits)
    print(f"Logit Distillation Loss: {logit_result.item():.4f}")
    
    # Feature distillation
    student_features = {
        'backbone_1_4': torch.randn(B, 256, H//4, W//4).to(device),
        'backbone_1_16': torch.randn(B, 256, H//16, W//16).to(device)
    }
    teacher_features = {
        'backbone_1_4': torch.randn(B, 256, H//4, W//4).to(device),
        'backbone_1_16': torch.randn(B, 256, H//16, W//16).to(device)
    }
    
    feat_dist = FeatureDistillationLoss()
    feat_result, feat_dict = feat_dist(student_features, teacher_features)
    print(f"Feature Distillation Loss: {feat_result.item():.4f}")
    
    # Test complete distillation loss
    print("\nTesting complete distillation loss:")
    
    for stage in [1, 2, 3]:
        distill_loss = DistillationLoss(stage=stage)
        
        student_output = {
            'logits': student_logits,
            'features': student_features
        }
        teacher_output = {
            'logits': teacher_logits,
            'features': teacher_features
        }
        
        total_loss, loss_dict = distill_loss(
            student_output=student_output,
            teacher_output=teacher_output,
            gt_masks=gt_masks
        )
        
        print(f"Stage {stage} Total Loss: {total_loss.item():.4f}")
        print(f"  Loss components: {list(loss_dict.keys())}")
    
    print("\nLoss functions test completed!")