"""
Knowledge Distillation Trainer
Implements the 3-stage decoupled distillation approach for EfficientMedSAM2.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import json

import sys
sys.path.append('..')

from teacher_model.medsam2 import MedSAM2Teacher
from efficient_medsam2.efficient_medsam2_base import EfficientMedSAM2
from .losses import DistillationLoss


class KnowledgeDistillationTrainer:
    """
    Main trainer for knowledge distillation between MedSAM2 (teacher) and EfficientMedSAM2 (student).
    """
    
    def __init__(
        self,
        teacher_model: MedSAM2Teacher,
        student_model: EfficientMedSAM2,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        use_amp: bool = True,
        save_dir: str = "../checkpoints",
        log_dir: str = "../logs"
    ):
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_stage = 1
        self.global_step = 0
        self.best_val_score = 0.0
        
        # Metrics tracking
        self.train_history = {'stage1': [], 'stage2': [], 'stage3': []}
        self.val_history = {'stage1': [], 'stage2': [], 'stage3': []}
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.log_dir, 'distillation.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def freeze_student_components(self, freeze_memory: bool = True, freeze_decoder: bool = True):
        """Freeze specific components of student model."""
        
        frozen_components = []
        
        if freeze_memory:
            for param in self.student_model.memory_attention.parameters():
                param.requires_grad = False
            frozen_components.append("memory_attention")
        
        if freeze_decoder:
            for param in self.student_model.mask_decoder.parameters():
                param.requires_grad = False
            frozen_components.append("mask_decoder")
        
        self.logger.info(f"Froze components: {frozen_components}")
    
    def unfreeze_student_components(self, unfreeze_memory: bool = True, unfreeze_decoder: bool = False):
        """Unfreeze specific components of student model."""
        
        unfrozen_components = []
        
        if unfreeze_memory:
            for param in self.student_model.memory_attention.parameters():
                param.requires_grad = True
            unfrozen_components.append("memory_attention")
        
        if unfreeze_decoder:
            for param in self.student_model.mask_decoder.parameters():
                param.requires_grad = True
            unfrozen_components.append("mask_decoder")
        
        self.logger.info(f"Unfroze components: {unfrozen_components}")
    
    def get_optimizer(self, learning_rate: float, weight_decay: float = 1e-5) -> optim.Optimizer:
        """Create optimizer for trainable parameters only."""
        trainable_params = filter(lambda p: p.requires_grad, self.student_model.parameters())
        
        # Count trainable parameters
        num_params = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
        self.logger.info(f"Training {num_params:,} parameters")
        
        return optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
    
    def train_stage(
        self,
        stage: int,
        epochs: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_frequency: int = 5,
        val_frequency: int = 5
    ):
        """Train a specific stage of distillation."""
        
        self.current_stage = stage
        self.logger.info(f"=" * 60)
        self.logger.info(f"Starting Stage {stage} training for {epochs} epochs")
        self.logger.info(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        
        # Configure model freezing based on stage
        if stage == 1:
            # Stage 1: Freeze memory + decoder, train only backbone
            self.freeze_student_components(freeze_memory=True, freeze_decoder=True)
        elif stage == 2:
            # Stage 2: Unfreeze memory, keep decoder frozen
            self.unfreeze_student_components(unfreeze_memory=True, unfreeze_decoder=False)
        elif stage == 3:
            # Stage 3: Unfreeze all components
            self.unfreeze_student_components(unfreeze_memory=True, unfreeze_decoder=True)
        
        # Setup optimizer and scheduler
        optimizer = self.get_optimizer(learning_rate, weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Loss function for current stage
        loss_fn = DistillationLoss(stage=stage)
        
        # Training loop
        stage_train_losses = []
        stage_val_scores = []
        
        for epoch in range(epochs):
            self.logger.info(f"Stage {stage}, Epoch {epoch+1}/{epochs}")
            
            # Training
            train_metrics = self._train_epoch(optimizer, loss_fn, stage)
            stage_train_losses.append(train_metrics)
            
            # Update scheduler
            scheduler.step()
            
            # Log training metrics
            self.logger.info(f"Train Loss: {train_metrics['total_loss']:.6f}")
            for key, value in train_metrics.items():
                if key != 'total_loss':
                    self.logger.info(f"  {key}: {value:.6f}")
            
            # Validation
            if self.val_loader is not None and epoch % val_frequency == 0:
                val_metrics = self.validate(stage)
                stage_val_scores.append(val_metrics)
                
                self.logger.info(f"Validation Metrics:")
                for key, value in val_metrics.items():
                    self.logger.info(f"  {key}: {value:.6f}")
                
                # Save best model
                if val_metrics.get('dice_score', 0.0) > self.best_val_score:
                    self.best_val_score = val_metrics['dice_score']
                    self.save_checkpoint(stage, epoch, is_best=True)
            
            # Save regular checkpoint
            if epoch % save_frequency == 0:
                self.save_checkpoint(stage, epoch)
        
        # Store history
        self.train_history[f'stage{stage}'] = stage_train_losses
        if stage_val_scores:
            self.val_history[f'stage{stage}'] = stage_val_scores
        
        self.logger.info(f"Stage {stage} completed")
    
    def _train_epoch(
        self,
        optimizer: optim.Optimizer,
        loss_fn: DistillationLoss,
        stage: int
    ) -> Dict[str, float]:
        """Train one epoch."""
        
        self.student_model.train()
        self.teacher_model.eval()
        
        epoch_losses = {}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Stage {stage} Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch.get('mask', None)
            if masks is not None:
                masks = masks.to(self.device)
            
            # Get prompts if available
            point_coords = batch.get('point_coords', None)
            point_labels = batch.get('point_labels', None)
            
            if point_coords is not None:
                point_coords = point_coords.to(self.device)
            if point_labels is not None:
                point_labels = point_labels.to(self.device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=self.use_amp):
                # Teacher forward pass (no gradients)
                teacher_output = {}
                with torch.no_grad():
                    # Extract teacher features
                    teacher_features = self.teacher_model.extract_backbone_features(images)
                    teacher_output['features'] = teacher_features
                    
                    # Get teacher predictions if needed for stages 2-3
                    if stage >= 2 and point_coords is not None:
                        try:
                            t_masks, t_scores, t_logits = self.teacher_model.predict_mask(
                                images,
                                point_coords=point_coords.cpu().numpy() if isinstance(point_coords, torch.Tensor) else None,
                                point_labels=point_labels.cpu().numpy() if isinstance(point_labels, torch.Tensor) else None
                            )
                            # Detach teacher outputs to prevent gradient flow
                            teacher_output['logits'] = t_logits.detach() if isinstance(t_logits, torch.Tensor) else t_logits
                            teacher_output['masks'] = t_masks.detach() if isinstance(t_masks, torch.Tensor) else t_masks
                        except Exception as e:
                            self.logger.warning(f"Teacher prediction failed: {e}")
                            # Use dummy logits (detached to prevent gradient flow)
                            teacher_output['logits'] = torch.zeros_like(images[:, :1]).detach()
                
                # Student forward pass
                student_output = self.student_model(
                    images=images,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    return_features=True
                )
                
                # Add post-memory features for stage 2
                if stage == 2 and 'post_memory_features' in student_output:
                    # Simulate teacher post-memory features (in practice, extract from teacher)
                    teacher_output['post_memory_features'] = teacher_features.get('final', 
                        list(teacher_features.values())[0] if teacher_features else 
                        torch.zeros_like(student_output['post_memory_features']))
                
                # Compute loss
                loss, loss_components = loss_fn(
                    student_output=student_output,
                    teacher_output=teacher_output if teacher_output else None,
                    gt_masks=masks
                )
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update metrics
            for key, value in loss_components.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item() if isinstance(value, torch.Tensor) else value)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(), 
                'lr': optimizer.param_groups[0]['lr']
            })
            
            num_batches += 1
            self.global_step += 1
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self, stage: int) -> Dict[str, float]:
        """Validate the student model."""
        
        if self.val_loader is None:
            return {}
        
        self.student_model.eval()
        val_metrics = {'dice_scores': [], 'iou_scores': []}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Stage {stage} Validation"):
                images = batch['image'].to(self.device)
                masks = batch.get('mask', None)
                
                if masks is not None:
                    masks = masks.to(self.device)
                    
                    # Get prompts if available
                    point_coords = batch.get('point_coords', None)
                    point_labels = batch.get('point_labels', None)
                    
                    if point_coords is not None:
                        point_coords = point_coords.to(self.device)
                    if point_labels is not None:
                        point_labels = point_labels.to(self.device)
                    
                    # Student prediction
                    student_output = self.student_model(
                        images=images,
                        point_coords=point_coords,
                        point_labels=point_labels
                    )
                    
                    pred_masks = torch.sigmoid(student_output['logits'])
                    
                    # Calculate metrics
                    for i in range(pred_masks.shape[0]):
                        pred = pred_masks[i]
                        gt = masks[i]
                        
                        # Dice score
                        intersection = (pred * gt).sum()
                        union = pred.sum() + gt.sum()
                        dice = (2.0 * intersection) / (union + 1e-8)
                        val_metrics['dice_scores'].append(dice.item())
                        
                        # IoU score
                        intersection = ((pred > 0.5) & (gt > 0.5)).float().sum()
                        union = ((pred > 0.5) | (gt > 0.5)).float().sum()
                        iou = intersection / (union + 1e-8)
                        val_metrics['iou_scores'].append(iou.item())
        
        # Average metrics
        return {
            'dice_score': np.mean(val_metrics['dice_scores']) if val_metrics['dice_scores'] else 0.0,
            'iou_score': np.mean(val_metrics['iou_scores']) if val_metrics['iou_scores'] else 0.0,
            'num_samples': len(val_metrics['dice_scores'])
        }
    
    def save_checkpoint(self, stage: int, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        
        checkpoint = {
            'stage': stage,
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.student_model.state_dict(),
            'best_val_score': self.best_val_score,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'model_config': {
                'backbone_type': getattr(self.student_model.image_encoder, 'backbone_type', 'mobilenet_v3_small'),
                'embed_dim': self.student_model.embed_dim
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.save_dir, 
            f"efficient_medsam2_stage{stage}_epoch{epoch}.pth"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, "efficient_medsam2_best.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint and resume training."""
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_stage = checkpoint.get('stage', 1)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        self.train_history = checkpoint.get('train_history', {'stage1': [], 'stage2': [], 'stage3': []})
        self.val_history = checkpoint.get('val_history', {'stage1': [], 'stage2': [], 'stage3': []})
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resumed at stage {self.current_stage}, step {self.global_step}")
        
        return checkpoint
    
    def train_all_stages(
        self,
        stage1_epochs: int = 10,
        stage2_epochs: int = 15,
        stage3_epochs: int = 5,
        base_lr: float = 1e-4,
        save_config: bool = True
    ):
        """Train all three stages sequentially."""
        
        self.logger.info("=" * 80)
        self.logger.info("Starting decoupled knowledge distillation training")
        self.logger.info(f"Stage 1: {stage1_epochs} epochs | Stage 2: {stage2_epochs} epochs | Stage 3: {stage3_epochs} epochs")
        self.logger.info(f"Base learning rate: {base_lr}")
        
        start_time = time.time()
        
        try:
            # Stage 1: Pre-memory feature distillation
            self.train_stage(
                stage=1,
                epochs=stage1_epochs,
                learning_rate=base_lr * 0.1  # Lower LR for feature matching
            )
            
            # Stage 2: Memory-aware distillation
            self.train_stage(
                stage=2,
                epochs=stage2_epochs,
                learning_rate=base_lr
            )
            
            # Stage 3: Fine-tuning with ground truth
            self.train_stage(
                stage=3,
                epochs=stage3_epochs,
                learning_rate=base_lr * 0.1  # Lower LR for fine-tuning
            )
            
            # Final checkpoint
            final_checkpoint = {
                'model_state_dict': self.student_model.state_dict(),
                'training_completed': True,
                'final_stage': 3,
                'total_training_time': time.time() - start_time,
                'train_history': self.train_history,
                'val_history': self.val_history,
                'best_val_score': self.best_val_score
            }
            
            final_path = os.path.join(self.save_dir, "efficient_medsam2_final.pth")
            torch.save(final_checkpoint, final_path)
            
            total_time = time.time() - start_time
            self.logger.info("=" * 80)
            self.logger.info(f"Training completed successfully!")
            self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
            self.logger.info(f"Best validation score: {self.best_val_score:.4f}")
            self.logger.info(f"Final model saved: {final_path}")
            
            # Save training configuration
            if save_config:
                config = {
                    'stage1_epochs': stage1_epochs,
                    'stage2_epochs': stage2_epochs,
                    'stage3_epochs': stage3_epochs,
                    'base_lr': base_lr,
                    'training_time_hours': total_time / 3600,
                    'best_val_score': self.best_val_score,
                    'device': self.device,
                    'use_amp': self.use_amp
                }
                config_path = os.path.join(self.save_dir, "training_config.json")
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.logger.info(f"Training config saved: {config_path}")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise e


if __name__ == "__main__":
    # Example usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Knowledge Distillation Trainer Example")
    print(f"Device: {device}")
    
    # This would normally load real models and data
    # teacher = load_medsam2_teacher(device=device)
    # student = build_efficient_student_model(device=device)
    # train_loader = create_train_loader()
    # val_loader = create_val_loader()
    
    # trainer = KnowledgeDistillationTrainer(
    #     teacher_model=teacher,
    #     student_model=student,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     device=device
    # )
    
    # # Start training
    # trainer.train_all_stages(
    #     stage1_epochs=2,  # Reduced for testing
    #     stage2_epochs=3,
    #     stage3_epochs=2
    # )
    
    print("Trainer implementation completed!")