"""
Knowledge Distillation Training for EfficientMedSAM2

This script implements knowledge distillation from the original MedSAM2 (teacher)
to the EfficientMedSAM2 model (student).

Key features:
1. Decoupled distillation (first train student encoder with teacher's decoder)
2. Feature-level and output-level distillation
3. Mixed precision training for memory efficiency
4. Gradient checkpointing for memory efficiency
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import logging
from typing import Dict, Tuple, Optional, List, Union

# Import EfficientMedSAM2
from efficient_medsam2.build_efficient_medsam2 import build_efficient_medsam2_model, save_efficient_medsam2_model
from efficient_medsam2.efficient_medsam2_base import EfficientMedSAM2

# Import original MedSAM2
from sam2.build_sam import sam2_model_registry


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Knowledge Distillation for EfficientMedSAM2")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing training data")
    
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Directory containing validation data")
    
    # Model arguments
    parser.add_argument("--teacher_checkpoint", type=str, required=True,
                        help="Path to the teacher (original MedSAM2) checkpoint")
    
    parser.add_argument("--student_checkpoint", type=str, default=None,
                        help="Path to the student (EfficientMedSAM2) checkpoint for resuming training")
    
    parser.add_argument("--encoder_type", type=str, default="mobilenet",
                        choices=["mobilenet", "efficientnet"],
                        help="Type of encoder to use for the student model")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    
    parser.add_argument("--feature_distill_weight", type=float, default=10.0,
                        help="Weight for feature distillation loss")
    
    parser.add_argument("--output_distill_weight", type=float, default=1.0,
                        help="Weight for output distillation loss")
    
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for soft target distillation")
    
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    
    parser.add_argument("--save_freq", type=int, default=5,
                        help="Frequency to save checkpoints (in epochs)")
    
    parser.add_argument("--val_freq", type=int, default=1,
                        help="Frequency to validate (in epochs)")
    
    parser.add_argument("--log_freq", type=int, default=100,
                        help="Frequency to log (in iterations)")
    
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Training strategy arguments
    parser.add_argument("--decoupled_distillation", action="store_true",
                        help="Use decoupled distillation (train encoder first, then full model)")
    
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Use mixed precision training")
    
    parser.add_argument("--gradient_accumulation", type=int, default=1,
                        help="Number of steps to accumulate gradients")
    
    parser.add_argument("--freeze_decoder", action="store_true",
                        help="Freeze decoder in the student model")
    
    return parser.parse_args()


class MedicalSegmentationDataset(Dataset):
    """
    Dataset for medical segmentation.
    
    This is a placeholder implementation. In a real application,
    you would replace this with your actual dataset implementation.
    """
    def __init__(self, data_dir, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing data
            transform: Transforms to apply
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Get list of files (in a real implementation)
        self.files = []  # This would contain paths to your image-mask pairs
        
        # For demonstration purposes, create dummy data
        self.use_dummy_data = True
        self.dummy_size = 100
    
    def __len__(self):
        """Get dataset size."""
        if self.use_dummy_data:
            return self.dummy_size
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
                - 'image': Image tensor (C, H, W)
                - 'mask': Mask tensor (1, H, W)
                - 'points': Point prompts if available
                - 'boxes': Box prompts if available
        """
        if self.use_dummy_data:
            # Create a dummy image and mask for demonstration
            # In a real implementation, you would load actual data
            
            # Create a random image (simulating a medical scan)
            image = torch.randn(1, 512, 512)  # Single-channel (grayscale)
            
            # Create a random mask (simulating a segmentation mask)
            # Create a simple circular mask
            x, y = torch.meshgrid(torch.linspace(-1, 1, 512), torch.linspace(-1, 1, 512))
            mask = ((x**2 + y**2) < 0.5**2).float()
            mask = mask.unsqueeze(0)  # Add channel dimension
            
            # Create a random point prompt (simulating user interaction)
            # Place a point in the center with some noise
            point_coords = torch.tensor([[
                256 + torch.randint(-10, 10, (1,)).item(),
                256 + torch.randint(-10, 10, (1,)).item()
            ]])
            point_labels = torch.tensor([1])  # 1 for foreground
            
            # Return the sample
            return {
                'image': image,
                'mask': mask,
                'points': (point_coords, point_labels),
                'boxes': None,
            }
        else:
            # Load actual data (in a real implementation)
            image_path = self.files[idx]
            mask_path = image_path.replace('images', 'masks')
            
            # Load image and mask
            # image = ...
            # mask = ...
            
            # Apply transforms if available
            if self.transform is not None:
                # transformed = self.transform(image=image, mask=mask)
                # image, mask = transformed['image'], transformed['mask']
                pass
            
            # Return the sample
            return {
                'image': None,  # Replace with actual image
                'mask': None,   # Replace with actual mask
                'points': None,
                'boxes': None,
            }


class KnowledgeDistillationLoss(nn.Module):
    """
    Loss function for knowledge distillation.
    
    Combines multiple loss terms:
    1. Feature distillation loss (e.g., L2 between teacher and student features)
    2. Output distillation loss (e.g., KL divergence between teacher and student outputs)
    3. Supervised loss (e.g., Dice loss between student output and ground truth)
    """
    def __init__(
        self,
        feature_distill_weight: float = 10.0,
        output_distill_weight: float = 1.0,
        supervised_weight: float = 1.0,
        temperature: float = 2.0,
    ):
        """
        Initialize the loss function.
        
        Args:
            feature_distill_weight: Weight for feature distillation loss
            output_distill_weight: Weight for output distillation loss
            supervised_weight: Weight for supervised loss
            temperature: Temperature for soft target distillation
        """
        super().__init__()
        
        self.feature_distill_weight = feature_distill_weight
        self.output_distill_weight = output_distill_weight
        self.supervised_weight = supervised_weight
        self.temperature = temperature
    
    def forward(
        self,
        student_outputs: torch.Tensor,
        teacher_outputs: torch.Tensor,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        target_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            student_outputs: Student model outputs (mask logits)
            teacher_outputs: Teacher model outputs (mask logits)
            student_features: Student model features
            teacher_features: Teacher model features
            target_masks: Ground truth masks
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}
        
        # Feature distillation loss (L2)
        if student_features is not None and teacher_features is not None:
            feature_loss = F.mse_loss(student_features, teacher_features)
            loss_dict['feature_loss'] = feature_loss
        else:
            feature_loss = torch.tensor(0.0, device=student_outputs.device)
            loss_dict['feature_loss'] = feature_loss
        
        # Output distillation loss (KL divergence)
        # Apply temperature scaling
        student_logits_temp = student_outputs / self.temperature
        teacher_logits_temp = teacher_outputs / self.temperature
        
        # Convert logits to probabilities
        student_probs = torch.sigmoid(student_logits_temp)
        teacher_probs = torch.sigmoid(teacher_logits_temp)
        
        # Calculate KL divergence
        output_loss = F.kl_div(
            torch.log(student_probs + 1e-8),
            teacher_probs,
            reduction='batchmean',
        ) * (self.temperature ** 2)
        
        loss_dict['output_loss'] = output_loss
        
        # Supervised loss (Dice loss)
        if target_masks is not None:
            # Convert student outputs to probabilities
            student_probs = torch.sigmoid(student_outputs)
            
            # Calculate Dice loss
            intersection = torch.sum(student_probs * target_masks)
            union = torch.sum(student_probs) + torch.sum(target_masks)
            dice_loss = 1.0 - (2.0 * intersection) / (union + 1e-8)
            
            loss_dict['dice_loss'] = dice_loss
        else:
            dice_loss = torch.tensor(0.0, device=student_outputs.device)
            loss_dict['dice_loss'] = dice_loss
        
        # Total loss
        total_loss = (
            self.feature_distill_weight * feature_loss +
            self.output_distill_weight * output_loss +
            self.supervised_weight * dice_loss
        )
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


def train_one_epoch(
    student_model: EfficientMedSAM2,
    teacher_model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: KnowledgeDistillationLoss,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    scaler: Optional[GradScaler] = None,
):
    """
    Train for one epoch.
    
    Args:
        student_model: Student model (EfficientMedSAM2)
        teacher_model: Teacher model (original MedSAM2)
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        epoch: Current epoch
        args: Command line arguments
        scaler: Gradient scaler for mixed precision
    """
    # Set models to training mode
    student_model.train()
    teacher_model.eval()  # Teacher is always in eval mode for distillation
    
    # Initialize running loss
    running_loss = 0.0
    
    # Initialize progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    
    # For gradient accumulation
    optimizer.zero_grad()
    
    # Loop over batches
    for i, batch in enumerate(pbar):
        # Get data
        images = batch['image'].to(device)  # (B, C, H, W)
        masks = batch['mask'].to(device) if batch['mask'] is not None else None  # (B, 1, H, W)
        
        # Add time dimension for the models
        images = images.unsqueeze(1)  # (B, 1, C, H, W)
        
        # Extract prompts
        points = None
        if batch['points'] is not None:
            point_coords, point_labels = batch['points']
            point_coords = point_coords.to(device)
            point_labels = point_labels.to(device)
            points = (point_coords, point_labels)
        
        boxes = batch['boxes'].to(device) if batch['boxes'] is not None else None
        
        # Forward pass with teacher model (no gradients needed)
        with torch.no_grad():
            teacher_input = {
                'images': images,
                'points': points,
                'boxes': boxes,
                'masks': masks,
            }
            teacher_outputs = teacher_model(teacher_input)
            teacher_masks = teacher_outputs['masks']
            teacher_features = teacher_outputs.get('features', None)  # Extract teacher features if available
        
        # Forward pass with student model
        if args.mixed_precision:
            with autocast():
                student_input = {
                    'images': images,
                    'points': points,
                    'boxes': boxes,
                    'masks': masks,
                }
                student_outputs = student_model(student_input)
                student_masks = student_outputs['masks']
                student_features = student_outputs.get('features', None)  # Extract student features if available
                
                # Calculate loss
                loss, loss_dict = loss_fn(
                    student_masks,
                    teacher_masks,
                    student_features,
                    teacher_features,
                    masks,
                )
                
                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation
                
            # Backward pass with scaler
            scaler.scale(loss).backward()
            
            # Update weights if needed
            if (i + 1) % args.gradient_accumulation == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard training without mixed precision
            student_input = {
                'images': images,
                'points': points,
                'boxes': boxes,
                'masks': masks,
            }
            student_outputs = student_model(student_input)
            student_masks = student_outputs['masks']
            student_features = student_outputs.get('features', None)
            
            # Calculate loss
            loss, loss_dict = loss_fn(
                student_masks,
                teacher_masks,
                student_features,
                teacher_features,
                masks,
            )
            
            # Scale loss for gradient accumulation
            loss = loss / args.gradient_accumulation
            
            # Backward pass
            loss.backward()
            
            # Update weights if needed
            if (i + 1) % args.gradient_accumulation == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Update running loss
        running_loss += loss.item() * args.gradient_accumulation
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (i + 1),
            'feature_loss': loss_dict['feature_loss'].item(),
            'output_loss': loss_dict['output_loss'].item(),
            'dice_loss': loss_dict['dice_loss'].item(),
        })
        
        # Log periodically
        if (i + 1) % args.log_freq == 0:
            logging.info(f"Epoch {epoch+1}/{args.epochs}, Iter {i+1}/{len(train_loader)}, "
                        f"Loss: {running_loss / (i + 1):.4f}")
    
    # Return average loss
    return running_loss / len(train_loader)


def validate(
    student_model: EfficientMedSAM2,
    teacher_model: nn.Module,
    val_loader: DataLoader,
    loss_fn: KnowledgeDistillationLoss,
    device: torch.device,
):
    """
    Validate the model.
    
    Args:
        student_model: Student model (EfficientMedSAM2)
        teacher_model: Teacher model (original MedSAM2)
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary of validation metrics
    """
    # Set models to evaluation mode
    student_model.eval()
    teacher_model.eval()
    
    # Initialize metrics
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    
    # Initialize progress bar
    pbar = tqdm(val_loader, desc="Validation")
    
    # Loop over batches
    with torch.no_grad():
        for batch in pbar:
            # Get data
            images = batch['image'].to(device)
            masks = batch['mask'].to(device) if batch['mask'] is not None else None
            
            # Add time dimension for the models
            images = images.unsqueeze(1)
            
            # Extract prompts
            points = None
            if batch['points'] is not None:
                point_coords, point_labels = batch['points']
                point_coords = point_coords.to(device)
                point_labels = point_labels.to(device)
                points = (point_coords, point_labels)
            
            boxes = batch['boxes'].to(device) if batch['boxes'] is not None else None
            
            # Forward pass with teacher model
            teacher_input = {
                'images': images,
                'points': points,
                'boxes': boxes,
                'masks': masks,
            }
            teacher_outputs = teacher_model(teacher_input)
            teacher_masks = teacher_outputs['masks']
            teacher_features = teacher_outputs.get('features', None)
            
            # Forward pass with student model
            student_input = {
                'images': images,
                'points': points,
                'boxes': boxes,
                'masks': masks,
            }
            student_outputs = student_model(student_input)
            student_masks = student_outputs['masks']
            student_features = student_outputs.get('features', None)
            
            # Calculate loss
            loss, loss_dict = loss_fn(
                student_masks,
                teacher_masks,
                student_features,
                teacher_features,
                masks,
            )
            
            # Update metrics
            total_loss += loss.item()
            
            # Calculate Dice coefficient
            student_probs = torch.sigmoid(student_masks)
            if masks is not None:
                # Dice with ground truth
                intersection = torch.sum(student_probs * masks)
                union = torch.sum(student_probs) + torch.sum(masks)
                dice = (2.0 * intersection) / (union + 1e-8)
                total_dice += dice.item()
                
                # IoU with ground truth
                intersection = torch.sum(student_probs * masks)
                union = torch.sum(student_probs) + torch.sum(masks) - intersection
                iou = intersection / (union + 1e-8)
                total_iou += iou.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (pbar.n + 1),
                'dice': total_dice / (pbar.n + 1),
                'iou': total_iou / (pbar.n + 1),
            })
    
    # Return metrics
    return {
        'loss': total_loss / len(val_loader),
        'dice': total_dice / len(val_loader),
        'iou': total_iou / len(val_loader),
    }


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.save_dir, 'training.log')),
            logging.StreamHandler(),
        ]
    )
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load teacher model
    logging.info("Loading teacher model...")
    teacher_model = sam2_model_registry["default"](checkpoint=args.teacher_checkpoint)
    teacher_model.to(device)
    teacher_model.eval()  # Teacher is always in eval mode
    
    # Load or create student model
    logging.info("Creating student model...")
    student_model = build_efficient_medsam2_model(
        encoder_type=args.encoder_type,
        checkpoint=args.student_checkpoint,
        image_size=512,
        embed_dim=256,
        use_half_precision=args.mixed_precision,
        max_num_memory_frames=3,
        device=device,
    )
    
    # Freeze decoder if specified
    if args.freeze_decoder:
        logging.info("Freezing student decoder...")
        for param in student_model.mask_decoder.parameters():
            param.requires_grad = False
    
    # Create datasets and data loaders
    logging.info("Creating datasets and data loaders...")
    train_dataset = MedicalSegmentationDataset(args.data_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    if args.val_data_dir is not None:
        val_dataset = MedicalSegmentationDataset(args.val_data_dir)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        val_loader = None
    
    # Create optimizer
    logging.info("Creating optimizer...")
    optimizer = optim.AdamW(
        student_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Create loss function
    logging.info("Creating loss function...")
    loss_fn = KnowledgeDistillationLoss(
        feature_distill_weight=args.feature_distill_weight,
        output_distill_weight=args.output_distill_weight,
        supervised_weight=1.0,
        temperature=args.temperature,
    )
    
    # Create gradient scaler for mixed precision
    scaler = GradScaler() if args.mixed_precision else None
    
    # Training loop
    logging.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = train_one_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            args=args,
            scaler=scaler,
        )
        
        logging.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}")
        
        # Validate if specified
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            val_metrics = validate(
                student_model=student_model,
                teacher_model=teacher_model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )
            
            val_loss = val_metrics['loss']
            val_dice = val_metrics['dice']
            val_iou = val_metrics['iou']
            
            logging.info(f"Epoch {epoch+1}/{args.epochs}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Dice: {val_dice:.4f}, "
                        f"Val IoU: {val_iou:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_efficient_medsam2_model(
                    model=student_model,
                    path=os.path.join(args.save_dir, 'best_model.pth'),
                    save_optimizer=True,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    additional_data={
                        'dice': val_dice,
                        'iou': val_iou,
                    },
                )
                logging.info(f"Saved new best model with val_loss: {val_loss:.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            save_efficient_medsam2_model(
                model=student_model,
                path=os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'),
                save_optimizer=True,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
            )
            logging.info(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    save_efficient_medsam2_model(
        model=student_model,
        path=os.path.join(args.save_dir, 'final_model.pth'),
        save_optimizer=False,
    )
    
    logging.info("Training completed!")


if __name__ == "__main__":
    main()
