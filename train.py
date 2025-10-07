"""
Training Script for EfficientMedSAM2 Knowledge Distillation
Run this script to start the 3-stage distillation process.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from teacher_model.medsam2 import load_medsam2_teacher
from efficient_medsam2.build_efficient_medsam2 import build_efficient_medsam2_model
from efficient_medsam2.efficient_medsam2_base import EfficientMedSAM2
from training.trainer import KnowledgeDistillationTrainer


class DummyMedicalDataset(Dataset):
    """
    Dummy dataset for testing. Replace with your actual medical dataset.
    
    For real training, you should load:
    - Medical images (CT, MRI scans)
    - Corresponding segmentation masks  
    - Point/box prompts for interactive segmentation
    """
    
    def __init__(self, num_samples=100, image_size=512):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy medical-like data
        # In practice, load real medical images and masks
        
        # Simulated CT/MRI slice (grayscale -> RGB)
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Simulated segmentation mask
        mask = torch.zeros(1, self.image_size, self.image_size)
        # Add some dummy lesion/organ regions
        center_x, center_y = np.random.randint(64, self.image_size-64, 2)
        size = np.random.randint(20, 60)
        mask[0, center_y-size:center_y+size, center_x-size:center_x+size] = 1.0
        
        # Simulated point prompt (center of mask)
        point_coords = torch.tensor([[center_x, center_y]], dtype=torch.float32)
        point_labels = torch.tensor([1], dtype=torch.long)  # positive point
        
        return {
            'image': image,
            'mask': mask,
            'point_coords': point_coords,
            'point_labels': point_labels
        }


def create_data_loaders(data_type="recist", batch_size=2, num_workers=0, **kwargs):
    """Create training and validation data loaders."""
    
    # Import medical dataset utilities
    from utils.medical_dataset import create_medical_data_loaders
    
    try:
        # Try to load real medical data first
        print(f"🔍 Loading {data_type} medical dataset...")
        train_loader, val_loader = create_medical_data_loaders(
            data_type=data_type,
            batch_size=batch_size,
            num_workers=num_workers,
            **kwargs
        )
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ Failed to load {data_type} dataset: {e}")
        print("🔄 Falling back to dummy dataset...")
        
        # Fallback to dummy dataset
        train_dataset = DummyMedicalDataset(num_samples=200)
        val_dataset = DummyMedicalDataset(num_samples=50)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        return train_loader, val_loader


def main():
    """Main training function."""
    
    print("🚀 Starting EfficientMedSAM2 Knowledge Distillation Training")
    print("=" * 60)
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    batch_size = 2  # Start small for testing
    learning_rate = 1e-4
    
    # Create directories for saving results
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Step 1: Load Teacher Model (MedSAM2)
        print("\n📚 Loading Teacher Model (MedSAM2)...")
        teacher_config = "sam2/configs/sam2.1_hiera_t512.yaml"
        teacher_checkpoint = "checkpoints/medsam2_model.pth"  # Download from HuggingFace: wanglab/MedSAM2
        
        teacher = load_medsam2_teacher(
            config_file=teacher_config,
            checkpoint_path=teacher_checkpoint,
            device=device
        )
        print("✅ Teacher model loaded successfully")
        
        # Step 2: Create Student Model (EfficientMedSAM2)
        print("\n🎓 Creating Student Model (EfficientMedSAM2)...")
        student = build_efficient_medsam2_model(
            encoder_type="mobilenet",  # Lightweight mobile architecture
            image_size=512,
            embed_dim=256,
            device=device
        )
        print("✅ Student model created successfully")
        
        # Step 3: Create Data Loaders
        print("\n📊 Creating Data Loaders...")
        # Options: "msd" (Medical Segmentation Decathlon), "recist" (CT lesion data), 
        #          "brain_mri" (Brain MRI Dataset.zip), "dummy" (testing)
        data_type = "msd"  # Using MSD dataset from Google Drive
        msd_task = "Task01_BrainTumour"  # Can change to other MSD tasks
        
        train_loader, val_loader = create_data_loaders(
            data_type=data_type,
            msd_task=msd_task,
            batch_size=batch_size,
            num_workers=0  # Set to 0 for debugging, increase for speed
        )
        print(f"✅ Data loaders created - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        # Step 4: Initialize Trainer
        print("\n🏋️ Initializing Knowledge Distillation Trainer...")
        trainer = KnowledgeDistillationTrainer(
            teacher_model=teacher,
            student_model=student,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            use_amp=True,  # Mixed precision for faster training
            save_dir="checkpoints",
            log_dir="logs"
        )
        print("✅ Trainer initialized successfully")
        
        # Step 5: Start Training (3-Stage Process)
        print("\n🎯 Starting 3-Stage Knowledge Distillation...")
        print("Stage 1: Feature Distillation (Backbone only)")
        print("Stage 2: Memory-Aware Distillation")  
        print("Stage 3: Fine-tuning with Ground Truth")
        
        trainer.train_all_stages(
            stage1_epochs=3,   # Reduced for testing - increase to 10+ for real training
            stage2_epochs=5,   # Reduced for testing - increase to 15+ for real training  
            stage3_epochs=2,   # Reduced for testing - increase to 5+ for real training
            base_lr=learning_rate
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"Best validation score: {trainer.best_val_score:.4f}")
        
        # Step 6: Test the trained model
        print("\n🧪 Testing trained model...")
        test_inference(student, device)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting tips:")
        print("1. Check if you have enough GPU memory (reduce batch_size if needed)")
        print("2. Verify dependencies are installed: torch, torchvision, timm")
        print("3. Make sure MedSAM2 path is correct in teacher model loading")


def test_inference(model, device):
    """Test the trained model with a sample input."""
    
    model.eval()
    
    # Create sample input
    sample_image = torch.randn(1, 3, 512, 512).to(device)
    sample_points = torch.tensor([[[256, 256]]], dtype=torch.float32).to(device)
    sample_labels = torch.tensor([[1]], dtype=torch.long).to(device)
    
    print("Running inference test...")
    
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        output = model(
            images=sample_image,
            point_coords=sample_points,
            point_labels=sample_labels
        )
        end_time.record()
        
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)
    
    print(f"✅ Inference successful!")
    print(f"   Output mask shape: {output['masks'].shape}")
    print(f"   IoU predictions shape: {output['iou_predictions'].shape}")
    print(f"   Inference time: {inference_time:.2f} ms")


if __name__ == "__main__":
    main()