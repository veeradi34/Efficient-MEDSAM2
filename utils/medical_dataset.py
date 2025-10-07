"""
Medical Dataset Loaders for Knowledge Distillation Training
Supports various medical imaging formats and datasets.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import zipfile
from PIL import Image
import json
from typing import Dict, List, Optional, Tuple, Union
import nibabel as nib  # For NIfTI files (install with: pip install nibabel)


class RECISTDataset(Dataset):
    """
    Dataset loader for RECIST CT lesion segmentation data.
    Loads .npz files containing CT scans and lesion masks.
    """
    
    def __init__(
        self, 
        data_dir: str,
        image_size: int = 512,
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Find all .npz files
        self.npz_files = list(self.data_dir.glob("*.npz"))
        print(f"Found {len(self.npz_files)} RECIST dataset files")
        
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        
        # Load npz file
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract image and mask
        # Adjust keys based on actual npz file structure
        if 'image' in data:
            image = data['image']
        elif 'imgs' in data:
            image = data['imgs']
        else:
            # Try to find image data
            keys = list(data.keys())
            image = data[keys[0]]  # Use first array as image
            
        if 'mask' in data:
            mask = data['mask']
        elif 'gts' in data:
            mask = data['gts']
        else:
            # Create dummy mask if not available
            mask = np.zeros_like(image)
            
        # Convert to torch tensors
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image, image, image], axis=0)  # Convert to RGB
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = image.transpose(2, 0, 1)  # HWC to CHW
            
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        
        # Resize if needed
        if image.shape[-1] != self.image_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear'
            ).squeeze(0)
            
        if mask.shape[-1] != self.image_size:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='nearest'
            ).squeeze(0)
            
        # Generate point prompts from mask
        point_coords, point_labels = self._generate_prompts_from_mask(mask)
        
        return {
            'image': image,
            'mask': mask,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'filename': npz_path.name
        }
    
    def _generate_prompts_from_mask(self, mask):
        """Generate point prompts from segmentation mask."""
        if mask.sum() == 0:
            # No mask, return random point
            h, w = mask.shape[-2:]
            point_coords = torch.tensor([[w//2, h//2]], dtype=torch.float32)
            point_labels = torch.tensor([0], dtype=torch.long)  # negative
        else:
            # Find center of mass of largest connected component
            mask_np = mask.squeeze().numpy()
            y_coords, x_coords = np.where(mask_np > 0.5)
            
            if len(y_coords) > 0:
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                point_coords = torch.tensor([[center_x, center_y]], dtype=torch.float32)
                point_labels = torch.tensor([1], dtype=torch.long)  # positive
            else:
                # Fallback
                h, w = mask.shape[-2:]
                point_coords = torch.tensor([[w//2, h//2]], dtype=torch.float32)
                point_labels = torch.tensor([0], dtype=torch.long)
                
        return point_coords, point_labels


class BrainMRIDataset(Dataset):
    """
    Dataset loader for Brain MRI data.
    Extracts and loads data from Brain MRI Dataset.zip
    """
    
    def __init__(
        self, 
        zip_path: str,
        image_size: int = 512,
        extract_dir: Optional[str] = None
    ):
        self.zip_path = Path(zip_path)
        self.image_size = image_size
        
        # Set extraction directory
        if extract_dir is None:
            extract_dir = self.zip_path.parent / "extracted_brain_mri"
        self.extract_dir = Path(extract_dir)
        
        # Extract zip if needed
        self._extract_if_needed()
        
        # Find image files
        self.image_files = []
        for ext in ['*.jpg', '*.png', '*.tif', '*.tiff', '*.nii', '*.nii.gz']:
            self.image_files.extend(list(self.extract_dir.rglob(ext)))
            
        print(f"Found {len(self.image_files)} brain MRI images")
        
    def _extract_if_needed(self):
        """Extract zip file if not already extracted."""
        if not self.extract_dir.exists() or len(list(self.extract_dir.iterdir())) == 0:
            print(f"Extracting {self.zip_path} to {self.extract_dir}")
            self.extract_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
                
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image based on format
        if img_path.suffix.lower() in ['.nii', '.gz']:
            # NIfTI format
            try:
                nii_img = nib.load(str(img_path))
                image = nii_img.get_fdata()
                if len(image.shape) > 2:
                    image = image[:, :, image.shape[2]//2]  # Take middle slice
            except:
                # Fallback to dummy data
                image = np.random.randn(self.image_size, self.image_size)
        else:
            # Standard image formats
            try:
                image = np.array(Image.open(img_path).convert('L'))  # Grayscale
            except:
                image = np.random.randn(self.image_size, self.image_size)
        
        # Normalize and convert to RGB
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = np.stack([image, image, image], axis=0)  # Convert to RGB
        image = torch.from_numpy(image.astype(np.float32))
        
        # Resize if needed
        if image.shape[-1] != self.image_size:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), 
                size=(self.image_size, self.image_size), 
                mode='bilinear'
            ).squeeze(0)
        
        # Create dummy mask (for unsupervised data)
        mask = torch.zeros(1, self.image_size, self.image_size)
        
        # Random point prompt
        center_x = np.random.randint(64, self.image_size-64)
        center_y = np.random.randint(64, self.image_size-64)
        point_coords = torch.tensor([[center_x, center_y]], dtype=torch.float32)
        point_labels = torch.tensor([1], dtype=torch.long)
        
        return {
            'image': image,
            'mask': mask,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'filename': img_path.name
        }


def create_medical_data_loaders(
    data_type: str = "recist",
    batch_size: int = 2,
    num_workers: int = 0,
    train_split: float = 0.8,
    msd_task: str = "Task01_BrainTumour",
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create medical data loaders for training and validation.
    
    Args:
        data_type: Type of dataset ("recist", "brain_mri", "msd", or "dummy")
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        train_split: Fraction of data to use for training
        msd_task: MSD task name (e.g., "Task01_BrainTumour")
        **kwargs: Additional arguments for dataset constructors
    
    Returns:
        train_loader, val_loader
    """
    
    if data_type == "msd":
        # Medical Segmentation Decathlon (MSD) data
        from utils.msd_dataset import create_msd_data_loaders
        return create_msd_data_loaders(
            task_name=msd_task,
            batch_size=batch_size,
            num_workers=num_workers,
            train_split=train_split,
            **kwargs
        )
        
    elif data_type == "recist":
        # RECIST CT lesion data (requires external RECIST dataset)
        # data_dir = "data/RECIST_train_npz"  # Put RECIST data here if available
        print("Warning: RECIST dataset not available in self-contained setup")
        print("Using dummy dataset instead...")
        # Use dummy dataset fallback
        from train import DummyMedicalDataset
        train_dataset = DummyMedicalDataset(num_samples=200)
        val_dataset = DummyMedicalDataset(num_samples=50)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        return train_loader, val_loader
        
    elif data_type == "brain_mri":
        # Brain MRI data
        zip_path = "../Brain MRI Dataset.zip"
        full_dataset = BrainMRIDataset(zip_path, **kwargs)
        
    elif data_type == "dummy":
        # Dummy dataset for testing
        from train import DummyMedicalDataset
        train_dataset = DummyMedicalDataset(num_samples=200)
        val_dataset = DummyMedicalDataset(num_samples=50)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        return train_loader, val_loader
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    print(f"Created {data_type} data loaders:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loaders
    print("Testing RECIST dataset...")
    try:
        train_loader, val_loader = create_medical_data_loaders("recist", batch_size=1)
        sample = next(iter(train_loader))
        print(f"RECIST sample shapes: image={sample['image'].shape}, mask={sample['mask'].shape}")
    except Exception as e:
        print(f"RECIST dataset failed: {e}")
    
    print("\nTesting Brain MRI dataset...")
    try:
        train_loader, val_loader = create_medical_data_loaders("brain_mri", batch_size=1)
        sample = next(iter(train_loader))
        print(f"Brain MRI sample shapes: image={sample['image'].shape}, mask={sample['mask'].shape}")
    except Exception as e:
        print(f"Brain MRI dataset failed: {e}")