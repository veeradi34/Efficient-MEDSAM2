"""
Medical Segmentation Decathlon (MSD) Dataset Loader
Supports loading MSD data from Google Drive or local storage.

MSD Tasks:
- Task01_BrainTumour: Brain tumor        # Find all .nii.gz files, excluding system files
        all_image_files = list(image_dir.glob("*.nii.gz"))
        image_files = sorted([f for f in all_image_files if not f.name.startswith('._')])
        label_files = []
        
        for img_file in image_files:
            # Find corresponding label file
            label_file = label_dir / img_file.name  
            if label_file.exists() and not label_file.name.startswith('._'):
                label_files.append(label_file)
            else:
                label_files.append(None)  # No label availableon (MRI)
- Task02_Heart: Cardiac segmentation (MRI)
- Task03_Liver: Liver tumor segmentation (CT)
- Task04_Hippocampus: Hippocampus segmentation (MRI)
- Task05_Prostate: Prostate segmentation (MRI)
- Task06_Lung: Lung tumor segmentation (CT)
- Task07_Pancreas: Pancreas segmentation (CT)
- Task08_HepaticVessel: Hepatic vessel segmentation (CT)
- Task09_Spleen: Spleen segmentation (CT)
- Task10_Colon: Colon cancer segmentation (CT)
"""

import os
import torch
import numpy as np
import nibabel as nib
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import gdown
import zipfile
import shutil


class MSDDataset(Dataset):
    """
    Medical Segmentation Decathlon Dataset Loader.
    Downloads and loads MSD tasks from Google Drive or local storage.
    """
    
    # Google Drive file IDs for MSD tasks (you may need to update these)
    MSD_DRIVE_IDS = {
        "Task01_BrainTumour": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
        "Task02_Heart": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2", 
        "Task03_Liver": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
        "Task04_Hippocampus": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
        "Task05_Prostate": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
        "Task06_Lung": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
        "Task07_Pancreas": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
        "Task08_HepaticVessel": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
        "Task09_Spleen": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
        "Task10_Colon": "1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2",
    }
    
    def __init__(
        self,
        task_name: str = "Task01_BrainTumour",
        data_dir: str = "./msd_data",
        split: str = "train",
        image_size: int = 512,
        slice_selection: str = "middle",  # "all", "middle", "random"
        download: bool = True,
        transform=None
    ):
        """
        Initialize MSD Dataset.
        
        Args:
            task_name: MSD task name (e.g., "Task01_BrainTumour")
            data_dir: Directory to store/load MSD data
            split: Dataset split ("train", "test", "val")
            image_size: Target image size for resizing
            slice_selection: How to select 2D slices from 3D volumes
            download: Whether to download data if not present
            transform: Optional data transforms
        """
        self.task_name = task_name
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.slice_selection = slice_selection
        self.transform = transform
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir = self.data_dir / task_name
        
        # Download data if needed
        if download and not self._is_data_available():
            self._download_msd_data()
            
        # Load dataset info
        self.dataset_info = self._load_dataset_info()
        
        # Get file lists
        self.image_files, self.label_files = self._get_file_lists()
        
        print(f"MSD {task_name} ({split}): {len(self.image_files)} samples")
        
    def _is_data_available(self) -> bool:
        """Check if MSD data is already downloaded."""
        return (self.task_dir.exists() and 
                (self.task_dir / "imagesTr").exists() and
                (self.task_dir / "labelsTr").exists())
    
    def _download_msd_data(self):
        """Download MSD data from Google Drive."""
        print(f"📥 Downloading {self.task_name} from Google Drive...")
        
        try:
            # Create download directory
            download_dir = self.data_dir / "downloads"
            download_dir.mkdir(exist_ok=True)
            
            # Download from Google Drive
            drive_id = self.MSD_DRIVE_IDS.get(self.task_name)
            if not drive_id:
                raise ValueError(f"No Google Drive ID for {self.task_name}")
                
            # Download file
            zip_path = download_dir / f"{self.task_name}.zip"
            url = f"https://drive.google.com/uc?id={drive_id}"
            
            print(f"Downloading from: {url}")
            gdown.download(url, str(zip_path), quiet=False)
            
            # Extract zip file
            print("📦 Extracting downloaded data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
                
            # Clean up
            zip_path.unlink()
            
            print(f"✅ {self.task_name} downloaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to download {self.task_name}: {e}")
            print("Please manually download the data and place it in the correct directory.")
            print(f"Expected structure: {self.task_dir}/imagesTr/, {self.task_dir}/labelsTr/")
            
    def _load_dataset_info(self) -> Dict:
        """Load dataset.json with task information."""
        dataset_json = self.task_dir / "dataset.json"
        
        if dataset_json.exists():
            with open(dataset_json, 'r') as f:
                return json.load(f)
        else:
            # Create basic info if not available
            return {
                "name": self.task_name,
                "description": f"MSD {self.task_name}",
                "modality": {"0": "CT" if "CT" in self.task_name else "MRI"},
                "labels": {"0": "background", "1": "target"}
            }
    
    def _get_file_lists(self) -> Tuple[List[Path], List[Path]]:
        """Get lists of image and label files."""
        if self.split == "train":
            image_dir = self.task_dir / "imagesTr"
            label_dir = self.task_dir / "labelsTr"
        else:
            # For test/val, might need different structure
            image_dir = self.task_dir / "imagesTs"
            label_dir = self.task_dir / "labelsTs"
            
            # Fallback to training data if test not available
            if not image_dir.exists():
                image_dir = self.task_dir / "imagesTr"
                label_dir = self.task_dir / "labelsTr"
        
        # Get all .nii.gz files
        image_files = sorted(list(image_dir.glob("*.nii.gz")))
        label_files = []
        
        for img_file in image_files:
            # Find corresponding label file
            label_file = label_dir / img_file.name
            if label_file.exists():
                label_files.append(label_file)
            else:
                label_files.append(None)  # No label available
                
        return image_files, label_files
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load NIfTI files
        img_path = self.image_files[idx]
        label_path = self.label_files[idx]
        
        # Load image
        img_nii = nib.load(str(img_path))
        image = img_nii.get_fdata()
        
        # Load label if available
        if label_path is not None:
            label_nii = nib.load(str(label_path))
            mask = label_nii.get_fdata()
        else:
            mask = np.zeros_like(image)
        
        # Select 2D slice from 3D volume
        image_2d, mask_2d = self._select_slice(image, mask)
        
        # Normalize image
        image_2d = self._normalize_image(image_2d)
        
        # Convert to RGB (duplicate grayscale channels)
        if len(image_2d.shape) == 2:
            image_2d = np.stack([image_2d, image_2d, image_2d], axis=0)
        else:
            image_2d = image_2d.transpose(2, 0, 1)  # HWC to CHW
            
        # Convert to tensors
        image_tensor = torch.from_numpy(image_2d.astype(np.float32))
        mask_tensor = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0)
        
        # Resize if needed
        if image_tensor.shape[-1] != self.image_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear'
            ).squeeze(0)
            
        if mask_tensor.shape[-1] != self.image_size:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='nearest'
            ).squeeze(0)
        
        # Generate point prompts
        point_coords, point_labels = self._generate_prompts_from_mask(mask_tensor)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'filename': img_path.name,
            'task': self.task_name
        }
    
    def _select_slice(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Select 2D slice from 3D volume."""
        if len(image.shape) == 2:
            return image, mask
            
        if self.slice_selection == "middle":
            # Take middle slice
            slice_idx = image.shape[2] // 2
        elif self.slice_selection == "random":
            # Random slice
            slice_idx = np.random.randint(0, image.shape[2])
        else:
            # Take middle as default
            slice_idx = image.shape[2] // 2
            
        return image[:, :, slice_idx], mask[:, :, slice_idx]
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        image = image.astype(np.float32)
        
        # Clip extreme values (optional)
        p1, p99 = np.percentile(image, (1, 99))
        image = np.clip(image, p1, p99)
        
        # Normalize to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image
    
    def _generate_prompts_from_mask(self, mask):
        """Generate point prompts from segmentation mask."""
        if mask.sum() == 0:
            # No mask, return random point
            h, w = mask.shape[-2:]
            point_coords = torch.tensor([[w//2, h//2]], dtype=torch.float32)
            point_labels = torch.tensor([0], dtype=torch.long)  # negative
        else:
            # Find center of mass
            mask_np = mask.squeeze().numpy()
            y_coords, x_coords = np.where(mask_np > 0.5)
            
            if len(y_coords) > 0:
                center_y = int(np.mean(y_coords))
                center_x = int(np.mean(x_coords))
                point_coords = torch.tensor([[center_x, center_y]], dtype=torch.float32)
                point_labels = torch.tensor([1], dtype=torch.long)  # positive
            else:
                h, w = mask.shape[-2:]
                point_coords = torch.tensor([[w//2, h//2]], dtype=torch.float32)
                point_labels = torch.tensor([0], dtype=torch.long)
                
        return point_coords, point_labels


def create_msd_data_loaders(
    task_name: str = "Task01_BrainTumour",
    batch_size: int = 2,
    num_workers: int = 0,
    train_split: float = 0.8,
    data_dir: str = "./msd_data",
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create MSD data loaders for training and validation.
    
    Args:
        task_name: MSD task name
        batch_size: Batch size
        num_workers: Number of worker processes
        train_split: Fraction for training (rest for validation)
        data_dir: Directory to store MSD data
        **kwargs: Additional arguments for MSDDataset
    
    Returns:
        train_loader, val_loader
    """
    
    # Create full dataset
    full_dataset = MSDDataset(
        task_name=task_name,
        data_dir=data_dir,
        split="train",
        download=True,
        **kwargs
    )
    
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
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Created MSD {task_name} data loaders:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


# Available MSD tasks
MSD_TASKS = {
    "Task01_BrainTumour": "Brain tumor segmentation (MRI)",
    "Task02_Heart": "Cardiac segmentation (MRI)", 
    "Task03_Liver": "Liver tumor segmentation (CT)",
    "Task04_Hippocampus": "Hippocampus segmentation (MRI)",
    "Task05_Prostate": "Prostate segmentation (MRI)",
    "Task06_Lung": "Lung tumor segmentation (CT)",
    "Task07_Pancreas": "Pancreas segmentation (CT)",
    "Task08_HepaticVessel": "Hepatic vessel segmentation (CT)",
    "Task09_Spleen": "Spleen segmentation (CT)",
    "Task10_Colon": "Colon cancer segmentation (CT)",
}


if __name__ == "__main__":
    print("🏥 Available MSD Tasks:")
    for task, desc in MSD_TASKS.items():
        print(f"  {task}: {desc}")
    
    print("\n📦 Testing MSD dataset loader...")
    try:
        # Test with Brain Tumor task
        train_loader, val_loader = create_msd_data_loaders(
            task_name="Task01_BrainTumour",
            batch_size=1
        )
        
        sample = next(iter(train_loader))
        print(f"Sample shapes: image={sample['image'].shape}, mask={sample['mask'].shape}")
        print(f"Task: {sample['task'][0]}, File: {sample['filename'][0]}")
        
    except Exception as e:
        print(f"❌ MSD dataset test failed: {e}")
        print("Make sure to install required packages: pip install gdown nibabel")