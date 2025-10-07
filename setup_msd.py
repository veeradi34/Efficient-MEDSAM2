"""
MSD Dataset Manual Download Guide

Since the Google Drive folder requires manual download, here's how to set up MSD data:
"""

import os
from pathlib import Path

def create_msd_setup_guide():
    """Create setup guide for MSD dataset."""
    
    guide = """
🏥 Medical Segmentation Decathlon (MSD) Setup Guide
================================================

The MSD dataset from your Google Drive link needs to be downloaded manually.

📂 Google Drive Folder: https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2

🔽 Manual Download Steps:
1. Visit the Google Drive folder above
2. Download the task(s) you want (e.g., Task01_BrainTumour.tar)
3. Extract the downloaded files
4. Organize them in the structure below

📁 Expected Directory Structure:
Distillation/
├── msd_data/
│   ├── Task01_BrainTumour/
│   │   ├── dataset.json
│   │   ├── imagesTr/
│   │   │   ├── BRATS_001.nii.gz
│   │   │   ├── BRATS_002.nii.gz
│   │   │   └── ...
│   │   └── labelsTr/
│   │       ├── BRATS_001.nii.gz
│   │       ├── BRATS_002.nii.gz
│   │       └── ...
│   ├── Task03_Liver/
│   │   ├── dataset.json
│   │   ├── imagesTr/
│   │   └── labelsTr/
│   └── ... (other tasks)

🎯 Available MSD Tasks:
• Task01_BrainTumour: Brain tumor segmentation (MRI) - 484 cases
• Task02_Heart: Cardiac segmentation (MRI) - 20 cases  
• Task03_Liver: Liver tumor segmentation (CT) - 131 cases
• Task04_Hippocampus: Hippocampus segmentation (MRI) - 260 cases
• Task05_Prostate: Prostate segmentation (MRI) - 32 cases
• Task06_Lung: Lung tumor segmentation (CT) - 63 cases
• Task07_Pancreas: Pancreas segmentation (CT) - 281 cases
• Task08_HepaticVessel: Hepatic vessel segmentation (CT) - 443 cases
• Task09_Spleen: Spleen segmentation (CT) - 41 cases
• Task10_Colon: Colon cancer segmentation (CT) - 126 cases

💡 Recommended Tasks for Testing:
1. Task01_BrainTumour (largest dataset, good for training)
2. Task03_Liver (medium size, CT imaging)
3. Task07_Pancreas (challenging segmentation task)

🚀 After Download:
1. Make sure files are in correct structure above
2. Run: python test_msd.py (to verify setup)
3. Run: python train.py (to start training)

⚙️ Training Configuration:
In train.py, set:
    data_type = "msd"
    msd_task = "Task01_BrainTumour"  # or your chosen task
"""
    
    print(guide)
    
    # Create directory structure
    msd_dir = Path("./msd_data")
    msd_dir.mkdir(exist_ok=True)
    
    # Save guide to file
    with open("MSD_SETUP_GUIDE.md", "w") as f:
        f.write(guide)
    
    print(f"📝 Setup guide saved to: MSD_SETUP_GUIDE.md")
    print(f"📂 MSD data directory created: {msd_dir.absolute()}")

if __name__ == "__main__":
    create_msd_setup_guide()