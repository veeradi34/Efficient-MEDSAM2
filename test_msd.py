"""
Test MSD Dataset Integration
Run this to verify MSD dataset functionality before training.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_msd_integration():
    """Test MSD dataset integration."""
    print("🏥 Testing Medical Segmentation Decathlon (MSD) Integration")
    print("=" * 60)
    
    # Test available tasks
    from utils.msd_dataset import MSD_TASKS
    print(f"📋 Available MSD Tasks ({len(MSD_TASKS)}):")
    for task, desc in MSD_TASKS.items():
        print(f"  • {task}: {desc}")
    
    # Test dataset creation (without download for now)
    print("\n🔧 Testing dataset creation...")
    try:
        from utils.msd_dataset import MSDDataset
        
        # Create dataset without downloading first
        dataset = MSDDataset(
            task_name="Task01_BrainTumour",
            data_dir="./msd_data",
            download=False  # Don't download yet
        )
        print("✅ MSDDataset class created successfully")
        
    except Exception as e:
        print(f"❌ MSDDataset creation failed: {e}")
    
    # Test data loader creation
    print("\n📊 Testing MSD data loader integration...")
    try:
        from utils.medical_dataset import create_medical_data_loaders
        
        print("Testing MSD data loader creation (will attempt download)...")
        train_loader, val_loader = create_medical_data_loaders(
            data_type="msd",
            msd_task="Task01_BrainTumour",
            batch_size=1,
            num_workers=0
        )
        
        print(f"✅ Data loaders created successfully!")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
        
        # Test loading a sample
        sample = next(iter(train_loader))
        print(f"✅ Sample loaded: image {sample['image'].shape}, mask {sample['mask'].shape}")
        
    except Exception as e:
        print(f"❌ MSD data loader test failed: {e}")
        print("This is expected if MSD data hasn't been downloaded yet.")
    
    print("\n" + "=" * 60)
    print("🚀 How to use MSD dataset:")
    print("1. The dataset will auto-download from Google Drive")
    print("2. Change data_type to 'msd' in train.py")
    print("3. Choose your preferred MSD task (Task01_BrainTumour, etc.)")
    print("4. Run training: python train.py")
    print("\n📂 Data will be stored in: ./msd_data/")

if __name__ == "__main__":
    test_msd_integration()