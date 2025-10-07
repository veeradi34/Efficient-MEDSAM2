"""
Quick Setup and Environment Check for EfficientMedSAM2
Run this before training to verify your setup.
"""

import sys
import subprocess
import importlib
import torch

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

def check_cuda():
    """Check CUDA availability."""
    if torch.cuda.is_available():
        print(f"✅ CUDA available - {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available - will use CPU (slower)")
        return False

def install_missing_packages():
    """Install missing packages."""
    print("\n📦 Installing missing packages...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def main():
    """Main setup check."""
    print("🔧 EfficientMedSAM2 Setup Check")
    print("=" * 40)
    
    all_good = True
    
    # Check Python version
    if not check_python_version():
        all_good = False
    
    print("\n📚 Checking required packages...")
    
    # Core packages
    required_packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"), 
        ("timm", "timm"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("tqdm", "tqdm"),
        ("PIL", "PIL")
    ]
    
    missing_packages = []
    for pkg_name, import_name in required_packages:
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)
            all_good = False
    
    # Check CUDA
    print("\n🚀 Checking CUDA...")
    cuda_available = check_cuda()
    
    # Install missing packages if needed
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        
        response = input("\nWould you like to install missing packages? (y/n): ")
        if response.lower() == 'y':
            if install_missing_packages():
                print("✅ Setup complete! You can now run training.")
            else:
                print("❌ Setup failed. Please install packages manually.")
        else:
            print("⚠️  Please install missing packages before training.")
    
    elif all_good:
        print("\n🎉 All checks passed! Ready to start training.")
        print("\nNext steps:")
        print("1. Prepare your medical image dataset")
        print("2. Update the dataset loading in train.py")
        print("3. Run: python train.py")
        
        if not cuda_available:
            print("\n⚠️  Consider installing CUDA for faster training:")
            print("   https://pytorch.org/get-started/locally/")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    main()