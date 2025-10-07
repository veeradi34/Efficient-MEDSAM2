"""
EfficientMedSAM2 Web Interface

This script implements a Streamlit web interface for the EfficientMedSAM2 model,
allowing users to upload medical images and perform segmentation using point prompts.

Example usage:
    streamlit run web_interface.py

Requirements:
    - streamlit
    - numpy
    - torch
    - matplotlib
    - pillow
    - nibabel (for NIfTI support)
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
import streamlit as st
from typing import Tuple, Dict, List, Optional, Union
import io
import base64
import sys

# Import EfficientMedSAM2
from efficient_medsam2.build_efficient_medsam2 import build_efficient_medsam2_model
from efficient_medsam2.efficient_medsam2_image_predictor import EfficientMedSAM2ImagePredictor

# Import system modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# For demonstration purposes, we'll use mock implementations
# In a real scenario, you would import the actual models from sam2 and efficient_medsam2


def load_image_data(uploaded_file):
    """
    Load an uploaded image file.
    
    Args:
        uploaded_file: Uploaded file from Streamlit
        
    Returns:
        Tuple of (image data, file type)
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_type in ['nii', 'gz']:
            # Handle NIfTI files
            bytes_data = uploaded_file.getvalue()
            with io.BytesIO(bytes_data) as f:
                nifti_img = nib.load(f)
                image_data = nifti_img.get_fdata()
                
                # Normalize to [0, 1]
                image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
                
                # Convert to float32
                image_data = image_data.astype(np.float32)
                
                return image_data, 'nifti'
        
        elif file_type in ['jpg', 'jpeg', 'png', 'bmp']:
            # Handle regular image files
            img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
            image_data = np.array(img).astype(np.float32) / 255.0
            
            return image_data, 'image'
        
        elif file_type == 'npz':
            # Handle NPZ files
            npz_data = np.load(io.BytesIO(uploaded_file.getvalue()), allow_pickle=True)
            
            # Check if the NPZ has 'image' key
            if 'image' in npz_data:
                image_data = npz_data['image']
            else:
                # Try to get the first array in the NPZ
                for key in npz_data.keys():
                    image_data = npz_data[key]
                    break
            
            # Normalize to [0, 1] if needed
            if image_data.max() > 1.0:
                image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
            
            # Convert to float32
            image_data = image_data.astype(np.float32)
            
            return image_data, 'npz'
        
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None, None
            
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None


@st.cache_resource
def load_models(original_model_path, efficient_model_path, encoder_type, use_half_precision):
    """
    Load the original and efficient models.
    
    Args:
        original_model_path: Path to original MedSAM2 model checkpoint
        efficient_model_path: Path to EfficientMedSAM2 model checkpoint
        encoder_type: Type of encoder to use for EfficientMedSAM2 ('mobilenet' or 'efficientnet')
        use_half_precision: Whether to use half precision for EfficientMedSAM2
        
    Returns:
        Tuple of (original predictor, efficient predictor)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For demonstration purposes, we'll use mock models
    # In a real scenario, you would load the actual models
    
    class MockOriginalModel(torch.nn.Module):
        """Mock implementation of MedSAM2 for demonstration"""
        def __init__(self):
            super().__init__()
            # Create some dummy layers to simulate model structure
            self.conv = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.transformer = torch.nn.Sequential(
                torch.nn.Linear(16, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 16)
            )
            # Add memory usage information for demo
            self.memory_usage = 15000  # MB
            
        def forward(self, x):
            # For demo, just return random masks
            if isinstance(x, dict):
                batch_size = 1
                return {
                    "masks": torch.sigmoid(torch.randn(batch_size, 3, 512, 512).to(device)),
                    "iou_predictions": torch.randn(batch_size, 3).to(device),
                    "low_res_masks": torch.sigmoid(torch.randn(batch_size, 3, 128, 128).to(device))
                }
            else:
                batch_size = x.shape[0]
                return {
                    "masks": torch.sigmoid(torch.randn(batch_size, 3, 512, 512).to(device)),
                    "iou_predictions": torch.randn(batch_size, 3).to(device),
                    "low_res_masks": torch.sigmoid(torch.randn(batch_size, 3, 128, 128).to(device))
                }
    
    class MockEfficientModel(torch.nn.Module):
        """Mock implementation of EfficientMedSAM2 for demonstration"""
        def __init__(self):
            super().__init__()
            # Create some lightweight dummy layers
            self.conv = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
            self.transformer = torch.nn.Sequential(
                torch.nn.Linear(8, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 8)
            )
            # Add memory usage information for demo
            self.memory_usage = 1500  # MB
            
        def forward(self, x):
            # For demo, just return random masks (similar to original but with slightly lower quality)
            if isinstance(x, dict):
                batch_size = 1
                return {
                    "masks": torch.sigmoid(0.9 * torch.randn(batch_size, 3, 512, 512).to(device)),
                    "iou_predictions": 0.9 * torch.randn(batch_size, 3).to(device),
                    "low_res_masks": torch.sigmoid(0.9 * torch.randn(batch_size, 3, 128, 128).to(device))
                }
            else:
                batch_size = x.shape[0]
                return {
                    "masks": torch.sigmoid(0.9 * torch.randn(batch_size, 3, 512, 512).to(device)),
                    "iou_predictions": 0.9 * torch.randn(batch_size, 3).to(device),
                    "low_res_masks": torch.sigmoid(0.9 * torch.randn(batch_size, 3, 128, 128).to(device))
                }
    
    # Create mock models
    st.info("Using mock models for demonstration purposes. In a real scenario, you would load the actual trained models.")
    
    # Create mock predictors
    original_model = MockOriginalModel().to(device)
    efficient_model = MockEfficientModel().to(device)
    
    # Create mock predictors
    class MockPredictor:
        def __init__(self, model):
            self.model = model
            self._is_image_set = False
            self.device = device
        
        def set_image(self, image):
            self._is_image_set = True
            self.image = image
        
        def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=True):
            # Simulate predictions with random masks
            batch_size = 1
            num_masks = 3 if multimask_output else 1
            
            # Generate random masks
            masks = torch.zeros((num_masks, 512, 512))
            
            # For each mask, create a blob centered at the point
            if point_coords is not None:
                for i in range(num_masks):
                    for point in point_coords:
                        x, y = int(point[0]), int(point[1])
                        radius = 50 + i * 20  # Different size for each mask
                        for dx in range(-radius, radius+1):
                            for dy in range(-radius, radius+1):
                                if dx*dx + dy*dy <= radius*radius:
                                    px, py = min(max(0, x + dx), 511), min(max(0, y + dy), 511)
                                    masks[i, py, px] = 1.0
            
            # Add noise
            noise = 0.1 * torch.randn_like(masks)
            masks = torch.clamp(masks + noise, 0, 1)
            
            # Random scores
            scores = torch.rand(num_masks) 
            
            # Random logits
            logits = torch.randn_like(masks)
            
            return masks.cpu().numpy(), scores.cpu().numpy(), logits.cpu().numpy()
    
    # Return mock predictors
    original_predictor = MockPredictor(original_model)
    efficient_predictor = MockPredictor(efficient_model)
    
    return original_predictor, efficient_predictor


def get_slice_from_3d_volume(image_data, axis, slice_idx):
    """
    Get a 2D slice from a 3D volume.
    
    Args:
        image_data: 3D image data
        axis: Axis to slice along (0, 1, or 2)
        slice_idx: Index of the slice
        
    Returns:
        2D slice
    """
    if axis == 0:
        return image_data[slice_idx, :, :]
    elif axis == 1:
        return image_data[:, slice_idx, :]
    else:  # axis == 2
        return image_data[:, :, slice_idx]


def get_image_with_point_overlay(image, point_coords=None):
    """
    Create an image with point overlay.
    
    Args:
        image: Input image
        point_coords: Point coordinates
        
    Returns:
        Image with point overlay (as base64 string)
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap='gray')
    
    if point_coords is not None:
        ax.scatter(point_coords[:, 0], point_coords[:, 1], color='red', s=40, marker='*')
    
    ax.axis('off')
    
    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    # Convert buffer to base64 string
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return img_str


def create_comparison_visualization(image, original_mask, efficient_mask, point_coords=None):
    """
    Create a comparison visualization of the segmentation results.
    
    Args:
        image: Input image
        original_mask: Mask from original MedSAM2
        efficient_mask: Mask from EfficientMedSAM2
        point_coords: Point coordinates
        
    Returns:
        Comparison visualization (as base64 string)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Input image
    axes[0].imshow(image, cmap='gray')
    if point_coords is not None:
        axes[0].scatter(point_coords[:, 0], point_coords[:, 1], color='red', s=40, marker='*')
    axes[0].set_title('Input with Prompt')
    axes[0].axis('off')
    
    # Original MedSAM2 result
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(original_mask, alpha=0.5, cmap='jet')
    if point_coords is not None:
        axes[1].scatter(point_coords[:, 0], point_coords[:, 1], color='red', s=40, marker='*')
    axes[1].set_title('Original MedSAM2')
    axes[1].axis('off')
    
    # EfficientMedSAM2 result
    axes[2].imshow(image, cmap='gray')
    axes[2].imshow(efficient_mask, alpha=0.5, cmap='jet')
    if point_coords is not None:
        axes[2].scatter(point_coords[:, 0], point_coords[:, 1], color='red', s=40, marker='*')
    axes[2].set_title('EfficientMedSAM2')
    axes[2].axis('off')
    
    # Calculate IoU
    intersection = np.logical_and(original_mask, efficient_mask).sum()
    union = np.logical_or(original_mask, efficient_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    
    # Add IoU information
    plt.figtext(0.5, 0.01, f"IoU between masks: {iou:.4f}", ha='center', fontsize=12,
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})
    
    plt.tight_layout()
    
    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    
    # Convert buffer to base64 string
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return img_str, iou


def measure_inference_time(predictor, image, point_coords, point_labels, num_runs=5):
    """
    Measure inference time.
    
    Args:
        predictor: Image predictor
        image: Input image
        point_coords: Point coordinates
        point_labels: Point labels
        num_runs: Number of runs
        
    Returns:
        Average inference time in seconds
    """
    # Set image
    predictor.set_image(image)
    
    # Warm-up
    for _ in range(3):
        _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    
    # Measure time
    start_time = time.time()
    for _ in range(num_runs):
        _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
    
    # Calculate average time
    avg_time = (time.time() - start_time) / num_runs
    
    return avg_time


def measure_memory_usage(model, input_shape):
    """
    Measure memory usage.
    
    Args:
        model: Model
        input_shape: Input shape
        
    Returns:
        Memory usage in MB
    """
    # For our mock models, we have predefined memory usage
    if hasattr(model, 'memory_usage'):
        return model.memory_usage
    
    # For real models, measure using CUDA
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
        # Run a forward pass
        device = next(model.parameters()).device
        x = torch.randn(*input_shape).to(device)
        _ = model(x)
        
        # Get peak memory usage
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return memory_usage
    else:
        return 0.0


def main():
    """Main function for Streamlit interface."""
    st.set_page_config(page_title="EfficientMedSAM2 Demo", page_icon="🧠", layout="wide")
    
    st.title("EfficientMedSAM2: Memory-Efficient Medical Image Segmentation")
    
    # Sidebar configuration
    st.sidebar.title("Model Configuration")
    
    original_model_path = st.sidebar.text_input(
        "Original MedSAM2 Model Path",
        value="./checkpoints/medsam2_model.pth",
        help="Path to the original MedSAM2 model checkpoint"
    )
    
    efficient_model_path = st.sidebar.text_input(
        "EfficientMedSAM2 Model Path",
        value="./checkpoints/efficient_medsam2_model.pth",
        help="Path to the EfficientMedSAM2 model checkpoint"
    )
    
    encoder_type = st.sidebar.selectbox(
        "Encoder Type",
        options=["mobilenet", "efficientnet"],
        index=0,
        help="Type of encoder to use for EfficientMedSAM2"
    )
    
    use_half_precision = st.sidebar.checkbox(
        "Use Half Precision (FP16)",
        value=True,
        help="Use half precision for EfficientMedSAM2"
    )
    
    # Check if model files exist (just for information)
    if not os.path.exists(original_model_path):
        st.sidebar.warning(f"Original model not found at: {original_model_path}")
    
    if not os.path.exists(efficient_model_path):
        st.sidebar.warning(f"Efficient model not found at: {efficient_model_path}")
    
    # Load mock models (regardless of whether the actual model files exist)
    models_ready = False
    try:
        with st.spinner("Loading mock models for demonstration..."):
            original_predictor, efficient_predictor = load_models(
                original_model_path, 
                efficient_model_path,
                encoder_type,
                use_half_precision
            )
        st.sidebar.success("Mock models loaded successfully!")
        models_ready = True
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
    
    # Main content
    st.header("Upload a Medical Image")
    
    uploaded_file = st.file_uploader(
        "Choose a medical image file (NIfTI, JPG, PNG, NPZ)",
        type=["nii", "nii.gz", "jpg", "jpeg", "png", "bmp", "npz"]
    )
    
    if uploaded_file is not None:
        # Load the image
        image_data, file_type = load_image_data(uploaded_file)
        
        if image_data is not None:
            # Display image information
            st.write(f"File type: {file_type}")
            st.write(f"Image shape: {image_data.shape}")
            
            # Handle 3D volumes
            if len(image_data.shape) == 3:
                st.subheader("3D Volume Controls")
                
                axis = st.radio("Slice axis", options=[0, 1, 2], index=2, horizontal=True)
                max_slice = image_data.shape[axis] - 1
                slice_idx = st.slider("Slice index", 0, max_slice, max_slice // 2)
                
                # Get the 2D slice
                current_slice = get_slice_from_3d_volume(image_data, axis, slice_idx)
            else:
                current_slice = image_data
            
            # Display the current slice
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(current_slice, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
            
            # Prompt selection
            st.subheader("Select Prompt Point")
            
            # Get image dimensions
            h, w = current_slice.shape
            
            # Create columns for coordinates
            col1, col2 = st.columns(2)
            with col1:
                point_x = st.slider("X coordinate", 0, w-1, w // 2)
            with col2:
                point_y = st.slider("Y coordinate", 0, h-1, h // 2)
            
            point_coords = np.array([[point_x, point_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)  # 1 for foreground
            
            # Display image with prompt point
            img_with_point = get_image_with_point_overlay(current_slice, point_coords)
            st.image(f"data:image/png;base64,{img_with_point}", caption="Image with prompt point", width="stretch")
            
            # Run segmentation
            if models_ready and st.button("Run Segmentation"):
                with st.spinner("Running segmentation..."):
                    try:
                        # Add batch and channel dimensions for model input
                        model_input = np.expand_dims(np.expand_dims(current_slice, axis=0), axis=0)
                        
                        # Measure inference time for original model
                        original_time_start = time.time()
                        original_predictor.set_image(model_input)
                        original_masks, original_scores, _ = original_predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                        original_time = time.time() - original_time_start
                        
                        # Select best mask
                        original_best_idx = np.argmax(original_scores)
                        original_mask = original_masks[original_best_idx]
                        
                        # Measure inference time for efficient model
                        efficient_time_start = time.time()
                        efficient_predictor.set_image(model_input)
                        efficient_masks, efficient_scores, _ = efficient_predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                        efficient_time = time.time() - efficient_time_start
                        
                        # Select best mask
                        efficient_best_idx = np.argmax(efficient_scores)
                        efficient_mask = efficient_masks[efficient_best_idx]
                        
                        # Create comparison visualization
                        comparison_img, iou = create_comparison_visualization(
                            current_slice,
                            original_mask,
                            efficient_mask,
                            point_coords
                        )
                        
                        # Display results
                        st.subheader("Segmentation Results")
                        st.image(f"data:image/png;base64,{comparison_img}", width="stretch")
                        
                        # Display metrics
                        st.subheader("Performance Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Original Model Time", f"{original_time*1000:.1f} ms")
                        
                        with col2:
                            st.metric("Efficient Model Time", f"{efficient_time*1000:.1f} ms")
                        
                        with col3:
                            speedup = original_time / efficient_time
                            st.metric("Speedup", f"{speedup:.2f}x")
                        
                        # Memory usage (if CUDA is available)
                        if torch.cuda.is_available():
                            device = torch.device('cuda')
                            
                            # Get memory usage (rough estimate)
                            original_model = original_predictor.model
                            efficient_model = efficient_predictor.model
                            
                            # Use small input for memory measurement
                            input_shape = (1, 1, 1, 512, 512)
                            
                            with st.spinner("Measuring memory usage..."):
                                # Original model memory usage
                                torch.cuda.empty_cache()
                                original_memory = measure_memory_usage(original_model, input_shape)
                                
                                # Efficient model memory usage
                                torch.cuda.empty_cache()
                                efficient_memory = measure_memory_usage(efficient_model, input_shape)
                                
                                memory_reduction = (original_memory - efficient_memory) / original_memory * 100
                                
                                # Display memory metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Original Model Memory", f"{original_memory:.1f} MB")
                                
                                with col2:
                                    st.metric("Efficient Model Memory", f"{efficient_memory:.1f} MB")
                                
                                with col3:
                                    st.metric("Memory Reduction", f"{memory_reduction:.1f}%")
                        
                        # Show more detailed comparison
                        with st.expander("Detailed Comparison"):
                            # Create comparison table
                            st.write("### Model Comparison")
                            
                            comparison_data = {
                                "Metric": ["Inference Time", "Memory Usage", "IoU between models"],
                                "Original MedSAM2": [f"{original_time*1000:.2f} ms", 
                                                     f"{original_memory:.2f} MB" if torch.cuda.is_available() else "N/A",
                                                     f"{iou:.4f}"],
                                "EfficientMedSAM2": [f"{efficient_time*1000:.2f} ms", 
                                                     f"{efficient_memory:.2f} MB" if torch.cuda.is_available() else "N/A",
                                                     f"{iou:.4f}"],
                                "Comparison": [f"{speedup:.2f}x faster", 
                                              f"{memory_reduction:.2f}% reduction" if torch.cuda.is_available() else "N/A",
                                              "N/A"]
                            }
                            
                            st.table(comparison_data)
                            
                            # Additional information about the models
                            st.write("### Model Architecture")
                            st.write("""
                            **Original MedSAM2:** Uses a heavy Hiera vision transformer backbone with multiple 
                            memory attention blocks for 3D context.
                            
                            **EfficientMedSAM2:** Uses a lightweight MobileNetV3/EfficientNet backbone with reduced 
                            memory attention layers and optimized for memory efficiency.
                            """)
                    
                    except Exception as e:
                        st.error(f"Error running segmentation: {e}")
    
    # Information about EfficientMedSAM2
    with st.expander("About EfficientMedSAM2"):
        st.write("""
        ### EfficientMedSAM2
        
        EfficientMedSAM2 is a memory-efficient version of MedSAM2, designed for prompt-guided medical image segmentation.
        
        **Key features:**
        - ~10x reduction in memory usage compared to MedSAM2
        - Faster inference time
        - Support for point, box, and mask prompts
        - Maintains segmentation accuracy close to the original model
        - Works on low-memory GPUs (e.g., 6GB VRAM)
        
        **Technical details:**
        - Replaces heavy Hiera backbone with lightweight MobileNetV3/EfficientNet
        - Reduces memory attention layers from 4 to 1
        - Uses half-precision (FP16) computation
        - Implements gradient checkpointing for memory efficiency
        - Supports INT8 quantization for further memory reduction
        """)


if __name__ == "__main__":
    main()
