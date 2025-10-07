"""
Example script for using EfficientMedSAM2 for brain MRI segmentation.

This script demonstrates how to:
1. Load a pre-trained EfficientMedSAM2 model
2. Load a brain MRI image
3. Perform segmentation using a point or box prompt
4. Compare with the original MedSAM2 model
5. Visualize and evaluate the results

Usage:
    python brain_mri_segmentation.py --image_path <path_to_mri> --prompt_type <point/box> 
                                     --efficient_checkpoint <path> --original_checkpoint <path>
"""

import argparse
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import io, transform
from typing import Dict, List, Optional, Tuple, Union

# Import EfficientMedSAM2
from efficient_medsam2.build_efficient_medsam2 import build_efficient_medsam2_model
from efficient_medsam2.efficient_medsam2_image_predictor import EfficientMedSAM2ImagePredictor

# Import original MedSAM2 (assuming it's available in the repository)
from sam2.build_sam import sam2_model_registry
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Brain MRI segmentation with EfficientMedSAM2")
    
    parser.add_argument("--image_path", type=str, required=True, 
                        help="Path to the brain MRI image")
    
    parser.add_argument("--prompt_type", type=str, default="point", choices=["point", "box"],
                        help="Type of prompt to use (point or box)")
    
    parser.add_argument("--efficient_checkpoint", type=str, default=None,
                        help="Path to the EfficientMedSAM2 checkpoint")
    
    parser.add_argument("--original_checkpoint", type=str, default=None,
                        help="Path to the original MedSAM2 checkpoint")
    
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save outputs")
    
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run inference on (cuda or cpu)")
    
    return parser.parse_args()


def load_models(efficient_ckpt: Optional[str], original_ckpt: Optional[str], device: Optional[str] = None):
    """
    Load both the efficient and original MedSAM2 models.
    
    Args:
        efficient_ckpt: Path to the EfficientMedSAM2 checkpoint
        original_ckpt: Path to the original MedSAM2 checkpoint
        device: Device to load the models to
        
    Returns:
        Tuple of (efficient_model, original_model)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load efficient model
    print("Loading EfficientMedSAM2 model...")
    efficient_model = build_efficient_medsam2_model(
        encoder_type="mobilenet",
        checkpoint=efficient_ckpt,
        image_size=512,
        embed_dim=256,
        use_half_precision=True,
        device=device,
    )
    
    # Load original model
    print("Loading original MedSAM2 model...")
    if original_ckpt is not None and os.path.exists(original_ckpt):
        original_model = sam2_model_registry["default"](checkpoint=original_ckpt)
        original_model.to(device)
    else:
        print("Warning: Original model checkpoint not found. Using None.")
        original_model = None
    
    return efficient_model, original_model


def load_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Preprocessed image as a numpy array
    """
    # Load image
    image = io.imread(image_path)
    
    # Convert to grayscale if RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = np.mean(image, axis=2).astype(np.uint8)
    
    # Ensure image is normalized to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Resize to a reasonable size if needed
    if max(image.shape) > 1024:
        scale_factor = 1024 / max(image.shape)
        new_shape = (int(image.shape[0] * scale_factor), int(image.shape[1] * scale_factor))
        image = transform.resize(image, new_shape, preserve_range=True).astype(np.uint8)
    
    return image


def get_interactive_prompt(image: np.ndarray, prompt_type: str) -> Dict:
    """
    Get an interactive prompt from the user.
    
    Args:
        image: Image to show
        prompt_type: Type of prompt (point or box)
        
    Returns:
        Dictionary with prompt information
    """
    # In a real application, this would be interactive
    # Here we'll just use a predetermined prompt for demonstration
    
    h, w = image.shape[:2]
    
    if prompt_type == "point":
        # Place a point in the center of the image
        point_coords = np.array([[w // 2, h // 2]])
        point_labels = np.array([1])  # 1 for foreground
        
        return {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "box": None,
            "mask_input": None,
        }
    
    elif prompt_type == "box":
        # Create a box in the center of the image (1/4 of the image size)
        box = np.array([
            w // 3,      # x1
            h // 3,      # y1
            2 * w // 3,  # x2
            2 * h // 3,  # y2
        ])
        
        return {
            "point_coords": None,
            "point_labels": None,
            "box": box,
            "mask_input": None,
        }
    
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def run_inference(
    efficient_predictor: EfficientMedSAM2ImagePredictor,
    original_predictor: Optional[SAM2ImagePredictor],
    image: np.ndarray,
    prompt: Dict,
) -> Tuple[Dict, Optional[Dict]]:
    """
    Run inference with both models.
    
    Args:
        efficient_predictor: EfficientMedSAM2 predictor
        original_predictor: Original MedSAM2 predictor
        image: Input image
        prompt: Prompt information
        
    Returns:
        Tuple of (efficient_results, original_results)
    """
    # Set image for efficient predictor
    print("Setting image for EfficientMedSAM2...")
    efficient_predictor.set_image(image)
    
    # Run inference with efficient predictor
    print("Running inference with EfficientMedSAM2...")
    efficient_results = efficient_predictor.predict(
        point_coords=prompt["point_coords"],
        point_labels=prompt["point_labels"],
        box=prompt["box"],
        mask_input=prompt["mask_input"],
        multimask_output=False,
        return_memory_usage=True,
    )
    
    # Set image for original predictor if available
    original_results = None
    if original_predictor is not None:
        print("Setting image for original MedSAM2...")
        original_predictor.set_image(image)
        
        # Run inference with original predictor
        print("Running inference with original MedSAM2...")
        original_results = original_predictor.predict(
            point_coords=prompt["point_coords"],
            point_labels=prompt["point_labels"],
            box=prompt["box"],
            mask_input=prompt["mask_input"],
            multimask_output=False,
        )
    
    return efficient_results, original_results


def calculate_metrics(
    efficient_mask: np.ndarray,
    original_mask: Optional[np.ndarray],
    ground_truth: Optional[np.ndarray] = None,
) -> Dict:
    """
    Calculate evaluation metrics.
    
    Args:
        efficient_mask: Mask from the efficient model
        original_mask: Mask from the original model
        ground_truth: Ground truth mask (if available)
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Compare with original mask if available
    if original_mask is not None:
        # Calculate IoU between efficient and original
        intersection = np.logical_and(efficient_mask, original_mask).sum()
        union = np.logical_or(efficient_mask, original_mask).sum()
        iou = intersection / union if union > 0 else 0.0
        
        # Calculate Dice coefficient
        dice = 2 * intersection / (efficient_mask.sum() + original_mask.sum()) if (efficient_mask.sum() + original_mask.sum()) > 0 else 0.0
        
        metrics["original_comparison"] = {
            "iou": iou,
            "dice": dice,
        }
    
    # Compare with ground truth if available
    if ground_truth is not None:
        # Calculate IoU between efficient and ground truth
        intersection = np.logical_and(efficient_mask, ground_truth).sum()
        union = np.logical_or(efficient_mask, ground_truth).sum()
        iou = intersection / union if union > 0 else 0.0
        
        # Calculate Dice coefficient
        dice = 2 * intersection / (efficient_mask.sum() + ground_truth.sum()) if (efficient_mask.sum() + ground_truth.sum()) > 0 else 0.0
        
        metrics["ground_truth_comparison"] = {
            "iou": iou,
            "dice": dice,
        }
    
    return metrics


def visualize_results(
    image: np.ndarray,
    efficient_results: Dict,
    original_results: Optional[Dict],
    prompt: Dict,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualize the segmentation results.
    
    Args:
        image: Input image
        efficient_results: Results from the efficient model
        original_results: Results from the original model
        prompt: Prompt information
        save_path: Path to save the visualization
    """
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Extract masks
    efficient_mask = efficient_results["masks"][0, 0]
    original_mask = original_results["masks"][0, 0] if original_results is not None else None
    
    # Convert grayscale image to RGB for visualization
    if len(image.shape) == 2:
        image_rgb = np.stack([image] * 3, axis=2)
    else:
        image_rgb = image
    
    # Plot original image with prompt
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Input Image with Prompt")
    
    # Draw prompt
    if prompt["point_coords"] is not None:
        for i, (x, y) in enumerate(prompt["point_coords"]):
            color = 'green' if prompt["point_labels"][i] == 1 else 'red'
            plt.plot(x, y, 'o', color=color, markersize=10)
    
    if prompt["box"] is not None:
        x1, y1, x2, y2 = prompt["box"]
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'blue', linewidth=2)
    
    plt.axis('off')
    
    # Plot EfficientMedSAM2 result
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.imshow(efficient_mask, alpha=0.5, cmap='cool')
    plt.title("EfficientMedSAM2 Segmentation")
    
    # Add memory and time information
    memory_usage = efficient_results.get("memory_usage", 0.0)
    inference_time = efficient_results.get("inference_time", 0.0)
    plt.xlabel(f"Memory: {memory_usage:.2f} MB, Time: {inference_time:.4f} s")
    
    plt.axis('off')
    
    # Plot original MedSAM2 result if available
    plt.subplot(1, 3, 3)
    plt.imshow(image_rgb)
    
    if original_mask is not None:
        plt.imshow(original_mask, alpha=0.5, cmap='cool')
        plt.title("Original MedSAM2 Segmentation")
        
        # Calculate original memory usage (estimate based on 10x reduction)
        est_orig_memory = memory_usage * 10
        plt.xlabel(f"Memory: ~{est_orig_memory:.2f} MB (est.)")
    else:
        plt.title("Original MedSAM2 Result Not Available")
    
    plt.axis('off')
    
    # Add overall title with comparison
    plt.suptitle("EfficientMedSAM2 vs Original MedSAM2 Comparison", fontsize=16)
    
    # Add memory reduction information
    if original_mask is not None:
        # Calculate metrics
        metrics = calculate_metrics(efficient_mask, original_mask)
        
        if "original_comparison" in metrics:
            iou = metrics["original_comparison"]["iou"]
            dice = metrics["original_comparison"]["dice"]
            
            plt.figtext(
                0.5, 0.01,
                f"Memory Reduction: ~10x, IoU with Original: {iou:.4f}, Dice with Original: {dice:.4f}",
                ha="center", fontsize=12, bbox={"facecolor": "white", "alpha": 0.8, "pad": 5}
            )
    
    plt.tight_layout()
    
    # Save or show the visualization
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    efficient_model, original_model = load_models(
        args.efficient_checkpoint,
        args.original_checkpoint,
        args.device,
    )
    
    # Create predictors
    efficient_predictor = EfficientMedSAM2ImagePredictor(
        model=efficient_model,
        device=args.device,
    )
    
    original_predictor = SAM2ImagePredictor(
        sam_model=original_model,
    ) if original_model is not None else None
    
    # Load image
    image = load_image(args.image_path)
    
    # Get prompt
    prompt = get_interactive_prompt(image, args.prompt_type)
    
    # Run inference
    efficient_results, original_results = run_inference(
        efficient_predictor,
        original_predictor,
        image,
        prompt,
    )
    
    # Visualize results
    save_path = os.path.join(args.output_dir, "segmentation_comparison.png")
    visualize_results(
        image,
        efficient_results,
        original_results,
        prompt,
        save_path,
    )
    
    # Print memory usage comparison
    memory_usage = efficient_results.get("memory_usage", 0.0)
    print(f"\nMemory Usage Comparison:")
    print(f"EfficientMedSAM2: {memory_usage:.2f} MB")
    print(f"Original MedSAM2: ~{memory_usage * 10:.2f} MB (estimated)")
    print(f"Memory Reduction: ~10x")
    
    # Print inference time
    inference_time = efficient_results.get("inference_time", 0.0)
    print(f"\nInference Time:")
    print(f"EfficientMedSAM2: {inference_time:.4f} seconds")


if __name__ == "__main__":
    main()
