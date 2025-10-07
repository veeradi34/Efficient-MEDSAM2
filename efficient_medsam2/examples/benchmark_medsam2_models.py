"""
Benchmark EfficientMedSAM2 against Original MedSAM2

This script benchmarks EfficientMedSAM2 against the original MedSAM2 model
on various metrics including memory usage, inference speed, and segmentation quality.

Example usage:
    python benchmark_medsam2_models.py --data_dir /path/to/data 
                                      --original_model /path/to/medsam2_model.pth 
                                      --efficient_model /path/to/efficient_medsam2_model.pth
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import nibabel as nib
from typing import Tuple, List, Dict, Optional, Union
import pandas as pd
import json
from tqdm import tqdm

# Import EfficientMedSAM2
from efficient_medsam2.build_efficient_medsam2 import build_efficient_medsam2_model
from efficient_medsam2.efficient_medsam2_image_predictor import EfficientMedSAM2ImagePredictor

# Import original MedSAM2
from sam2.build_sam import sam2_model_registry
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark EfficientMedSAM2 against Original MedSAM2")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test data (NPZ files)")
    
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                        help="Directory to save benchmark results")
    
    parser.add_argument("--original_model", type=str, required=True,
                        help="Path to original MedSAM2 model checkpoint")
    
    parser.add_argument("--efficient_model", type=str, required=True,
                        help="Path to EfficientMedSAM2 model checkpoint")
    
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to use for benchmarking")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on ('cuda' or 'cpu')")
    
    parser.add_argument("--fp16", action="store_true",
                        help="Use half-precision (FP16) for EfficientMedSAM2")
    
    parser.add_argument("--quantize", action="store_true",
                        help="Apply INT8 quantization to EfficientMedSAM2")
    
    parser.add_argument("--save_visualizations", action="store_true",
                        help="Save visualizations of segmentation results")
    
    parser.add_argument("--encoder_type", type=str, default="mobilenet",
                        choices=["mobilenet", "efficientnet"],
                        help="Type of encoder to use for EfficientMedSAM2")
    
    return parser.parse_args()


def load_npz_data(file_path: str) -> Dict:
    """
    Load a NPZ file containing medical image data.
    
    Args:
        file_path: Path to NPZ file
        
    Returns:
        Dictionary with image and mask data
    """
    data = np.load(file_path, allow_pickle=True)
    
    # Extract data
    image = data['image']
    mask = data['label'] if 'label' in data else None
    
    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image = (image - image.min()) / (image.max() - image.min())
    
    # Ensure image is float32
    image = image.astype(np.float32)
    
    # Get a representative slice if image is 3D
    if len(image.shape) == 3:
        # Find the middle slice with most non-zero values
        if mask is not None:
            # Find the slice with most foreground pixels
            slice_sums = np.sum(mask > 0, axis=(0, 1))
            best_slice = np.argmax(slice_sums)
        else:
            # Use middle slice
            best_slice = image.shape[2] // 2
        
        image_slice = image[:, :, best_slice]
        mask_slice = mask[:, :, best_slice] if mask is not None else None
    else:
        image_slice = image
        mask_slice = mask
    
    return {
        'image': image,
        'mask': mask,
        'image_slice': image_slice,
        'mask_slice': mask_slice,
    }


def generate_prompts(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate point and box prompts from a mask.
    
    Args:
        mask: Binary segmentation mask
        
    Returns:
        Tuple of (point_coords, point_labels, box_coords, box_labels)
    """
    if mask is None:
        # Use center point as default prompt
        h, w = 512, 512  # Default size
        point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        
        # No box prompt
        box_coords = None
        box_labels = None
        
        return point_coords, point_labels, box_coords, box_labels
    
    # Find connected components in mask
    from scipy import ndimage
    labeled_mask, num_features = ndimage.label(mask)
    
    # If no regions found, use center point
    if num_features == 0:
        h, w = mask.shape
        point_coords = np.array([[w // 2, h // 2]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        
        # No box prompt
        box_coords = None
        box_labels = None
    else:
        # Choose the largest component
        sizes = [(labeled_mask == i).sum() for i in range(1, num_features + 1)]
        largest_component_idx = np.argmax(sizes) + 1
        largest_component = (labeled_mask == largest_component_idx)
        
        # Find centroid of largest component
        y_indices, x_indices = np.where(largest_component)
        centroid_y = int(np.mean(y_indices))
        centroid_x = int(np.mean(x_indices))
        
        point_coords = np.array([[centroid_x, centroid_y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        
        # Create bounding box
        min_y, min_x = np.min(y_indices), np.min(x_indices)
        max_y, max_x = np.max(y_indices), np.max(x_indices)
        
        # Add margin
        margin = 5
        min_y = max(0, min_y - margin)
        min_x = max(0, min_x - margin)
        max_y = min(mask.shape[0] - 1, max_y + margin)
        max_x = min(mask.shape[1] - 1, max_x + margin)
        
        box_coords = np.array([[min_x, min_y, max_x, max_y]], dtype=np.float32)
        box_labels = np.array([2, 3])  # 2 for box start, 3 for box end
    
    return point_coords, point_labels, box_coords, box_labels


def measure_memory_usage(model: torch.nn.Module, input_shape: Tuple[int, ...], device: torch.device) -> float:
    """
    Measure peak GPU memory usage of a model.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch, time, channels, height, width)
        device: Device to measure memory on
        
    Returns:
        Peak memory usage in MB
    """
    if device.type != "cuda":
        return 0.0
    
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Get current memory usage
    start_memory = torch.cuda.memory_allocated(device) / 1024**2
    
    # Run a small forward pass to measure memory
    x = torch.randn(*input_shape).to(device)
    model(x)
    
    # Get peak memory usage
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2
    
    # Return difference (peak usage caused by the model)
    return peak_memory - start_memory


def measure_inference_time(
    predictor,
    image: np.ndarray,
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    box_coords: Optional[np.ndarray] = None,
    num_iterations: int = 10,
    warmup: int = 3,
) -> float:
    """
    Measure inference time of a predictor.
    
    Args:
        predictor: Image predictor
        image: Input image
        point_coords: Point coordinates
        point_labels: Point labels
        box_coords: Box coordinates
        num_iterations: Number of iterations to measure
        warmup: Number of warmup iterations
        
    Returns:
        Average inference time in seconds
    """
    # Set image
    predictor.set_image(image)
    
    # Warmup
    for _ in range(warmup):
        _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords,
            multimask_output=True,
        )
    
    # Measure time
    start_time = time.time()
    for _ in range(num_iterations):
        _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords,
            multimask_output=True,
        )
    
    # Calculate average time
    avg_time = (time.time() - start_time) / num_iterations
    
    return avg_time


def calculate_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculate segmentation metrics.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Dictionary with metrics
    """
    # Convert masks to binary
    pred_mask_bin = pred_mask > 0.5
    gt_mask_bin = gt_mask > 0.5
    
    # Intersection and union
    intersection = np.logical_and(pred_mask_bin, gt_mask_bin).sum()
    union = np.logical_or(pred_mask_bin, gt_mask_bin).sum()
    
    # Metrics
    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (pred_mask_bin.sum() + gt_mask_bin.sum()) if (pred_mask_bin.sum() + gt_mask_bin.sum()) > 0 else 0.0
    
    # Precision and recall
    true_positive = intersection
    false_positive = pred_mask_bin.sum() - true_positive
    false_negative = gt_mask_bin.sum() - true_positive
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'iou': float(iou),
        'dice': float(dice),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def visualize_results(
    image: np.ndarray,
    original_mask: np.ndarray,
    efficient_mask: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    point_coords: Optional[np.ndarray] = None,
    box_coords: Optional[np.ndarray] = None,
    metrics: Optional[Dict] = None,
    save_path: Optional[str] = None,
):
    """
    Visualize segmentation results.
    
    Args:
        image: Input image
        original_mask: Mask from original MedSAM2
        efficient_mask: Mask from EfficientMedSAM2
        gt_mask: Ground truth mask (if available)
        point_coords: Point coordinates used as prompts
        box_coords: Box coordinates used as prompts
        metrics: Dictionary with metrics to display
        save_path: Path to save visualization
    """
    # Determine number of columns (3 or 4 depending on ground truth availability)
    n_cols = 4 if gt_mask is not None else 3
    
    plt.figure(figsize=(n_cols * 5, 5))
    
    # Plot original image
    plt.subplot(1, n_cols, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")
    
    # Plot original MedSAM2 result
    plt.subplot(1, n_cols, 2)
    plt.imshow(image, cmap="gray")
    plt.imshow(original_mask, alpha=0.5, cmap="jet")
    if point_coords is not None:
        plt.scatter(point_coords[:, 0], point_coords[:, 1], c="red", s=40, marker="*")
    if box_coords is not None:
        for box in box_coords:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              fill=False, edgecolor="red", linewidth=2))
    plt.title("Original MedSAM2")
    plt.axis("off")
    
    # Plot EfficientMedSAM2 result
    plt.subplot(1, n_cols, 3)
    plt.imshow(image, cmap="gray")
    plt.imshow(efficient_mask, alpha=0.5, cmap="jet")
    if point_coords is not None:
        plt.scatter(point_coords[:, 0], point_coords[:, 1], c="red", s=40, marker="*")
    if box_coords is not None:
        for box in box_coords:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              fill=False, edgecolor="red", linewidth=2))
    plt.title("EfficientMedSAM2")
    plt.axis("off")
    
    # Plot ground truth if available
    if gt_mask is not None:
        plt.subplot(1, n_cols, 4)
        plt.imshow(image, cmap="gray")
        plt.imshow(gt_mask, alpha=0.5, cmap="jet")
        plt.title("Ground Truth")
        plt.axis("off")
    
    # Add metrics if available
    if metrics:
        info_text = "Metrics:\n"
        if "memory_original" in metrics and "memory_efficient" in metrics:
            memory_reduction = (metrics["memory_original"] - metrics["memory_efficient"]) / metrics["memory_original"] * 100
            info_text += f"Memory: {metrics['memory_original']:.1f}MB vs {metrics['memory_efficient']:.1f}MB "
            info_text += f"({memory_reduction:.1f}% reduction)\n"
        
        if "time_original" in metrics and "time_efficient" in metrics:
            speedup = metrics["time_original"] / metrics["time_efficient"]
            info_text += f"Time: {metrics['time_original']*1000:.1f}ms vs {metrics['time_efficient']*1000:.1f}ms "
            info_text += f"({speedup:.1f}x speedup)\n"
        
        if "original_vs_gt" in metrics and gt_mask is not None:
            info_text += f"Original vs GT: Dice={metrics['original_vs_gt']['dice']:.4f}, IoU={metrics['original_vs_gt']['iou']:.4f}\n"
        
        if "efficient_vs_gt" in metrics and gt_mask is not None:
            info_text += f"Efficient vs GT: Dice={metrics['efficient_vs_gt']['dice']:.4f}, IoU={metrics['efficient_vs_gt']['iou']:.4f}\n"
        
        if "original_vs_efficient" in metrics:
            info_text += f"Original vs Efficient: Dice={metrics['original_vs_efficient']['dice']:.4f}, IoU={metrics['original_vs_efficient']['iou']:.4f}"
        
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=10, 
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8})
    
    # Save or show
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def benchmark_models(args: argparse.Namespace) -> Dict:
    """
    Benchmark original MedSAM2 against EfficientMedSAM2.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary with benchmark results
    """
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load original MedSAM2 model
    print(f"Loading original MedSAM2 model from {args.original_model}...")
    original_model = sam2_model_registry["default"](checkpoint=args.original_model)
    original_model.to(device)
    original_predictor = SAM2ImagePredictor(original_model)
    
    # Load EfficientMedSAM2 model
    print(f"Loading EfficientMedSAM2 model ({args.encoder_type}) from {args.efficient_model}...")
    efficient_model = build_efficient_medsam2_model(
        encoder_type=args.encoder_type,
        checkpoint=args.efficient_model,
        device=device,
        use_half_precision=args.fp16,
    )
    
    # Apply quantization if requested
    if args.quantize:
        try:
            print("Applying INT8 quantization to EfficientMedSAM2...")
            # Note: Full quantization would require proper calibration
            # This is a simplified example
            from torch.ao.quantization import quantize_dynamic
            efficient_model = quantize_dynamic(
                efficient_model, {torch.nn.Linear}, dtype=torch.qint8
            )
        except Exception as e:
            print(f"Warning: Failed to apply quantization: {e}")
    
    efficient_predictor = EfficientMedSAM2ImagePredictor(efficient_model)
    
    # List data files
    if os.path.isdir(args.data_dir):
        data_files = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')]
        data_files = [os.path.join(args.data_dir, f) for f in data_files]
    else:
        # Single file
        data_files = [args.data_dir]
    
    # Limit number of files if requested
    if args.num_samples < len(data_files):
        data_files = data_files[:args.num_samples]
    
    print(f"Found {len(data_files)} data files")
    
    # Measure memory usage
    print("Measuring memory usage...")
    memory_original = measure_memory_usage(original_model, (1, 1, 1, 512, 512), device)
    memory_efficient = measure_memory_usage(efficient_model, (1, 1, 1, 512, 512), device)
    
    memory_reduction = (memory_original - memory_efficient) / memory_original * 100
    print(f"Original MedSAM2: {memory_original:.2f} MB")
    print(f"EfficientMedSAM2: {memory_efficient:.2f} MB")
    print(f"Memory reduction: {memory_reduction:.2f}%")
    
    # Initialize results
    results = {
        'memory': {
            'original': float(memory_original),
            'efficient': float(memory_efficient),
            'reduction_percent': float(memory_reduction),
        },
        'samples': [],
    }
    
    # Process each data file
    for i, data_file in enumerate(tqdm(data_files, desc="Processing samples")):
        try:
            # Load data
            data = load_npz_data(data_file)
            image = data['image_slice']
            gt_mask = data['mask_slice']
            
            # Generate prompts
            point_coords, point_labels, box_coords, box_labels = generate_prompts(gt_mask)
            
            # Prepare model input (add batch and channel dimensions)
            model_input = np.expand_dims(np.expand_dims(image, axis=0), axis=0)
            
            # Measure inference time for original model
            time_original = measure_inference_time(
                original_predictor,
                model_input,
                point_coords,
                point_labels,
                box_coords,
            )
            
            # Measure inference time for efficient model
            time_efficient = measure_inference_time(
                efficient_predictor,
                model_input,
                point_coords,
                point_labels,
                box_coords,
            )
            
            speedup = time_original / time_efficient
            print(f"Sample {i+1}/{len(data_files)}: "
                  f"Original: {time_original*1000:.2f}ms, "
                  f"Efficient: {time_efficient*1000:.2f}ms, "
                  f"Speedup: {speedup:.2f}x")
            
            # Run inference for visualization
            original_predictor.set_image(model_input)
            original_masks, original_scores, _ = original_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_coords,
                multimask_output=True,
            )
            
            efficient_predictor.set_image(model_input)
            efficient_masks, efficient_scores, _ = efficient_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_coords,
                multimask_output=True,
            )
            
            # Select best mask for both models
            original_best_idx = np.argmax(original_scores)
            efficient_best_idx = np.argmax(efficient_scores)
            
            original_mask = original_masks[original_best_idx]
            efficient_mask = efficient_masks[efficient_best_idx]
            
            # Calculate metrics
            metrics = {}
            
            # Original vs Ground Truth
            if gt_mask is not None:
                metrics['original_vs_gt'] = calculate_metrics(original_mask, gt_mask)
            
            # Efficient vs Ground Truth
            if gt_mask is not None:
                metrics['efficient_vs_gt'] = calculate_metrics(efficient_mask, gt_mask)
            
            # Original vs Efficient
            metrics['original_vs_efficient'] = calculate_metrics(original_mask, efficient_mask)
            
            # Add timing information
            metrics['time_original'] = float(time_original)
            metrics['time_efficient'] = float(time_efficient)
            metrics['speedup'] = float(speedup)
            
            # Add memory information
            metrics['memory_original'] = float(memory_original)
            metrics['memory_efficient'] = float(memory_efficient)
            
            # Save sample results
            sample_result = {
                'file': os.path.basename(data_file),
                'metrics': metrics,
            }
            
            results['samples'].append(sample_result)
            
            # Visualize if requested
            if args.save_visualizations:
                save_path = os.path.join(args.output_dir, f"viz_{i+1:03d}.png")
                visualize_results(
                    image=image,
                    original_mask=original_mask,
                    efficient_mask=efficient_mask,
                    gt_mask=gt_mask,
                    point_coords=point_coords,
                    box_coords=box_coords,
                    metrics=metrics,
                    save_path=save_path,
                )
        
        except Exception as e:
            print(f"Error processing {data_file}: {e}")
            continue
    
    # Calculate overall metrics
    if results['samples']:
        # Calculate averages
        avg_metrics = {
            'time_original': np.mean([s['metrics']['time_original'] for s in results['samples']]),
            'time_efficient': np.mean([s['metrics']['time_efficient'] for s in results['samples']]),
            'speedup': np.mean([s['metrics']['speedup'] for s in results['samples']]),
        }
        
        # Calculate averages for segmentation metrics if available
        if all('original_vs_gt' in s['metrics'] for s in results['samples']):
            avg_metrics['original_vs_gt'] = {
                'iou': np.mean([s['metrics']['original_vs_gt']['iou'] for s in results['samples']]),
                'dice': np.mean([s['metrics']['original_vs_gt']['dice'] for s in results['samples']]),
            }
        
        if all('efficient_vs_gt' in s['metrics'] for s in results['samples']):
            avg_metrics['efficient_vs_gt'] = {
                'iou': np.mean([s['metrics']['efficient_vs_gt']['iou'] for s in results['samples']]),
                'dice': np.mean([s['metrics']['efficient_vs_gt']['dice'] for s in results['samples']]),
            }
        
        if all('original_vs_efficient' in s['metrics'] for s in results['samples']):
            avg_metrics['original_vs_efficient'] = {
                'iou': np.mean([s['metrics']['original_vs_efficient']['iou'] for s in results['samples']]),
                'dice': np.mean([s['metrics']['original_vs_efficient']['dice'] for s in results['samples']]),
            }
        
        results['average_metrics'] = avg_metrics
        
        # Generate summary table
        summary_table = pd.DataFrame({
            'Metric': [
                'Memory Usage (MB)',
                'Memory Reduction (%)',
                'Inference Time (ms)',
                'Speedup Factor (x)',
                'IoU vs Ground Truth',
                'Dice vs Ground Truth',
                'IoU between Models',
                'Dice between Models',
            ],
            'Original MedSAM2': [
                f"{memory_original:.2f}",
                'N/A',
                f"{avg_metrics['time_original']*1000:.2f}",
                'N/A',
                f"{avg_metrics.get('original_vs_gt', {}).get('iou', 'N/A'):.4f}" 
                    if 'original_vs_gt' in avg_metrics else 'N/A',
                f"{avg_metrics.get('original_vs_gt', {}).get('dice', 'N/A'):.4f}"
                    if 'original_vs_gt' in avg_metrics else 'N/A',
                f"{avg_metrics.get('original_vs_efficient', {}).get('iou', 'N/A'):.4f}"
                    if 'original_vs_efficient' in avg_metrics else 'N/A',
                f"{avg_metrics.get('original_vs_efficient', {}).get('dice', 'N/A'):.4f}"
                    if 'original_vs_efficient' in avg_metrics else 'N/A',
            ],
            'EfficientMedSAM2': [
                f"{memory_efficient:.2f}",
                f"{memory_reduction:.2f}",
                f"{avg_metrics['time_efficient']*1000:.2f}",
                f"{avg_metrics['speedup']:.2f}",
                f"{avg_metrics.get('efficient_vs_gt', {}).get('iou', 'N/A'):.4f}"
                    if 'efficient_vs_gt' in avg_metrics else 'N/A',
                f"{avg_metrics.get('efficient_vs_gt', {}).get('dice', 'N/A'):.4f}"
                    if 'efficient_vs_gt' in avg_metrics else 'N/A',
                f"{avg_metrics.get('original_vs_efficient', {}).get('iou', 'N/A'):.4f}"
                    if 'original_vs_efficient' in avg_metrics else 'N/A',
                f"{avg_metrics.get('original_vs_efficient', {}).get('dice', 'N/A'):.4f}"
                    if 'original_vs_efficient' in avg_metrics else 'N/A',
            ]
        })
        
        # Print summary table
        print("\nSummary of Benchmark Results:")
        print(summary_table.to_string(index=False))
        
        # Save summary table
        summary_table.to_csv(os.path.join(args.output_dir, "summary_metrics.csv"), index=False)
    
    # Save detailed results
    with open(os.path.join(args.output_dir, "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create and save summary plot
    if results['samples']:
        plt.figure(figsize=(12, 10))
        
        # Memory comparison
        plt.subplot(2, 2, 1)
        plt.bar(['Original MedSAM2', 'EfficientMedSAM2'], 
                [memory_original, memory_efficient])
        plt.title('Memory Usage (MB)')
        plt.ylabel('Memory (MB)')
        
        # Time comparison
        plt.subplot(2, 2, 2)
        plt.bar(['Original MedSAM2', 'EfficientMedSAM2'], 
                [avg_metrics['time_original']*1000, avg_metrics['time_efficient']*1000])
        plt.title('Inference Time (ms)')
        plt.ylabel('Time (ms)')
        
        # IoU comparison if available
        plt.subplot(2, 2, 3)
        if 'original_vs_gt' in avg_metrics and 'efficient_vs_gt' in avg_metrics:
            plt.bar(['Original MedSAM2', 'EfficientMedSAM2'], 
                    [avg_metrics['original_vs_gt']['iou'], avg_metrics['efficient_vs_gt']['iou']])
            plt.title('IoU vs Ground Truth')
            plt.ylabel('IoU')
        else:
            plt.text(0.5, 0.5, 'No Ground Truth Available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('IoU vs Ground Truth')
        
        # Dice comparison if available
        plt.subplot(2, 2, 4)
        if 'original_vs_gt' in avg_metrics and 'efficient_vs_gt' in avg_metrics:
            plt.bar(['Original MedSAM2', 'EfficientMedSAM2'], 
                    [avg_metrics['original_vs_gt']['dice'], avg_metrics['efficient_vs_gt']['dice']])
            plt.title('Dice vs Ground Truth')
            plt.ylabel('Dice')
        else:
            plt.text(0.5, 0.5, 'No Ground Truth Available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.title('Dice vs Ground Truth')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "benchmark_summary.png"), dpi=150)
    
    print(f"Benchmark results saved to {args.output_dir}")
    
    return results


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Run benchmark
    results = benchmark_models(args)
    
    print("Benchmarking completed!")


if __name__ == "__main__":
    main()
