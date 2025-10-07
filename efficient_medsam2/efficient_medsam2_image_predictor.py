"""
EfficientMedSAM2 Image Predictor

This module implements a lightweight predictor for the EfficientMedSAM2 model,
which allows for efficient segmentation of medical images using prompts.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Union, Dict
import time

from .efficient_medsam2_base import EfficientMedSAM2
from PIL.Image import Image


class EfficientMedSAM2ImagePredictor:
    """
    Uses EfficientMedSAM2 to calculate the image embedding for a medical image,
    and then allow repeated, efficient mask prediction given prompts.
    """
    def __init__(
        self,
        model: EfficientMedSAM2,
        mask_threshold: float = 0.0,
        max_hole_area: float = 0.0,
        max_sprinkle_area: float = 0.0,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the efficient image predictor.
        
        Args:
            model: EfficientMedSAM2 model
            mask_threshold: Threshold for binary masks (default: 0.0)
            max_hole_area: Maximum area of holes to fill (default: 0.0)
            max_sprinkle_area: Maximum area of sprinkles to remove (default: 0.0)
            device: Device to run the model on (default: None, auto-detect)
        """
        super().__init__()
        
        # Set model and device
        self.model = model
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Set mask post-processing parameters
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        
        # Set to eval mode by default
        self.model.eval()
        
        # Track memory usage
        self.memory_usage = None
        
        # Initialize image embeddings
        self.reset_image()
    
    def reset_image(self) -> None:
        """Reset the image embeddings."""
        self.features = None
        self.original_size = None
        self.input_size = None
    
    def set_image(
        self,
        image: Union[np.ndarray, torch.Tensor, Image],
        preprocessing: bool = True,
    ) -> None:
        """
        Set the image for which masks will be predicted.
        
        Args:
            image: Input image (numpy array, torch tensor, or PIL image)
            preprocessing: Whether to apply preprocessing (default: True)
        """
        # Reset previous image
        self.reset_image()
        
        # Convert PIL image to numpy array
        if isinstance(image, Image):
            image = np.array(image)
        
        # Save original image size
        if isinstance(image, np.ndarray):
            self.original_size = image.shape[:2]
            
            # Preprocess image if needed
            if preprocessing:
                # Normalize image to [0, 1] range
                if image.dtype == np.uint8:
                    image = image.astype(np.float32) / 255.0
                
                # Add channel dimension if grayscale
                if len(image.shape) == 2:
                    image = image[:, :, None]
                
                # Convert to torch tensor
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                # Just convert to tensor
                if len(image.shape) == 2:
                    image = image[None, :, :]
                else:
                    image = image.transpose(2, 0, 1)
                
                image = torch.from_numpy(image).float()
        
        # Track input size
        self.input_size = tuple(image.shape[-2:])
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # Add time dimension for the model
        if len(image.shape) == 4:
            image = image.unsqueeze(1)  # (B, T, C, H, W)
        
        # Move to device
        image = image.to(self.device)
        
        # Track memory usage before
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        # Get image embeddings
        with torch.no_grad():
            self.features = self.model._encode_images({"images": image})
        
        # Track memory usage after
        if torch.cuda.is_available():
            mem_after = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            self.memory_usage = mem_after - mem_before
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_memory_usage: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Predict segmentation masks for the current image.
        
        Args:
            point_coords: Point coordinates of shape (N, 2)
            point_labels: Point labels of shape (N,), where 1 is foreground and 0 is background
            box: Box coordinates of shape (4,) in format [x1, y1, x2, y2]
            mask_input: Input mask of shape (H, W)
            multimask_output: Whether to return multiple masks
            return_memory_usage: Whether to return memory usage information
            
        Returns:
            Dictionary containing:
                - "masks": Predicted masks of shape (1, num_masks, H, W)
                - "scores": Confidence scores of shape (1, num_masks)
                - "memory_usage": Memory usage in MB (if return_memory_usage=True)
        """
        # Ensure we have image features
        if self.features is None:
            raise ValueError("Image features not found. Please call set_image() first.")
        
        # Prepare point prompts
        points = None
        if point_coords is not None:
            if point_labels is None:
                point_labels = np.ones(point_coords.shape[0])
            
            # Convert to torch tensors
            point_coords = torch.from_numpy(point_coords).float().to(self.device)
            point_labels = torch.from_numpy(point_labels).float().to(self.device)
            
            # Reshape and add batch dimension
            point_coords = point_coords.reshape(1, -1, 2)
            point_labels = point_labels.reshape(1, -1)
            
            # Normalize coordinates
            point_coords = self._normalize_coordinates(point_coords)
            
            points = (point_coords, point_labels)
        
        # Prepare box prompt
        boxes = None
        if box is not None:
            # Convert to torch tensor
            boxes = torch.from_numpy(box).float().to(self.device)
            
            # Add batch dimension
            boxes = boxes.reshape(1, 4)
            
            # Normalize coordinates
            boxes = self._normalize_coordinates(boxes, is_box=True)
        
        # Prepare mask prompt
        masks = None
        if mask_input is not None:
            # Convert to torch tensor
            masks = torch.from_numpy(mask_input).float().to(self.device)
            
            # Add batch and channel dimensions
            masks = masks.reshape(1, 1, *masks.shape)
        
        # Track memory usage before
        torch.cuda.empty_cache()
        if torch.cuda.is_available() and return_memory_usage:
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        # Time the prediction
        start_time = time.time()
        
        # Forward pass
        with torch.no_grad():
            # Prepare input dictionary
            batched_input = {
                "images": self.features.shape[0] * [None],  # Dummy value, features already computed
                "points": points,
                "boxes": boxes,
                "masks": masks,
            }
            
            # Run model
            outputs = self.model(
                batched_input=batched_input,
                multimask_output=multimask_output,
            )
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Track memory usage after
        if torch.cuda.is_available() and return_memory_usage:
            mem_after = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            memory_usage = mem_after - mem_before
        else:
            memory_usage = self.memory_usage if self.memory_usage is not None else 0.0
        
        # Get masks and scores
        masks = outputs["masks"]
        iou_scores = outputs["iou_scores"]
        
        # Convert to numpy arrays
        masks = masks.cpu().numpy()
        iou_scores = iou_scores.cpu().numpy()
        
        # Apply thresholding
        masks = masks > self.mask_threshold
        
        # Post-process masks (fill holes, remove small regions)
        if self.max_hole_area > 0 or self.max_sprinkle_area > 0:
            masks = self._post_process_masks(masks)
        
        # Return results
        results = {
            "masks": masks,
            "scores": iou_scores,
            "inference_time": inference_time,
        }
        
        if return_memory_usage:
            results["memory_usage"] = memory_usage
        
        return results
    
    def _normalize_coordinates(
        self,
        coords: torch.Tensor,
        is_box: bool = False,
    ) -> torch.Tensor:
        """
        Normalize coordinates to the model's input size.
        
        Args:
            coords: Coordinates tensor
            is_box: Whether the coordinates are for a box
            
        Returns:
            Normalized coordinates
        """
        if is_box:
            # For box: [x1, y1, x2, y2]
            h, w = self.input_size
            coords = coords.clone()
            coords[:, 0] = coords[:, 0] / w
            coords[:, 1] = coords[:, 1] / h
            coords[:, 2] = coords[:, 2] / w
            coords[:, 3] = coords[:, 3] / h
        else:
            # For points: [x, y]
            h, w = self.input_size
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h
        
        return coords
    
    def _post_process_masks(self, masks: np.ndarray) -> np.ndarray:
        """
        Post-process masks by filling small holes and removing small regions.
        
        Args:
            masks: Binary masks of shape (B, N, H, W)
            
        Returns:
            Post-processed masks of the same shape
        """
        from skimage import morphology
        
        processed_masks = np.zeros_like(masks)
        
        # Process each mask
        for b in range(masks.shape[0]):
            for n in range(masks.shape[1]):
                mask = masks[b, n].astype(np.uint8)
                
                # Fill small holes
                if self.max_hole_area > 0:
                    mask = morphology.remove_small_holes(
                        mask.astype(bool),
                        area_threshold=self.max_hole_area,
                    ).astype(np.uint8)
                
                # Remove small regions
                if self.max_sprinkle_area > 0:
                    mask = morphology.remove_small_objects(
                        mask.astype(bool),
                        min_size=self.max_sprinkle_area,
                    ).astype(np.uint8)
                
                processed_masks[b, n] = mask
        
        return processed_masks
    
    def get_memory_usage(self) -> float:
        """Get the memory usage in MB."""
        return self.memory_usage
