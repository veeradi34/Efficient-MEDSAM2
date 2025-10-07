"""
Visualization Tools for Model Comparison
Creates side-by-side visualizations of teacher vs student predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import cv2
from PIL import Image


class MaskVisualizer:
    """
    Visualize and compare model predictions.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        # Setup nice color schemes
        self.setup_colors()
        
    def setup_colors(self):
        """Setup color schemes for visualization."""
        # Custom colormap for masks
        colors = ['black', 'red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
        self.mask_cmap = ListedColormap(colors)
        
        # Overlay colors
        self.teacher_color = '#ff7f0e'  # orange
        self.student_color = '#2ca02c'  # green
        self.agreement_color = '#1f77b4'  # blue
        self.disagreement_color = '#d62728'  # red
    
    def visualize_single_prediction(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        title: str = "Prediction",
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        Visualize a single model prediction.
        
        Args:
            image: Input image [H, W] or [H, W, C]
            mask: Prediction mask [H, W]
            points: Point prompts [[x, y], ...]
            boxes: Box prompts [[x1, y1, x2, y2], ...]
            title: Plot title
            ax: Matplotlib axes (optional)
            
        Returns:
            Matplotlib axes object
        """
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Handle grayscale images
        if len(image.shape) == 2:
            ax.imshow(image, cmap='gray')
        else:
            ax.imshow(image)
        
        # Overlay mask
        mask_colored = np.ma.masked_where(mask < 0.5, mask)
        ax.imshow(mask_colored, alpha=0.6, cmap='Reds')
        
        # Add point prompts
        if points is not None:
            for point in points:
                ax.plot(point[0], point[1], 'r*', markersize=15, markeredgecolor='white', markeredgewidth=2)
        
        # Add box prompts
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                       edgecolor='yellow', facecolor='none')
                ax.add_patch(rect)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        
        return ax
    
    def compare_predictions(
        self,
        image: np.ndarray,
        teacher_mask: np.ndarray,
        student_mask: np.ndarray,
        points: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
        teacher_name: str = "MedSAM2",
        student_name: str = "EfficientMedSAM2",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create side-by-side comparison of teacher and student predictions.
        
        Args:
            image: Input image
            teacher_mask: Teacher model prediction
            student_mask: Student model prediction
            points: Point prompts
            boxes: Box prompts
            teacher_name: Teacher model name
            student_name: Student model name
            save_path: Path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        
        fig, axes = plt.subplots(2, 3, figsize=self.figsize)
        
        # Original image with prompts
        axes[0, 0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        if points is not None:
            for point in points:
                axes[0, 0].plot(point[0], point[1], 'r*', markersize=15, 
                              markeredgecolor='white', markeredgewidth=2)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, 
                                       edgecolor='yellow', facecolor='none')
                axes[0, 0].add_patch(rect)
        axes[0, 0].set_title('Original Image with Prompts', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Teacher prediction
        self.visualize_single_prediction(
            image, teacher_mask, points, boxes, 
            f'{teacher_name} Prediction', axes[0, 1]
        )
        
        # Student prediction
        self.visualize_single_prediction(
            image, student_mask, points, boxes, 
            f'{student_name} Prediction', axes[0, 2]
        )
        
        # Difference analysis
        self._create_difference_analysis(
            image, teacher_mask, student_mask, axes[1, :]
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        
        return fig
    
    def _create_difference_analysis(
        self,
        image: np.ndarray,
        teacher_mask: np.ndarray,
        student_mask: np.ndarray,
        axes: List[plt.Axes]
    ):
        """Create difference analysis plots."""
        
        # Convert masks to binary
        teacher_binary = (teacher_mask > 0.5).astype(np.float32)
        student_binary = (student_mask > 0.5).astype(np.float32)
        
        # Calculate metrics
        intersection = (teacher_binary * student_binary).sum()
        union = ((teacher_binary + student_binary) > 0).sum()
        iou = intersection / (union + 1e-8)
        dice = (2 * intersection) / (teacher_binary.sum() + student_binary.sum() + 1e-8)
        
        # Difference map
        diff_map = np.abs(teacher_binary - student_binary)
        axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[0].imshow(diff_map, alpha=0.7, cmap='hot')
        axes[0].set_title(f'Difference Map\nIoU: {iou:.3f}, Dice: {dice:.3f}', fontweight='bold')
        axes[0].axis('off')
        
        # Overlap visualization
        overlap_viz = np.zeros((*teacher_binary.shape, 3))
        overlap_viz[teacher_binary == 1] = [1, 0.5, 0]  # Orange for teacher only
        overlap_viz[student_binary == 1] = [0, 1, 0.5]  # Green for student only  
        overlap_viz[(teacher_binary == 1) & (student_binary == 1)] = [0, 0, 1]  # Blue for agreement
        
        axes[1].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
        axes[1].imshow(overlap_viz, alpha=0.6)
        axes[1].set_title('Prediction Overlap\n🟠Teacher 🟢Student 🔵Both', fontweight='bold')
        axes[1].axis('off')
        
        # Statistics
        teacher_area = teacher_binary.sum()
        student_area = student_binary.sum()
        agreement_area = intersection
        
        stats_text = f"""Prediction Statistics:
        
Teacher Area: {teacher_area:.0f} pixels
Student Area: {student_area:.0f} pixels
Agreement Area: {agreement_area:.0f} pixels

IoU Score: {iou:.3f}
Dice Score: {dice:.3f}
Area Ratio: {student_area/teacher_area:.3f}

Sensitivity: {intersection/teacher_area:.3f}
Specificity: {intersection/student_area:.3f}
        """
        
        axes[2].text(0.05, 0.95, stats_text, transform=axes[2].transAxes,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[2].axis('off')
    
    def create_batch_comparison(
        self,
        images: List[np.ndarray],
        teacher_masks: List[np.ndarray],
        student_masks: List[np.ndarray],
        titles: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a grid comparison for multiple images.
        """
        
        n_images = len(images)
        fig, axes = plt.subplots(n_images, 4, figsize=(16, 4*n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_images):
            title = titles[i] if titles else f"Sample {i+1}"
            
            # Original image
            axes[i, 0].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
            axes[i, 0].set_title(f'{title}\nOriginal')
            axes[i, 0].axis('off')
            
            # Teacher prediction
            axes[i, 1].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
            teacher_colored = np.ma.masked_where(teacher_masks[i] < 0.5, teacher_masks[i])
            axes[i, 1].imshow(teacher_colored, alpha=0.6, cmap='Oranges')
            axes[i, 1].set_title('Teacher')
            axes[i, 1].axis('off')
            
            # Student prediction  
            axes[i, 2].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
            student_colored = np.ma.masked_where(student_masks[i] < 0.5, student_masks[i])
            axes[i, 2].imshow(student_colored, alpha=0.6, cmap='Greens')
            axes[i, 2].set_title('Student')
            axes[i, 2].axis('off')
            
            # Difference
            diff = np.abs((teacher_masks[i] > 0.5).astype(float) - (student_masks[i] > 0.5).astype(float))
            axes[i, 3].imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
            axes[i, 3].imshow(diff, alpha=0.7, cmap='Reds')
            
            # Calculate IoU for title
            teacher_bin = teacher_masks[i] > 0.5
            student_bin = student_masks[i] > 0.5
            intersection = (teacher_bin & student_bin).sum()
            union = (teacher_bin | student_bin).sum()
            iou = intersection / (union + 1e-8)
            
            axes[i, 3].set_title(f'Difference\nIoU: {iou:.3f}')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Batch comparison saved to {save_path}")
        
        return fig
    
    def create_performance_dashboard(
        self,
        benchmark_results: Dict[str, Any],
        accuracy_scores: Dict[str, float],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a comprehensive performance dashboard.
        """
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Model comparison metrics
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_performance_bars(ax1, benchmark_results)
        
        # Accuracy comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_accuracy_comparison(ax2, accuracy_scores)
        
        # Speed vs Memory scatter
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_speed_memory_scatter(ax3, benchmark_results)
        
        # Model size comparison
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_model_size_comparison(ax4, benchmark_results)
        
        # Efficiency radar chart
        ax5 = fig.add_subplot(gs[1, 2:], projection='polar')
        self._plot_efficiency_radar(ax5, benchmark_results, accuracy_scores)
        
        # Summary statistics
        ax6 = fig.add_subplot(gs[2, :])
        self._create_summary_table(ax6, benchmark_results, accuracy_scores)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance dashboard saved to {save_path}")
        
        return fig
    
    def _plot_performance_bars(self, ax, benchmark_results):
        """Plot performance comparison bars."""
        teacher_result = benchmark_results['teacher_result']
        student_result = benchmark_results['student_result']
        
        metrics = ['Time (ms)', 'Memory (MB)', 'Parameters (M)']
        teacher_vals = [
            teacher_result.inference_time * 1000,
            teacher_result.memory_usage,
            teacher_result.params / 1e6
        ]
        student_vals = [
            student_result.inference_time * 1000,
            student_result.memory_usage,
            student_result.params / 1e6
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, teacher_vals, width, label=teacher_result.model_name, 
               color=self.teacher_color, alpha=0.8)
        ax.bar(x + width/2, student_vals, width, label=student_result.model_name, 
               color=self.student_color, alpha=0.8)
        
        ax.set_ylabel('Value')
        ax.set_title('Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_accuracy_comparison(self, ax, accuracy_scores):
        """Plot accuracy metrics comparison."""
        metrics = list(accuracy_scores.keys())
        values = list(accuracy_scores.values())
        
        bars = ax.bar(metrics, values, color=[self.teacher_color, self.student_color], alpha=0.8)
        ax.set_ylabel('Score')
        ax.set_title('Accuracy Comparison')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_speed_memory_scatter(self, ax, benchmark_results):
        """Plot speed vs memory scatter plot."""
        teacher_result = benchmark_results['teacher_result']
        student_result = benchmark_results['student_result']
        
        ax.scatter(teacher_result.inference_time * 1000, teacher_result.memory_usage,
                  s=100, color=self.teacher_color, label=teacher_result.model_name, alpha=0.8)
        ax.scatter(student_result.inference_time * 1000, student_result.memory_usage,
                  s=100, color=self.student_color, label=student_result.model_name, alpha=0.8)
        
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Speed vs Memory')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_model_size_comparison(self, ax, benchmark_results):
        """Plot model size comparison."""
        teacher_result = benchmark_results['teacher_result']
        student_result = benchmark_results['student_result']
        
        sizes = [teacher_result.params / 1e6, student_result.params / 1e6]
        labels = [teacher_result.model_name, student_result.model_name]
        colors = [self.teacher_color, self.student_color]
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1fM',
                                         startangle=90, alpha=0.8)
        ax.set_title('Model Size Distribution')
    
    def _plot_efficiency_radar(self, ax, benchmark_results, accuracy_scores):
        """Plot efficiency radar chart."""
        teacher_result = benchmark_results['teacher_result']
        student_result = benchmark_results['student_result']
        improvements = benchmark_results['improvements']
        
        # Metrics for radar chart
        categories = ['Speed', 'Memory\nEfficiency', 'Parameter\nEfficiency', 'Accuracy']
        
        # Student values (normalized)
        student_values = [
            improvements['speedup'],
            teacher_result.memory_usage / student_result.memory_usage,
            teacher_result.params / student_result.params,
            accuracy_scores.get('student_dice', 0.8) / accuracy_scores.get('teacher_dice', 0.8)
        ]
        
        # Teacher baseline (all 1.0)
        teacher_values = [1.0] * len(categories)
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add values to complete the circle
        student_values += student_values[:1]
        teacher_values += teacher_values[:1]
        
        # Plot
        ax.plot(angles, teacher_values, 'o-', linewidth=2, label=teacher_result.model_name, 
                color=self.teacher_color)
        ax.fill(angles, teacher_values, alpha=0.25, color=self.teacher_color)
        
        ax.plot(angles, student_values, 'o-', linewidth=2, label=student_result.model_name, 
                color=self.student_color)
        ax.fill(angles, student_values, alpha=0.25, color=self.student_color)
        
        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, max(max(student_values), max(teacher_values)) * 1.1)
        ax.set_title('Efficiency Radar\n(Higher is Better)', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
    
    def _create_summary_table(self, ax, benchmark_results, accuracy_scores):
        """Create summary statistics table."""
        teacher_result = benchmark_results['teacher_result']
        student_result = benchmark_results['student_result']
        improvements = benchmark_results['improvements']
        
        # Create table data
        table_data = [
            ['Metric', teacher_result.model_name, student_result.model_name, 'Improvement'],
            ['Inference Time (ms)', f"{teacher_result.inference_time*1000:.2f}", 
             f"{student_result.inference_time*1000:.2f}", f"{improvements['speedup']:.2f}x faster"],
            ['Memory Usage (MB)', f"{teacher_result.memory_usage:.1f}", 
             f"{student_result.memory_usage:.1f}", f"{improvements['memory_reduction']:.1f}% less"],
            ['Parameters (M)', f"{teacher_result.params/1e6:.1f}", 
             f"{student_result.params/1e6:.1f}", f"{improvements['param_reduction']:.1f}% fewer"],
            ['Dice Score', f"{accuracy_scores.get('teacher_dice', 0.0):.3f}", 
             f"{accuracy_scores.get('student_dice', 0.0):.3f}", 
             f"{(accuracy_scores.get('student_dice', 0.0) - accuracy_scores.get('teacher_dice', 0.0)):.3f}"],
        ]
        
        # Create table
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    if j == 3:  # Improvement column
                        cell.set_facecolor('#d4edda')
                    else:
                        cell.set_facecolor('#f8f9fa')
        
        ax.axis('off')
        ax.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)


def create_comparison_plot(
    image: np.ndarray,
    teacher_mask: np.ndarray,
    student_mask: np.ndarray,
    points: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    teacher_name: str = "MedSAM2",
    student_name: str = "EfficientMedSAM2"
) -> plt.Figure:
    """
    Convenience function to create a comparison plot.
    """
    
    visualizer = MaskVisualizer()
    return visualizer.compare_predictions(
        image=image,
        teacher_mask=teacher_mask,
        student_mask=student_mask,
        points=points,
        teacher_name=teacher_name,
        student_name=student_name,
        save_path=save_path
    )


if __name__ == "__main__":
    # Example usage
    print("Visualization Tools Test")
    
    # Create dummy data
    image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    teacher_mask = np.random.rand(512, 512)
    student_mask = np.random.rand(512, 512)
    points = np.array([[256, 256], [100, 400]])
    
    # Create visualizer
    visualizer = MaskVisualizer()
    
    # Test single prediction visualization
    fig1 = plt.figure(figsize=(8, 8))
    ax = fig1.add_subplot(111)
    visualizer.visualize_single_prediction(image, teacher_mask, points, title="Test Prediction", ax=ax)
    plt.show()
    
    # Test comparison
    print("Creating comparison plot...")
    fig2 = visualizer.compare_predictions(
        image=image,
        teacher_mask=teacher_mask,
        student_mask=student_mask,
        points=points
    )
    plt.show()
    
    # Test batch comparison
    images = [image] * 3
    teacher_masks = [teacher_mask * np.random.rand()] * 3
    student_masks = [student_mask * np.random.rand()] * 3
    
    print("Creating batch comparison...")
    fig3 = visualizer.create_batch_comparison(images, teacher_masks, student_masks)
    plt.show()
    
    print("Visualization tools test completed!")