"""
Benchmarking and Performance Profiling Tools
Compares teacher vs student models for speed, memory usage, and accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: thop not available for FLOP counting. Install with: pip install thop")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    inference_time: float  # seconds
    memory_usage: float    # MB
    peak_memory: float     # MB
    flops: Optional[int] = None  # FLOPs
    params: Optional[int] = None  # Parameters
    accuracy_metrics: Optional[Dict[str, float]] = None


class PerformanceProfiler:
    """
    Profile model performance including speed, memory, and computational complexity.
    """
    
    def __init__(self, device: str = "cuda", warmup_runs: int = 5):
        self.device = device
        self.warmup_runs = warmup_runs
    
    def profile_model(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        model_name: str = "model",
        num_runs: int = 100
    ) -> BenchmarkResult:
        """
        Profile a model's performance.
        
        Args:
            model: Model to profile
            input_data: Input tensor
            model_name: Name for identification
            num_runs: Number of inference runs for timing
            
        Returns:
            BenchmarkResult containing performance metrics
        """
        
        model.eval()
        model.to(self.device)
        input_data = input_data.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(input_data)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        
        # Memory profiling
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():
                output = model(input_data)
                
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        else:
            # CPU memory profiling
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024**2
            
            with torch.no_grad():
                output = model(input_data)
                
            mem_after = process.memory_info().rss / 1024**2
            memory_usage = mem_after - mem_before
            peak_memory = mem_after
        
        # Timing
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                _ = model(input_data)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        
        # FLOPs and parameter counting
        flops, params = None, None
        if THOP_AVAILABLE:
            try:
                flops, params = profile(model, inputs=(input_data,), verbose=False)
            except Exception as e:
                print(f"FLOP counting failed: {e}")
        
        # Count parameters manually if thop failed
        if params is None:
            params = sum(p.numel() for p in model.parameters())
        
        return BenchmarkResult(
            model_name=model_name,
            inference_time=avg_time,
            memory_usage=memory_usage,
            peak_memory=peak_memory,
            flops=flops,
            params=params
        )
    
    def profile_with_prompts(
        self,
        model: nn.Module,
        predictor_class: type,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        model_name: str = "model",
        num_runs: int = 50
    ) -> BenchmarkResult:
        """
        Profile model with prompt-based prediction.
        """
        
        predictor = predictor_class(model)
        predictor.set_image(image)
        
        # Warmup
        for _ in range(self.warmup_runs):
            _, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels
            )
        
        # Memory tracking
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Timing
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels
            )
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        
        # Memory usage
        if self.device == "cuda":
            memory_usage = torch.cuda.memory_allocated() / 1024**2
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        else:
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024**2
            peak_memory = memory_usage
        
        return BenchmarkResult(
            model_name=model_name,
            inference_time=avg_time,
            memory_usage=memory_usage,
            peak_memory=peak_memory,
            params=sum(p.numel() for p in model.parameters())
        )


class ModelBenchmark:
    """
    Comprehensive benchmarking suite for model comparison.
    """
    
    def __init__(self, device: str = "cuda", save_dir: str = "./benchmark_results"):
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.profiler = PerformanceProfiler(device)
        self.results = []
    
    def add_model_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
    
    def benchmark_model_pair(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        input_data: torch.Tensor,
        teacher_name: str = "Teacher",
        student_name: str = "Student"
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Benchmark teacher and student models.
        """
        
        print(f"Benchmarking {teacher_name}...")
        teacher_result = self.profiler.profile_model(
            teacher_model, input_data, teacher_name
        )
        
        print(f"Benchmarking {student_name}...")
        student_result = self.profiler.profile_model(
            student_model, input_data, student_name
        )
        
        self.add_model_result(teacher_result)
        self.add_model_result(student_result)
        
        return teacher_result, student_result
    
    def calculate_improvements(
        self,
        teacher_result: BenchmarkResult,
        student_result: BenchmarkResult
    ) -> Dict[str, float]:
        """
        Calculate improvement metrics.
        """
        
        improvements = {}
        
        # Speed improvement (speedup factor)
        improvements['speedup'] = teacher_result.inference_time / student_result.inference_time
        
        # Memory reduction (percentage)
        improvements['memory_reduction'] = (
            (teacher_result.memory_usage - student_result.memory_usage) / 
            teacher_result.memory_usage * 100
        )
        
        # Parameter reduction
        if teacher_result.params and student_result.params:
            improvements['param_reduction'] = (
                (teacher_result.params - student_result.params) / 
                teacher_result.params * 100
            )
        
        # FLOP reduction
        if teacher_result.flops and student_result.flops:
            improvements['flop_reduction'] = (
                (teacher_result.flops - student_result.flops) / 
                teacher_result.flops * 100
            )
        
        return improvements
    
    def generate_report(
        self,
        teacher_result: BenchmarkResult,
        student_result: BenchmarkResult,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate a detailed comparison report.
        """
        
        improvements = self.calculate_improvements(teacher_result, student_result)
        
        report = f"""
=== Model Performance Comparison Report ===

Teacher Model ({teacher_result.model_name}):
  • Inference Time: {teacher_result.inference_time*1000:.2f} ms
  • Memory Usage: {teacher_result.memory_usage:.1f} MB
  • Peak Memory: {teacher_result.peak_memory:.1f} MB
  • Parameters: {teacher_result.params:,} ({teacher_result.params/1e6:.1f}M)
"""

        if teacher_result.flops:
            flop_str = clever_format([teacher_result.flops], "%.2f")[0] if THOP_AVAILABLE else f"{teacher_result.flops/1e9:.2f}G"
            report += f"  • FLOPs: {flop_str}\n"

        report += f"""
Student Model ({student_result.model_name}):
  • Inference Time: {student_result.inference_time*1000:.2f} ms
  • Memory Usage: {student_result.memory_usage:.1f} MB
  • Peak Memory: {student_result.peak_memory:.1f} MB
  • Parameters: {student_result.params:,} ({student_result.params/1e6:.1f}M)
"""

        if student_result.flops:
            flop_str = clever_format([student_result.flops], "%.2f")[0] if THOP_AVAILABLE else f"{student_result.flops/1e9:.2f}G"
            report += f"  • FLOPs: {flop_str}\n"

        report += f"""
Performance Improvements:
  • Speedup: {improvements['speedup']:.2f}x faster
  • Memory Reduction: {improvements['memory_reduction']:.1f}%
"""

        if 'param_reduction' in improvements:
            report += f"  • Parameter Reduction: {improvements['param_reduction']:.1f}%\n"
        
        if 'flop_reduction' in improvements:
            report += f"  • FLOP Reduction: {improvements['flop_reduction']:.1f}%\n"

        report += f"""
Efficiency Metrics:
  • Throughput Improvement: {improvements['speedup']:.2f}x
  • Memory Efficiency: {teacher_result.memory_usage/student_result.memory_usage:.2f}x more efficient
  • Parameter Efficiency: {teacher_result.params/student_result.params:.2f}x fewer parameters
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report
    
    def plot_comparison(
        self,
        teacher_result: BenchmarkResult,
        student_result: BenchmarkResult,
        save_path: Optional[str] = None
    ):
        """
        Create comparison plots.
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Teacher vs Student Model Comparison', fontsize=16)
        
        models = [teacher_result.model_name, student_result.model_name]
        colors = ['#ff7f0e', '#2ca02c']  # Orange for teacher, green for student
        
        # Inference time comparison
        times = [teacher_result.inference_time * 1000, student_result.inference_time * 1000]
        axes[0, 0].bar(models, times, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel('Inference Time (ms)')
        axes[0, 0].set_title('Inference Speed')
        
        # Add speedup annotation
        speedup = teacher_result.inference_time / student_result.inference_time
        axes[0, 0].text(0.5, max(times) * 0.8, f'{speedup:.1f}x faster', 
                       ha='center', fontsize=12, fontweight='bold')
        
        # Memory usage comparison
        memory = [teacher_result.memory_usage, student_result.memory_usage]
        axes[0, 1].bar(models, memory, color=colors, alpha=0.7)
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Consumption')
        
        # Parameter count comparison
        params = [teacher_result.params / 1e6, student_result.params / 1e6]
        axes[1, 0].bar(models, params, color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Parameters (M)')
        axes[1, 0].set_title('Model Size')
        
        # Efficiency radar chart (if both models have all metrics)
        if all([teacher_result.flops, student_result.flops]):
            # Normalize metrics for radar chart
            metrics = ['Speed', 'Memory', 'Parameters', 'FLOPs']
            teacher_norm = [
                1.0,  # Speed baseline
                1.0,  # Memory baseline  
                1.0,  # Parameters baseline
                1.0   # FLOPs baseline
            ]
            student_norm = [
                speedup,  # Speed improvement
                teacher_result.memory_usage / student_result.memory_usage,  # Memory efficiency
                teacher_result.params / student_result.params,  # Parameter efficiency
                teacher_result.flops / student_result.flops  # FLOP efficiency
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            teacher_norm += teacher_norm[:1]
            student_norm += student_norm[:1]
            
            ax = plt.subplot(2, 2, 4, projection='polar')
            ax.plot(angles, teacher_norm, 'o-', linewidth=2, label=teacher_result.model_name, color=colors[0])
            ax.plot(angles, student_norm, 'o-', linewidth=2, label=student_result.model_name, color=colors[1])
            ax.fill(angles, teacher_norm, alpha=0.25, color=colors[0])
            ax.fill(angles, student_norm, alpha=0.25, color=colors[1])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_title('Efficiency Comparison\n(Higher is Better)')
            ax.legend()
        else:
            # Simple efficiency bar chart
            efficiency_metrics = ['Speed', 'Memory Eff.', 'Param Eff.']
            teacher_eff = [1.0, 1.0, 1.0]
            student_eff = [
                speedup,
                teacher_result.memory_usage / student_result.memory_usage,
                teacher_result.params / student_result.params
            ]
            
            x = np.arange(len(efficiency_metrics))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, teacher_eff, width, label=teacher_result.model_name, 
                          color=colors[0], alpha=0.7)
            axes[1, 1].bar(x + width/2, student_eff, width, label=student_result.model_name, 
                          color=colors[1], alpha=0.7)
            
            axes[1, 1].set_ylabel('Efficiency Factor')
            axes[1, 1].set_title('Efficiency Comparison')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(efficiency_metrics)
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()


def compare_models(
    teacher_model: nn.Module,
    student_model: nn.Module,
    input_data: torch.Tensor,
    teacher_name: str = "MedSAM2",
    student_name: str = "EfficientMedSAM2",
    device: str = "cuda",
    save_dir: str = "./benchmark_results"
) -> Dict[str, Any]:
    """
    Convenience function to compare two models.
    
    Returns:
        Dictionary containing benchmark results and improvements
    """
    
    benchmark = ModelBenchmark(device=device, save_dir=save_dir)
    
    # Run benchmarks
    teacher_result, student_result = benchmark.benchmark_model_pair(
        teacher_model, student_model, input_data, teacher_name, student_name
    )
    
    # Calculate improvements
    improvements = benchmark.calculate_improvements(teacher_result, student_result)
    
    # Generate report
    report_path = Path(save_dir) / "comparison_report.txt"
    report = benchmark.generate_report(teacher_result, student_result, str(report_path))
    
    # Create plots
    plot_path = Path(save_dir) / "comparison_plot.png"
    benchmark.plot_comparison(teacher_result, student_result, str(plot_path))
    
    print(report)
    
    return {
        'teacher_result': teacher_result,
        'student_result': student_result,
        'improvements': improvements,
        'report': report
    }


if __name__ == "__main__":
    # Example usage
    print("Model Benchmarking Tool")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create dummy models for testing
    class DummyTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
            self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
            self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
            self.final = nn.Conv2d(256, 1, 1)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            return self.final(x)
    
    class DummyStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
            self.final = nn.Conv2d(64, 1, 1)
            
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return self.final(x)
    
    # Test models
    teacher = DummyTeacher()
    student = DummyStudent()
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # Run comparison
    results = compare_models(
        teacher_model=teacher,
        student_model=student,
        input_data=dummy_input,
        device=device
    )
    
    print("\nBenchmarking completed!")
    print(f"Speedup: {results['improvements']['speedup']:.2f}x")
    print(f"Memory reduction: {results['improvements']['memory_reduction']:.1f}%")
    print(f"Parameter reduction: {results['improvements']['param_reduction']:.1f}%")