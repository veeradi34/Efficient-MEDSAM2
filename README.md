# EfficientMedSAM2: Knowledge Distillation Framework

This repository implements a comprehensive knowledge distillation framework to create **EfficientMedSAM2**, a lightweight and efficient variant of MedSAM2 for medical image segmentation in resource-constrained environments.

## 🎯 Overview

EfficientMedSAM2 is designed through a **3-stage decoupled knowledge distillation** approach that progressively transfers knowledge from the full MedSAM2 model (teacher) to a compact student model while maintaining high segmentation accuracy.

### Key Features

- **3-Stage Distillation**: Decoupled approach for systematic knowledge transfer
- **ROI-Aware Memory Attention**: Efficient spatial masking for reduced computation
- **Multiple Backbone Options**: MobileNet, EfficientNet, or ViT-Tiny
- **Comprehensive Benchmarking**: Performance, memory, and accuracy comparisons
- **Visualization Tools**: Side-by-side prediction comparisons

## 🏗️ Architecture

### Teacher Model (MedSAM2)
- **Image Encoder**: Hierarchical ViT (Hiera) producing multi-scale features (1/4 & 1/16)
- **Prompt Encoder**: Processes points, boxes, and optional masks
- **Memory Attention**: 3D context from recent slices for consistency
- **Mask Decoder**: Lightweight decoder with upsampling
- **Loss**: Weighted focal + Dice (≈20:1 ratio)

### Student Model (EfficientMedSAM2)
- **Efficient Backbone**: MobileNet/EfficientNet/ViT-Tiny
- **Slim Memory Attention**: 1-2 layers, fewer heads, ROI masking
- **Lightweight Decoder**: Reduced transformer layers
- **Temporal Truncation**: Limited to last 2-4 slices

## 📚 Distillation Strategy

### Stage 1: Pre-Memory Feature Distillation
- **Objective**: Match teacher backbone features at 1/4 & 1/16 scales
- **Loss**: Feature MSE + Cosine similarity
- **Frozen**: Memory attention + Mask decoder
- **Duration**: 10 epochs with lower learning rate

### Stage 2: Memory-Aware Distillation  
- **Objective**: Match post-memory features + logit distillation
- **Loss**: Post-memory MSE + KL divergence + Soft Dice
- **Unfrozen**: Memory attention
- **Duration**: 15 epochs with full learning rate

### Stage 3: Fine-Tuning with Ground Truth
- **Objective**: Ground truth supervision + small KD term
- **Loss**: Dice + BCE + 0.1 × KL divergence
- **Unfrozen**: All components (optionally decoder)
- **Duration**: 5 epochs with lower learning rate

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Distillation

# Install dependencies
pip install torch torchvision
pip install timm matplotlib seaborn tqdm
pip install thop  # For FLOP counting (optional)

# Install MedSAM2 dependencies (if using real teacher model)
cd ../MedSAM2
pip install -e .
```

### Basic Usage

```python
import torch
from teacher_model import load_medsam2_teacher
from student_model import build_efficient_student_model
from training import KnowledgeDistillationTrainer

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
teacher = load_medsam2_teacher(
    config_file="configs/sam2.1_hiera_t512.yaml",
    checkpoint_path="checkpoints/MedSAM2_latest.pt",
    device=device
)

student = build_efficient_student_model(
    backbone_type="mobilenet_v3_small",
    device=device
)

# Create trainer
trainer = KnowledgeDistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device
)

# Start 3-stage training
trainer.train_all_stages(
    stage1_epochs=10,
    stage2_epochs=15, 
    stage3_epochs=5,
    base_lr=1e-4
)
```

### Model Comparison

```python
from utils import compare_models

# Compare performance
dummy_input = torch.randn(1, 3, 512, 512)
results = compare_models(
    teacher_model=teacher,
    student_model=student,
    input_data=dummy_input,
    device=device
)

print(f"Speedup: {results['improvements']['speedup']:.2f}x")
print(f"Memory reduction: {results['improvements']['memory_reduction']:.1f}%")
```

### Visualization

```python
from utils import MaskVisualizer

visualizer = MaskVisualizer()
fig = visualizer.compare_predictions(
    image=medical_image,
    teacher_mask=teacher_prediction,
    student_mask=student_prediction,
    points=point_prompts,
    save_path="comparison.png"
)
```

## 📁 Project Structure

```
Distillation/
├── teacher_model/
│   ├── __init__.py
│   └── medsam2.py              # MedSAM2 wrapper with feature extraction
├── student_model/
│   ├── __init__.py
│   ├── efficient_medsam2.py    # Main student architecture
│   ├── backbone.py             # Efficient backbone implementations
│   └── roi_memory_attention.py # ROI-aware memory attention
├── training/
│   ├── __init__.py
│   ├── losses.py               # Distillation loss functions
│   └── trainer.py              # 3-stage training pipeline
├── utils/
│   ├── __init__.py
│   ├── benchmarking.py         # Performance profiling tools
│   └── visualization.py        # Comparison visualization
└── README.md
```

## 🎛️ Configuration Options

### Backbone Choices
- **MobileNetV3**: Fastest inference, good for mobile devices
- **EfficientNet-B0**: Balanced speed/accuracy trade-off  
- **ViT-Tiny**: Best feature quality, moderate speed

### Memory Attention Settings
```python
roi_attention = ROIAwareMemoryAttention(
    d_model=256,
    nhead=4,
    num_layers=1,           # Reduced from teacher's 2-4 layers
    max_memory_frames=2,    # Reduced from teacher's 4 frames
    roi_dilation=2          # ROI expansion for context
)
```

### Loss Configuration
```python
distillation_loss = DistillationLoss(
    stage=2,                # Current training stage
    feature_weight=1.0,     # Feature distillation weight
    logit_weight=1.0,       # Logit distillation weight
    temperature=4.0,        # Softmax temperature
    dice_weight=20.0,       # Dice loss weight (MedSAM2 style)
    focal_weight=1.0        # Focal loss weight
)
```

## 📊 Expected Performance

### Efficiency Improvements
- **Speed**: 2-4x faster inference
- **Memory**: 50-70% reduction in GPU memory
- **Parameters**: 60-80% fewer parameters
- **Accuracy**: 95%+ retention of teacher performance

### Benchmark Results (Example)
| Model | Time (ms) | Memory (MB) | Parameters | Dice Score |
|-------|-----------|-------------|------------|------------|
| MedSAM2 (Teacher) | 150 | 1200 | 89.7M | 0.892 |
| EfficientMedSAM2 | 45 | 400 | 12.3M | 0.875 |
| **Improvement** | **3.3x faster** | **67% less** | **86% fewer** | **98% retained** |

## 🔧 Advanced Usage

### Custom Training Loop

```python
# Stage-by-stage training with custom settings
trainer.train_stage(
    stage=1,
    epochs=10,
    learning_rate=1e-5,    # Lower LR for feature matching
    save_frequency=5
)

# Load checkpoint and resume
checkpoint = trainer.load_checkpoint("efficient_medsam2_stage1_epoch10.pth")

# Continue with stage 2
trainer.train_stage(stage=2, epochs=15, learning_rate=1e-4)
```

### Custom Loss Weights

```python
# Adjust loss weights for different stages
stage2_loss = DistillationLoss(
    stage=2,
    feature_weight=0.5,     # Reduce feature weight
    logit_weight=2.0,       # Increase logit weight
    temperature=2.0         # Lower temperature for sharper distillation
)
```

### Quantization (Post-Training)

```python
# Apply INT8 quantization for deployment
import torch.quantization as quant

# Prepare model for quantization
student_model.qconfig = quant.get_default_qconfig('fbgemm')
student_prepared = quant.prepare(student_model)

# Calibrate with representative data
with torch.no_grad():
    for batch in calibration_loader:
        student_prepared(batch['image'])

# Convert to quantized model
student_quantized = quant.convert(student_prepared)
```

## 🧪 Evaluation and Testing

### Comprehensive Benchmarking

```python
from utils import ModelBenchmark, PerformanceProfiler

# Create benchmark suite
benchmark = ModelBenchmark(device="cuda")

# Profile models
teacher_result, student_result = benchmark.benchmark_model_pair(
    teacher_model=teacher,
    student_model=student,
    input_data=test_images
)

# Generate detailed report
report = benchmark.generate_report(
    teacher_result, student_result, 
    save_path="performance_report.txt"
)

# Create comparison plots
benchmark.plot_comparison(
    teacher_result, student_result,
    save_path="performance_comparison.png"
)
```

### Accuracy Evaluation

```python
def evaluate_model_accuracy(model, test_loader):
    model.eval()
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get predictions
            output = model(batch['images'], batch['points'], batch['labels'])
            pred_masks = torch.sigmoid(output['logits']) > 0.5
            
            # Calculate metrics
            dice = calculate_dice(pred_masks, batch['masks'])
            iou = calculate_iou(pred_masks, batch['masks'])
            
            dice_scores.extend(dice.cpu().numpy())
            iou_scores.extend(iou.cpu().numpy())
    
    return {
        'dice_mean': np.mean(dice_scores),
        'dice_std': np.std(dice_scores),
        'iou_mean': np.mean(iou_scores),
        'iou_std': np.std(iou_scores)
    }
```

## 🎨 Visualization Examples

### Side-by-Side Comparison
```python
# Create detailed comparison
visualizer = MaskVisualizer(figsize=(15, 10))
fig = visualizer.compare_predictions(
    image=ct_slice,
    teacher_mask=teacher_prediction,
    student_mask=student_prediction,
    points=[[256, 256]],  # Point prompt
    teacher_name="MedSAM2",
    student_name="EfficientMedSAM2",
    save_path="detailed_comparison.png"
)
```

### Performance Dashboard
```python
# Create comprehensive dashboard
dashboard = visualizer.create_performance_dashboard(
    benchmark_results={
        'teacher_result': teacher_result,
        'student_result': student_result,
        'improvements': improvements
    },
    accuracy_scores={
        'teacher_dice': 0.892,
        'student_dice': 0.875
    },
    save_path="performance_dashboard.png"
)
```

## 🔬 Research Applications

### Ablation Studies
- Compare different backbone architectures
- Evaluate ROI dilation sizes
- Test memory frame truncation limits
- Analyze temperature scaling effects

### Deployment Scenarios
- **Mobile Devices**: MobileNet backbone + INT8 quantization
- **Edge Computing**: EfficientNet with reduced memory frames
- **Real-time Applications**: ViT-Tiny with aggressive ROI masking

## 📈 Monitoring Training

### Logging and Visualization
```python
# Monitor training progress
trainer = KnowledgeDistillationTrainer(
    teacher_model=teacher,
    student_model=student,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    log_wandb=True  # Enable Weights & Biases logging
)

# Training history is automatically saved
print(trainer.train_history)
print(trainer.val_history)
```

### Checkpointing
```python
# Save/load checkpoints
trainer.save_checkpoint(stage=2, epoch=10, is_best=True)
checkpoint = trainer.load_checkpoint("efficient_medsam2_best.pth")
```

## 🚀 Deployment

### Export for Inference
```python
# Save final model for deployment
final_model = build_efficient_student_model(
    backbone_type="mobilenet_v3_small",
    checkpoint_path="efficient_medsam2_final.pth",
    device="cpu"
)

# Export to ONNX (optional)
torch.onnx.export(
    final_model,
    dummy_input,
    "efficient_medsam2.onnx",
    export_params=True,
    opset_version=11
)
```

### Integration Example
```python
from student_model import EfficientMedSAM2Predictor

# Load trained model
predictor = EfficientMedSAM2Predictor(final_model)

# Use in application
predictor.set_image(medical_image)
masks, scores, logits = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1])
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original MedSAM2 authors for the foundational work
- Meta AI for the SAM architecture
- Medical imaging community for dataset contributions
- PyTorch team for the deep learning framework

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{efficientmedsam2,
    title={EfficientMedSAM2: Knowledge Distillation for Efficient Medical Image Segmentation},
    author={Your Name},
    year={2025},
    eprint={arXiv:xxxx.xxxxx},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## 📞 Contact

For questions, issues, or collaborations:
- Email: your.email@institution.edu
- GitHub Issues: [Create an issue](../../issues)
- Project Link: [https://github.com/yourusername/EfficientMedSAM2](https://github.com/yourusername/EfficientMedSAM2)

---

**Happy distilling! 🔬✨**