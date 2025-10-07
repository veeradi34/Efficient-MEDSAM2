"""
EfficientMedSAM2 Knowledge Distillation Package

This package implements knowledge distillation from MedSAM2 to create
a more efficient variant suitable for resource-constrained environments.

Modules:
- teacher_model: Wrapper for original MedSAM2 model
- student_model: Efficient student architecture
- training: Knowledge distillation training pipeline
- utils: Utilities for benchmarking and visualization
"""

__version__ = "1.0.0"
__author__ = "EfficientMedSAM2 Team"

# Import key components
from .teacher_model import MedSAM2Teacher
from .student_model import EfficientMedSAM2Student
from .training import KnowledgeDistillationTrainer