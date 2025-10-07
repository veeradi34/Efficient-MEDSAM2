"""
Training package for knowledge distillation.
"""

from .losses import DistillationLoss, FocalLoss, DiceLoss
from .trainer import KnowledgeDistillationTrainer