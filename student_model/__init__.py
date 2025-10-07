"""
Student model package for EfficientMedSAM2.
"""

from .efficient_medsam2 import EfficientMedSAM2Student, build_efficient_student_model
from .backbone import EfficientBackbone
from .roi_memory_attention import ROIAwareMemoryAttention