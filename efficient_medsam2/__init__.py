"""
EfficientMedSAM2 - A memory-efficient implementation of MedSAM2 for medical image segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Type, Union

# Import constants
NO_OBJ_SCORE = -1024.0  # a large negative value as a placeholder score for missing objects
