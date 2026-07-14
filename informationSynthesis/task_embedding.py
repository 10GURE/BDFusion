"""
Task Embedding Module for All-in-One Image Restoration

This module implements learnable task embeddings for different degradation types,
following the methodology of prompt-based learning in vision transformers.

Key Design Principles:
1. Each degradation type has learnable token embeddings that capture task-specific semantics
2. Task embeddings participate in cross-attention with image features
3. The model can learn degradation-aware feature representations

References:
- PromptIR: Prompting for All-in-One Blind Image Restoration (NeurIPS 2023)
- AirNet: All-in-One Image Restoration for Unknown Corruption (CVPR 2022)
- Visual Prompt Tuning (ECCV 2022)

Author: Generated for BDFusion Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Dict, List


class DegradationType:
    """
    Enumeration of supported degradation types.
    
    Each degradation type is assigned a unique integer ID for embedding lookup.
    The design follows the taxonomy commonly used in image restoration literature.
    """
    CLEAN = 0           # No degradation (for testing/baseline)
    RAIN = 1            # Rain streak removal
    FOG = 2             # Dehazing / Fog removal  
    LOW_LIGHT = 3       # Low-light enhancement
    NOISE = 4           # Denoising (Gaussian, Poisson, etc.)
    BLUR = 5            # Deblurring (motion, defocus)
    MIXED = 6           # Mixed/compound degradations
    JPEG = 7            # JPEG artifact removal
    SR = 8              # Super-resolution (optional)
    
    NUM_TYPES = 9       # Total number of degradation types


# Human-readable names for logging and visualization
DEGRADATION_NAMES: Dict[int, str] = {
    0: 'clean',
    1: 'rain', 
    2: 'fog',
    3: 'low_light',
    4: 'noise',
    5: 'blur',
    6: 'mixed',
    7: 'jpeg',
    8: 'sr'
}

# Reverse mapping: name -> id
DEGRADATION_NAME_TO_ID: Dict[str, int] = {v: k for k, v in DEGRADATION_NAMES.items()}


class TaskTokenEmbedding(nn.Module):
    """
    Learnable Task Token Embeddings for degradation-aware conditioning.
    
    This module generates task-specific token sequences that can be used
    as additional context in cross-attention mechanisms.
    
    Mathematical Formulation:
    - Each task t has K learnable tokens: T_t = {t_1, t_2, ..., t_K} ∈ R^{K×D}
    - These tokens serve as "prompts" that guide the restoration process
    - During cross-attention: Attention(Q_image, K_task, V_task)
    
    Args:
        num_tasks: Number of degradation types
        embed_dim: Dimension of each token embedding
        num_tokens: Number of tokens per task (more tokens = more expressiveness)
        dropout: Dropout rate for regularization
        
    Shape:
        Input: task_id [B] or [B, 1]
        Output: [B, num_tokens, embed_dim]
    """
    
    def __init__(
        self,
        num_tasks: int = DegradationType.NUM_TYPES,
        embed_dim: int = 256,
        num_tokens: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        # Learnable task tokens: [num_tasks, num_tokens, embed_dim]
        # Initialized with Xavier uniform for stable training
        self.task_tokens = nn.Parameter(
            torch.empty(num_tasks, num_tokens, embed_dim)
        )
        nn.init.xavier_uniform_(self.task_tokens)
        
        # Optional: Learnable position embeddings for token ordering
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Projection layer for dimension matching (if needed downstream)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, task_id: torch.Tensor) -> torch.Tensor:
        """
        Retrieve task tokens for given task IDs.
        
        Args:
            task_id: Task indices [B] or [B, 1]
            
        Returns:
            Task tokens [B, num_tokens, embed_dim]
        """
        if task_id.dim() == 2:
            task_id = task_id.squeeze(-1)
            
        # Index into task tokens
        tokens = self.task_tokens[task_id]  # [B, num_tokens, embed_dim]
        
        # Add positional embeddings
        tokens = tokens + self.pos_embed
        
        # Project and normalize
        tokens = self.proj(tokens)
        tokens = self.norm(tokens)
        
        return tokens
    
    def get_all_tokens(self) -> torch.Tensor:
        """Return all task tokens for visualization/analysis."""
        return self.task_tokens.data


class TaskScaleEmbedding(nn.Module):
    """
    Task Embedding that produces scale and shift parameters (FiLM-style).
    
    This is an alternative embedding that produces per-channel modulation
    parameters instead of attention tokens. Useful for ResBlock conditioning.
    
    Mathematical Formulation:
    - For task t, produce γ_t and β_t ∈ R^C
    - Feature modulation: y = γ_t * x + β_t
    
    Args:
        num_tasks: Number of degradation types
        feature_dim: Number of feature channels to modulate
        
    Shape:
        Input: task_id [B]
        Output: (gamma, beta) each [B, feature_dim]
    """
    
    def __init__(
        self,
        num_tasks: int = DegradationType.NUM_TYPES,
        feature_dim: int = 256
    ):
        super().__init__()
        
        self.gamma = nn.Embedding(num_tasks, feature_dim)
        self.beta = nn.Embedding(num_tasks, feature_dim)
        
        # Initialize gamma to 1, beta to 0 (identity mapping initially)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        
    def forward(self, task_id: torch.Tensor):
        """
        Get scale and shift parameters for given tasks.
        
        Args:
            task_id: [B]
            
        Returns:
            Tuple of (gamma, beta), each [B, feature_dim]
        """
        gamma = self.gamma(task_id)
        beta = self.beta(task_id)
        return gamma, beta


class TaskAwareEncoder(nn.Module):
    """
    Multi-scale Task-Aware Feature Encoder.
    
    Produces task-conditioned features at multiple resolutions for
    injection into the UNet at different levels.
    
    This follows the hierarchical conditioning approach where task
    information is injected at each resolution level.
    
    Args:
        num_tasks: Number of degradation types
        embed_dim: Base embedding dimension
        num_tokens: Tokens per task
        num_levels: Number of resolution levels (matching UNet)
        channel_mult: Channel multipliers at each level
    """
    
    def __init__(
        self,
        num_tasks: int = DegradationType.NUM_TYPES,
        embed_dim: int = 256,
        num_tokens: int = 4,
        num_levels: int = 4,
        channel_mult: tuple = (1, 2, 4, 4)
    ):
        super().__init__()
        
        self.num_levels = num_levels
        
        # Base task embedding
        self.task_embed = TaskTokenEmbedding(
            num_tasks=num_tasks,
            embed_dim=embed_dim,
            num_tokens=num_tokens
        )
        
        # Level-specific projections
        self.level_projs = nn.ModuleList()
        for i, mult in enumerate(channel_mult):
            out_dim = embed_dim * mult
            self.level_projs.append(
                nn.Sequential(
                    nn.Linear(embed_dim, out_dim),
                    nn.GELU(),
                    nn.Linear(out_dim, out_dim),
                    nn.LayerNorm(out_dim)
                )
            )
    
    def forward(self, task_id: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multi-scale task embeddings.
        
        Args:
            task_id: [B]
            
        Returns:
            List of task embeddings at each level
        """
        base_tokens = self.task_embed(task_id)  # [B, num_tokens, embed_dim]
        
        level_tokens = []
        for proj in self.level_projs:
            tokens = proj(base_tokens)
            level_tokens.append(tokens)
            
        return level_tokens


class DegradationClassifier(nn.Module):
    """
    Lightweight Degradation Type Classifier.
    
    Optional module for automatic degradation detection when task_id
    is not provided during inference. This enables blind restoration.
    
    Architecture: Simple CNN followed by global pooling and classifier.
    
    Args:
        in_channels: Input image channels (3 for RGB)
        num_classes: Number of degradation types
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = DegradationType.NUM_TYPES,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict degradation type from input image.
        
        Args:
            x: Input image [B, C, H, W]
            
        Returns:
            Logits [B, num_classes]
        """
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)


# ============================================================================
# Utility Functions
# ============================================================================

def get_task_id_from_name(name: str) -> int:
    """Convert degradation name to task ID."""
    name_lower = name.lower().strip()
    if name_lower in DEGRADATION_NAME_TO_ID:
        return DEGRADATION_NAME_TO_ID[name_lower]
    else:
        raise ValueError(f"Unknown degradation type: {name}. "
                        f"Available: {list(DEGRADATION_NAME_TO_ID.keys())}")


def get_task_name_from_id(task_id: int) -> str:
    """Convert task ID to degradation name."""
    if task_id in DEGRADATION_NAMES:
        return DEGRADATION_NAMES[task_id]
    else:
        raise ValueError(f"Unknown task ID: {task_id}. "
                        f"Available: {list(DEGRADATION_NAMES.keys())}")


def create_task_batch(task_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create a batch of task IDs for a given degradation type."""
    task_id = get_task_id_from_name(task_name)
    return torch.full((batch_size,), task_id, dtype=torch.long, device=device)
