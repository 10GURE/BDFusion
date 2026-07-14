"""
Degradation Encoder for Blind All-in-One Image Restoration

This module implements a degradation-aware encoder that automatically extracts
task-relevant representations from degraded images WITHOUT explicit task labels.

Key Design Principles:
1. The encoder learns to extract degradation-specific features from image content
2. No explicit degradation type labels are needed during training
3. Similar degradations are mapped to similar embeddings (via contrastive learning)
4. The learned embeddings serve as task vectors to guide the restoration network

References:
- AirNet: All-in-One Image Restoration for Unknown Corruption (CVPR 2022)
- IDR: Ingredient-oriented Degradation Reformulation (CVPR 2023)
- Unified Image Restoration (ICLR 2023)

Author: Generated for BDFusion Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, List
import math


class DegradationEncoder(nn.Module):
    """
    Degradation-Aware Encoder that extracts task vectors from degraded images.
    
    This encoder automatically learns to identify degradation patterns from
    image content, without requiring explicit degradation type labels.
    
    Architecture:
    - Lightweight CNN backbone for feature extraction
    - Global pooling to obtain image-level representation
    - Projection head to produce task vector
    
    The encoder is trained end-to-end with the restoration network.
    Optionally, contrastive learning can be used to improve degradation discrimination.
    
    Args:
        in_channels: Input image channels (3 for RGB)
        embed_dim: Dimension of the output task vector
        hidden_dims: Hidden layer dimensions in the encoder
        num_tokens: Number of task tokens to generate (for cross-attention)
        use_multi_scale: Whether to use multi-scale features
        
    Shape:
        Input: [B, C, H, W] degraded image
        Output: [B, num_tokens, embed_dim] task tokens
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        hidden_dims: List[int] = [64, 128, 256, 256],
        num_tokens: int = 4,
        use_multi_scale: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.use_multi_scale = use_multi_scale
        
        # Build encoder backbone
        layers = []
        prev_dim = in_channels
        for i, dim in enumerate(hidden_dims):
            layers.extend([
                nn.Conv2d(prev_dim, dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            prev_dim = dim
        
        self.backbone = nn.Sequential(*layers)
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-scale feature extraction (optional)
        if use_multi_scale:
            self.scale_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dims[0], kernel_size=k, stride=k//2, padding=k//4),
                    nn.BatchNorm2d(hidden_dims[0]),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.AdaptiveAvgPool2d(1)
                )
                for k in [3, 5, 7]
            ])
            total_feat_dim = hidden_dims[-1] + hidden_dims[0] * 3
        else:
            total_feat_dim = hidden_dims[-1]
        
        # Projection to task vector space
        self.projector = nn.Sequential(
            nn.Linear(total_feat_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim * num_tokens),
        )
        
        # Layer norm for output
        self.out_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract task vector from degraded image.
        
        Args:
            x: Degraded image [B, C, H, W], normalized to [-1, 1]
            
        Returns:
            Task tokens [B, num_tokens, embed_dim]
        """
        B = x.shape[0]
        
        # Main backbone features
        feat = self.backbone(x)
        feat = self.global_pool(feat).flatten(1)  # [B, hidden_dims[-1]]
        
        # Multi-scale features
        if self.use_multi_scale:
            scale_feats = [conv(x).flatten(1) for conv in self.scale_convs]
            feat = torch.cat([feat] + scale_feats, dim=1)
        
        # Project to task token space
        tokens = self.projector(feat)  # [B, embed_dim * num_tokens]
        tokens = tokens.view(B, self.num_tokens, self.embed_dim)
        tokens = self.out_norm(tokens)
        
        return tokens
    
    def get_degradation_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get a single degradation embedding vector (mean of tokens).
        Useful for visualization and analysis.
        """
        tokens = self.forward(x)
        return tokens.mean(dim=1)  # [B, embed_dim]


class ContrastiveDegradationEncoder(DegradationEncoder):
    """
    Degradation Encoder with Contrastive Learning Support.
    
    Extends DegradationEncoder with contrastive learning capability
    following AirNet's CBDE (Contrastive-Based Degradation Encoder).
    
    The contrastive loss encourages:
    - Images with same degradation type → similar embeddings
    - Images with different degradation types → dissimilar embeddings
    
    Note: During training, this requires sampling pairs/triplets of images.
    The contrastive loss is computed separately and added to the main loss.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        projection_dim: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.temperature = temperature
        
        # Additional projection head for contrastive learning
        self.contrast_projector = nn.Sequential(
            nn.Linear(kwargs.get('embed_dim', 256), projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def get_contrastive_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features for contrastive loss computation."""
        tokens = self.forward(x)
        pooled = tokens.mean(dim=1)  # [B, embed_dim]
        projected = self.contrast_projector(pooled)  # [B, projection_dim]
        return F.normalize(projected, dim=-1)
    
    def contrastive_loss(
        self, 
        anchor: torch.Tensor, 
        positive: torch.Tensor, 
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            anchor: Anchor features [B, D]
            positive: Positive (same degradation) features [B, D]
            negatives: Negative features [N, D] or None (use in-batch negatives)
            
        Returns:
            Scalar loss
        """
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [B]
        
        if negatives is None:
            # Use in-batch negatives (other samples in the batch)
            # similarity matrix: [B, B]
            sim_matrix = torch.mm(anchor, positive.t()) / self.temperature
            # Mask out positive pairs (diagonal)
            mask = torch.eye(anchor.shape[0], device=anchor.device).bool()
            neg_sim = sim_matrix.masked_fill(mask, float('-inf'))
            # Log-sum-exp over negatives
            neg_logsumexp = torch.logsumexp(neg_sim, dim=-1)
        else:
            # Use provided negatives
            neg_sim = torch.mm(anchor, negatives.t()) / self.temperature  # [B, N]
            neg_logsumexp = torch.logsumexp(neg_sim, dim=-1)
        
        # InfoNCE loss
        loss = -pos_sim + torch.logsumexp(
            torch.stack([pos_sim, neg_logsumexp], dim=-1), dim=-1
        )
        
        return loss.mean()


class HierarchicalDegradationEncoder(nn.Module):
    """
    Hierarchical Degradation Encoder for Multi-Scale Task Vectors.
    
    Produces task vectors at multiple scales to match the UNet's
    hierarchical structure. Each level receives task guidance.
    
    This design allows degradation-aware modulation at every resolution level.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_tokens: int = 4,
        num_levels: int = 4,
        channel_mult: Tuple[int, ...] = (1, 2, 3, 4)
    ):
        super().__init__()
        
        self.num_levels = num_levels
        
        # Base encoder
        self.base_encoder = DegradationEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_tokens=num_tokens
        )
        
        # Level-specific projections
        self.level_projectors = nn.ModuleList()
        for mult in channel_mult:
            out_dim = embed_dim * mult
            self.level_projectors.append(
                nn.Sequential(
                    nn.Linear(embed_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Linear(out_dim, out_dim)
                )
            )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale task vectors.
        
        Args:
            x: Degraded image [B, C, H, W]
            
        Returns:
            List of task tokens at each level
        """
        base_tokens = self.base_encoder(x)  # [B, num_tokens, embed_dim]
        
        level_tokens = []
        for proj in self.level_projectors:
            tokens = proj(base_tokens)
            level_tokens.append(tokens)
        
        return level_tokens


class DualStreamDegradationEncoder(nn.Module):
    """
    Dual-Stream Degradation Encoder for RGB-IR Fusion.
    
    Extracts degradation information from both RGB and IR modalities,
    then fuses them to produce a unified task vector.
    
    This is specifically designed for the RGB-IR fusion task where
    both modalities may have different degradation patterns.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_tokens: int = 4,
        fusion_type: str = 'concat'  # 'concat', 'add', 'attention'
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Separate encoders for RGB and IR
        self.rgb_encoder = DegradationEncoder(
            in_channels=3,
            embed_dim=embed_dim,
            num_tokens=num_tokens
        )
        self.ir_encoder = DegradationEncoder(
            in_channels=3,  # IR is converted to 3-channel
            embed_dim=embed_dim,
            num_tokens=num_tokens
        )
        
        # Fusion layer
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU()
            )
        elif fusion_type == 'attention':
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim, num_heads=4, batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(embed_dim)
        # 'add' fusion doesn't need extra layers
    
    def forward(
        self, 
        rgb: torch.Tensor, 
        ir: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract fused task vector from RGB and IR.
        
        Args:
            rgb: RGB degraded image [B, 3, H, W]
            ir: IR degraded image [B, 3, H, W]
            
        Returns:
            Fused task tokens [B, num_tokens, embed_dim]
        """
        rgb_tokens = self.rgb_encoder(rgb)  # [B, K, D]
        ir_tokens = self.ir_encoder(ir)     # [B, K, D]
        
        if self.fusion_type == 'concat':
            fused = torch.cat([rgb_tokens, ir_tokens], dim=-1)
            fused = self.fusion(fused)
        elif self.fusion_type == 'add':
            fused = rgb_tokens + ir_tokens
        elif self.fusion_type == 'attention':
            # Cross-attention: RGB attends to IR
            attended, _ = self.fusion_attn(rgb_tokens, ir_tokens, ir_tokens)
            fused = self.fusion_norm(rgb_tokens + attended)
        
        return fused


# ============================================================================
# Utility Functions
# ============================================================================

def build_degradation_encoder(
    encoder_type: str = 'basic',
    **kwargs
) -> nn.Module:
    """
    Factory function to build degradation encoder.
    
    Args:
        encoder_type: One of 'basic', 'contrastive', 'hierarchical', 'dual_stream'
        **kwargs: Arguments passed to the encoder
        
    Returns:
        Degradation encoder module
    """
    encoders = {
        'basic': DegradationEncoder,
        'contrastive': ContrastiveDegradationEncoder,
        'hierarchical': HierarchicalDegradationEncoder,
        'dual_stream': DualStreamDegradationEncoder
    }
    
    if encoder_type not in encoders:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                        f"Available: {list(encoders.keys())}")
    
    return encoders[encoder_type](**kwargs)
