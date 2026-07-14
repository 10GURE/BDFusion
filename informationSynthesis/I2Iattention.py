from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class Channel_Attention(nn.Module):
    def __init__(self, F_h, F_hs, F_bg, F_int):
        super(Channel_Attention, self).__init__()
        self.W_h = nn.Sequential(
            nn.Conv2d(F_h, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_hs = nn.Sequential(
            nn.Conv2d(F_hs, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_bg = nn.Sequential(
            nn.Conv2d(F_bg, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, h, hs, bg):
        h1 = self.W_h(h)
        hs1 = self.W_hs(hs)
        bg1 = self.W_bg(bg)
        psi = self.relu(h1 + hs1 +bg1)
        psi = self.psi(psi)

        return hs * psi


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class CrossAttention(nn.Module):
    """Standard multi-head cross-attention using F.scaled_dot_product_attention.
    Replaces DeformableCrossAttention — faster on small token counts (64-1024 tokens),
    FP16-compatible, torch.compile-friendly. See FlashAttention paper.
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        """x: [B, N, D]  context: [B, M, D_ctx]"""
        context = default(context, x)
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b m (h d) -> b h m d', h=h)
        v = rearrange(v, 'b m (h d) -> b h m d', h=h)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# ── Legacy DeformableCrossAttention (kept for weight compatibility) ──

class BasicDMSAttn(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, contextv=None, contexti=None):
        return checkpoint(self._forward, (x, contextv, contexti), self.parameters(), self.checkpoint)

    def _forward(self, x, contextv=None, contexti=None):
        # x: [B, C, H, W], contextv/contexti: [B, C, Hc, Wc]
        x = rearrange(x, 'b c h w -> b (h w) c')
        contextv = rearrange(contextv, 'b c h w -> b (h w) c')
        contexti = rearrange(contexti, 'b c h w -> b (h w) c')

        x = self.attn1(self.norm1(x), context=contextv) + x
        x = self.attn2(self.norm2(x), context=contexti) + x
        x = self.ff(self.norm3(x)) + x
        return x


class DualModalSpraseAttn(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicDMSAttn(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for _ in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context1=None, context2=None):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, contextv=context1, contexti=context2)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


# ============================================================================
# Task-Conditioned Attention Modules for All-in-One Restoration
# ============================================================================

class TaskCrossAttention(nn.Module):
    """
    Cross-Attention with Task Token Conditioning.
    
    This module implements task-aware cross-attention where task tokens
    participate as additional key-value pairs in the attention mechanism.
    
    Mathematical Formulation:
    Given image query Q, and context (K_ctx, V_ctx) from RGB/IR features,
    we augment with task tokens (K_task, V_task):
    
        K = [K_ctx; K_task]
        V = [V_ctx; V_task]
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    
    This allows the model to attend to task-specific information while
    processing spatial features, enabling degradation-aware restoration.
    
    References:
    - Perceiver (Jaegle et al., ICML 2021) - cross-attention with learnable queries
    - PromptIR (NeurIPS 2023) - prompt-based image restoration
    
    Args:
        query_dim: Dimension of query features
        context_dim: Dimension of context features (can differ from query)
        task_dim: Dimension of task tokens
        heads: Number of attention heads
        dim_head: Dimension per head
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int = None,
        task_dim: int = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        
        context_dim = default(context_dim, query_dim)
        task_dim = default(task_dim, query_dim)
        
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        # Query projection from image features
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        
        # Key-Value projections for context (RGB/IR features)
        self.to_k_ctx = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_ctx = nn.Linear(context_dim, inner_dim, bias=False)
        
        # Key-Value projections for task tokens
        self.to_k_task = nn.Linear(task_dim, inner_dim, bias=False)
        self.to_v_task = nn.Linear(task_dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable weight for task attention contribution
        self.task_weight = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x, context=None, task_tokens=None):
        """
        Forward pass with optional task conditioning.
        
        Args:
            x: Query features [B, N, D] where N = H*W
            context: Context features [B, M, D_ctx]
            task_tokens: Task token embeddings [B, K, D_task]
            
        Returns:
            Output features [B, N, D]
        """
        h = self.heads
        
        # Compute query
        q = self.to_q(x)  # [B, N, inner_dim]
        
        # Compute context key-value
        context = default(context, x)
        k_ctx = self.to_k_ctx(context)
        v_ctx = self.to_v_ctx(context)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k_ctx = rearrange(k_ctx, 'b m (h d) -> b h m d', h=h)
        v_ctx = rearrange(v_ctx, 'b m (h d) -> b h m d', h=h)
        
        if task_tokens is not None:
            # Compute task key-value
            k_task = self.to_k_task(task_tokens)
            v_task = self.to_v_task(task_tokens)
            
            k_task = rearrange(k_task, 'b k (h d) -> b h k d', h=h)
            v_task = rearrange(v_task, 'b k (h d) -> b h k d', h=h)
            
            # Concatenate context and task key-values
            k = torch.cat([k_ctx, k_task], dim=2)  # [B, h, M+K, d]
            v = torch.cat([v_ctx, v_task], dim=2)  # [B, h, M+K, d]
        else:
            k = k_ctx
            v = v_ctx
        
        # Compute attention
        sim = torch.einsum('b h n d, b h m d -> b h n m', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.einsum('b h n m, b h m d -> b h n d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class TaskConditionedDMSAttn(nn.Module):
    """
    Task-Conditioned Dual-Modal Sparse Attention Block.
    
    Extends BasicDMSAttn to incorporate task token conditioning.
    The task tokens are injected via an additional cross-attention layer
    that modulates the fused RGB-IR features.
    
    Architecture:
    1. Cross-attention with RGB context
    2. Cross-attention with IR context  
    3. Cross-attention with Task tokens (NEW)
    4. Feed-forward network
    
    This hierarchical design allows the model to first fuse multi-modal
    information, then apply task-specific modulation.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        d_head: int,
        dropout: float = 0.0,
        context_dim: int = None,
        task_dim: int = None,
        gated_ff: bool = True,
        checkpoint: bool = True
    ):
        super().__init__()
        
        context_dim = default(context_dim, dim)
        task_dim = default(task_dim, dim)
        
        # Dual-modal attention (existing)
        self.attn1 = DeformableCrossAttention(
            query_dim=dim, context_dim=context_dim,
            heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.attn2 = DeformableCrossAttention(
            query_dim=dim, context_dim=context_dim,
            heads=n_heads, dim_head=d_head, dropout=dropout
        )
        
        # Task-conditioned attention (NEW)
        self.attn_task = TaskCrossAttention(
            query_dim=dim, task_dim=task_dim,
            heads=n_heads, dim_head=d_head, dropout=dropout
        )
        
        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)  # For task attention
        self.norm4 = nn.LayerNorm(dim)  # For FFN
        
        self.checkpoint = checkpoint
        
    def forward(self, x, contextv=None, contexti=None, task_tokens=None):
        """
        Forward with optional task conditioning.
        
        Args:
            x: Input features [B, C, H, W]
            contextv: RGB context features [B, C, H, W]
            contexti: IR context features [B, C, H, W]
            task_tokens: Task embeddings [B, K, D] (optional)
        """
        bs, c, h, w = x.shape
        
        # Prepare spatial shapes for deformable attention
        spatial_shapes = torch.as_tensor([(h, w)], dtype=torch.long, device=x.device)
        
        bs, c, h_v, w_v = contextv.shape
        spatial_shapes_v = torch.as_tensor([(h_v, w_v)], dtype=torch.long, device=contextv.device)
        
        bs, c, h_i, w_i = contexti.shape
        spatial_shapes_i = torch.as_tensor([(h_i, w_i)], dtype=torch.long, device=contexti.device)
        
        # Reshape to sequence format
        x = rearrange(x, 'b c h w -> b (h w) c')
        contextv = rearrange(contextv, 'b c h w -> b (h w) c')
        contexti = rearrange(contexti, 'b c h w -> b (h w) c')
        
        # Cross-attention with RGB
        x = self.attn1(self.norm1(x), spatial_shapes, spatial_shapes_v, context=contextv) + x
        
        # Cross-attention with IR
        x = self.attn2(self.norm2(x), spatial_shapes, spatial_shapes_i, context=contexti) + x
        
        # Task-conditioned attention (only if task_tokens provided)
        if task_tokens is not None:
            x = self.attn_task(self.norm3(x), task_tokens=task_tokens) + x
        
        # Feed-forward
        x = self.ff(self.norm4(x)) + x
        
        return x


class TriModalSparseAttn(nn.Module):
    """
    Tri-Modal Sparse Attention: RGB + IR + Task Tokens.
    
    This is the main attention block for All-in-One restoration.
    It extends DualModalSpraseAttn to incorporate task conditioning.
    
    Key Features:
    1. Preserves original dual-modal attention behavior
    2. Adds optional task token conditioning
    3. Backward compatible (works without task tokens)
    
    Args:
        in_channels: Input feature channels
        n_heads: Number of attention heads
        d_head: Dimension per head
        depth: Number of transformer blocks
        dropout: Dropout rate
        context_dim: Context feature dimension
        task_dim: Task token dimension (default: same as context_dim)
        use_task_conditioning: Whether to use task conditioning
    """
    
    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: int = None,
        task_dim: int = None,
        use_task_conditioning: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.use_task_conditioning = use_task_conditioning
        
        inner_dim = n_heads * d_head
        task_dim = default(task_dim, inner_dim)
        
        self.norm = Normalize(in_channels)
        
        self.proj_in = nn.Conv2d(
            in_channels, inner_dim,
            kernel_size=1, stride=1, padding=0
        )
        
        # Use task-conditioned blocks if enabled
        if use_task_conditioning:
            self.transformer_blocks = nn.ModuleList([
                TaskConditionedDMSAttn(
                    inner_dim, n_heads, d_head,
                    dropout=dropout, context_dim=context_dim,
                    task_dim=task_dim
                )
                for _ in range(depth)
            ])
        else:
            self.transformer_blocks = nn.ModuleList([
                BasicDMSAttn(
                    inner_dim, n_heads, d_head,
                    dropout=dropout, context_dim=context_dim
                )
                for _ in range(depth)
            ])
        
        self.proj_out = zero_module(nn.Conv2d(
            inner_dim, in_channels,
            kernel_size=1, stride=1, padding=0
        ))
        
    def forward(self, x, context1=None, context2=None, task_tokens=None):
        """
        Forward pass with optional task conditioning.
        
        Args:
            x: Input features [B, C, H, W]
            context1: RGB context features
            context2: IR context features
            task_tokens: Task embeddings [B, K, D] (optional)
            
        Returns:
            Output features [B, C, H, W]
        """
        b, c, h, w = x.shape
        x_in = x
        
        x = self.norm(x)
        x = self.proj_in(x)
        
        for block in self.transformer_blocks:
            if self.use_task_conditioning and task_tokens is not None:
                x = block(x, contextv=context1, contexti=context2, task_tokens=task_tokens)
            else:
                x = block(x, contextv=context1, contexti=context2)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        
        return x + x_in

