"""
Fast inference samplers for BDFusion: DPM-Solver++ and optimized DDIM.
Target: reduce sampling latency from ~1.2s to <100ms while preserving quality.
"""

import torch
import numpy as np
import math
from tqdm import tqdm
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like


# ──────────────────────────────────────────────
# DPM-Solver++ 2S (Singlestep, 2nd order)
# Reference: https://arxiv.org/abs/2206.00927
# ──────────────────────────────────────────────

class DPMSolverSampler:
    """
    DPM-Solver++ 2S sampler for fast inference.
    
    Supports:
    - order=2 (recommended, best quality/speed tradeoff)
    - order=1 (equivalent to DDIM, but with better scheduler)
    - use_karras: Karras et al. noise schedule for better quality at low steps
    """
    
    def __init__(self, model, algorithm_type="dpmsolver++", solver_type="heun"):
        self.model = model
        self.algorithm_type = algorithm_type
        self.solver_type = solver_type
        self._prepared = False
        self._cached_order = None
        self._cached_steps = None
    
    def _prepare(self, steps, order, device):
        """Pre-compute sigmas and corresponding timesteps once."""
        if self._prepared and self._cached_steps == steps and self._cached_order == order:
            return
        
        alphas_cumprod = self.model.alphas_cumprod.to(device)
        n_total = len(alphas_cumprod)
        
        # Get n+1 sigmas (including sigma=0 at the end)
        indices = torch.linspace(n_total - 1, 0, steps + 1, device=device).long()
        acp = alphas_cumprod[indices]
        
        # sigma = sqrt((1 - alpha_cumprod) / alpha_cumprod)
        sigmas = torch.sqrt((1.0 - acp) / acp.clamp(min=1e-10))
        sigmas[-1] = 0.0
        
        # Pre-compute alpha_t, sigma_t for each step
        self._sigmas = sigmas
        self._timesteps = indices
        self._alpha_t = 1.0 / torch.sqrt(1.0 + sigmas ** 2)
        self._sigma_t = sigmas * self._alpha_t
        
        # Pre-compute log(sigma) for DPM-Solver++ coefficients
        self._log_sigmas = torch.log(sigmas.clamp(min=1e-10))
        
        self._prepared = True
        self._cached_steps = steps
        self._cached_order = order
    
    @torch.no_grad()
    def sample(self, cond1, cond2, batch_size, shape, steps=10, order=2,
               eta=0.0, verbose=False, use_karras=False, task_tokens=None,
               use_cuda_graph=False, z_init=None):
        """
        DPM-Solver++ sampling (optimized, no searchsorted overhead).
        
        Set use_cuda_graph=True to capture and replay the sampling loop via CUDA Graph.
        Requires order=1 (first-order, uniform operations per step).
        
        Args:
            z_init: Optional [B, C, H, W] tensor to use as initial noise.
                    When shared across frames, this enforces temporal noise consistency.
        """
        device = self.model.betas.device
        C, H, W = shape
        b = batch_size
        
        if use_cuda_graph and order != 1:
            order = 1  # CUDA Graph requires uniform steps (first-order only)
        
        # Pre-compute schedule
        self._prepare(steps, order, device)
        sigmas = self._sigmas
        timesteps = self._timesteps
        alpha_t = self._alpha_t
        sigma_t = self._sigma_t
        log_sigmas = self._log_sigmas
        
        # Initialize noise (shared or fresh)
        if z_init is not None:
            x = z_init[:b].to(device=device, dtype=torch.float32)
        else:
            x = torch.randn((b, C, H, W), device=device)
        
        # Match cond batch size
        if cond1.shape[0] != b:
            cond1 = cond1[:b]
        if cond2.shape[0] != b:
            cond2 = cond2[:b]
        
        if use_cuda_graph and order == 1:
            return self._sample_cuda_graph(x, cond1, cond2, b, steps, alpha_t, sigma_t, 
                                           timesteps, sigmas, task_tokens, device, verbose)
        
        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, desc=f'DPM-Solver++ ({order}S, {steps} steps)', total=steps)
        
        denoised_prev = None
        
        for i in iterator:
            s_i = timesteps[i].item()
            ts = torch.full((b,), s_i, device=device, dtype=torch.long)
            
            # Model call: epsilon prediction (pass cached task tokens)
            denoised = self.model.apply_model(x, ts, cond1, cond2, task_tokens=task_tokens)
            
            if order == 1 or denoised_prev is None:
                # First-order update
                at = alpha_t[i]
                st = sigma_t[i]
                
                # x0 = (x - sigma_t * eps) / alpha_t
                x0 = (x - st * denoised) / at.clamp(min=1e-10)
                
                # Update
                at_next = alpha_t[i + 1]
                st_next = sigma_t[i + 1]
                x = at_next * x0 + st_next * denoised
                
                if eta > 0 and sigmas[i + 1] > 0:
                    noise = torch.randn_like(x)
                    x = x + eta * (st_next ** 2 / (st ** 2 + 1e-10)).sqrt() * noise
            else:
                # Second-order DPM-Solver++ 2S
                lambda_i = log_sigmas[i]
                lambda_s = log_sigmas[i - 1]
                lambda_ip1 = log_sigmas[i + 1]
                
                h = lambda_ip1 - lambda_i
                r = (lambda_i - lambda_s) / h.clamp(min=1e-10)
                
                at = alpha_t[i]
                st = sigma_t[i]
                
                x0 = (x - st * denoised) / at.clamp(min=1e-10)
                
                at_next = alpha_t[i + 1]
                st_next = sigma_t[i + 1]
                
                # Heun correction
                D_combined = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * denoised_prev
                x = at_next * x0 + st_next * D_combined
                
                if eta > 0 and sigmas[i + 1] > 0:
                    noise = torch.randn_like(x)
                    x = x + eta * st_next * noise
            
            denoised_prev = denoised
        
        return x, None

    def _sample_cuda_graph(self, x, cond1, cond2, b, steps, alpha_t, sigma_t, 
                           timesteps, sigmas, task_tokens, device, verbose):
        """CUDA Graph captured first-order sampling loop."""
        # Static tensors for graph capture
        x_static = x.clone()
        ts_static = torch.full((b,), 0, device=device, dtype=torch.long)
        
        # Schedule values as graph inputs (scalars expanded to tensors for capture)
        at_in = torch.zeros(1, device=device)
        st_in = torch.zeros(1, device=device)
        at_next_in = torch.zeros(1, device=device)
        st_next_in = torch.zeros(1, device=device)
        
        # Capture a single step
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            denoised = self.model.apply_model(x_static, ts_static, cond1, cond2, task_tokens=task_tokens)
            x0 = (x_static - st_in * denoised) / at_in.clamp(min=1e-10)
            x_out = at_next_in * x0 + st_next_in * denoised
        
        iterator = range(steps)
        if verbose:
            iterator = tqdm(iterator, desc=f'DPM-Solver (1S+Graph, {steps} steps)', total=steps)
        
        for i in iterator:
            s_i = timesteps[i].item()
            ts_static.fill_(s_i)
            x_static.copy_(x)
            at_in.fill_(alpha_t[i].item())
            st_in.fill_(sigma_t[i].item())
            at_next_in.fill_(alpha_t[i + 1].item())
            st_next_in.fill_(sigma_t[i + 1].item())
            g.replay()
            x.copy_(x_out)
        
        return x, None


# ──────────────────────────────────────────────
# Optimized DDIM Sampler (with reduced overhead)
# ──────────────────────────────────────────────

class OptimizedDDIMSampler:
    """Optimized DDIM sampler with fused operations and reduced python overhead."""
    
    def __init__(self, model, schedule="linear"):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self._buffers_registered = False
        self._prepared = False
    
    def _register_buffers(self, ddim_num_steps, ddim_eta=0.0):
        """Register all required buffers on CUDA once."""
        alphas_cumprod = self.model.alphas_cumprod
        
        # Pre-compute DDIM timesteps
        ddim_ts = make_ddim_timesteps(
            ddim_discr_method="uniform",
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=False
        )
        self.ddim_timesteps = ddim_ts
        
        # Move to CUDA once
        device = self.model.device
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(device)
        
        self.betas = to_torch(self.model.betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(self.model.alphas_cumprod_prev)
        
        # DDIM parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,
            verbose=False
        )
        
        self.ddim_sigmas = torch.as_tensor(ddim_sigmas, device=device)
        self.ddim_alphas = torch.as_tensor(ddim_alphas, device=device)
        self.ddim_alphas_prev = torch.as_tensor(ddim_alphas_prev, device=device)
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)
        
        self._buffers_registered = True
        self._ddim_steps = ddim_num_steps
    
    @torch.no_grad()
    def sample(self, cond1, cond2, batch_size, shape, steps, eta=0.0, verbose=True,
               task_tokens=None, z_init=None):
        """Optimized single DDIM sampling pass."""
        if not self._buffers_registered or self._ddim_steps != steps:
            self._register_buffers(steps, eta)
        
        device = self.model.betas.device
        b, C, H, W = batch_size, shape[0], shape[1], shape[2]
        
        # Initialize noise (shared or fresh)
        if z_init is not None:
            img = z_init[:b].to(device=device, dtype=torch.float32)
        else:
            img = torch.randn((b, C, H, W), device=device)
        
        ddim_alphas = self.ddim_alphas
        ddim_alphas_prev = self.ddim_alphas_prev
        ddim_sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        ddim_sigmas = self.ddim_sigmas
        
        total_steps = len(self.ddim_timesteps)
        ts_np = np.asarray(self.ddim_timesteps)
        time_range = np.flip(ts_np)
        
        iterator = tqdm(time_range, desc='DDIM (opt)', total=total_steps) if verbose else time_range
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            # Single model call (pass cached task tokens)
            e_t = self.model.apply_model(img, ts, cond1, cond2, task_tokens=task_tokens)
            
            # Fused DDIM update
            a_t = ddim_alphas[index]
            a_prev = ddim_alphas_prev[index]
            sigma_t = ddim_sigmas[index]
            sqrt_one_minus_at = ddim_sqrt_one_minus_alphas[index]
            
            # pred_x0
            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            
            # direction to x_t
            dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
            
            if sigma_t > 0:
                noise = torch.randn_like(img)
                img = a_prev.sqrt() * pred_x0 + dir_xt + sigma_t * noise
            else:
                img = a_prev.sqrt() * pred_x0 + dir_xt
        
        return img, None
