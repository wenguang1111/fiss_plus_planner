"""Conditional Gaussian (K = 1) / Gaussian mixture (K > 1) sampler of the Sparse Planner parameters [t, d, v], as the
baseline of the attnCVAE (CVAE_efficient_sampling/attn_cvae.py) with the latent removed.

Same condition path as the attnCVAE: the 3 BEV frames go through the same ResNetTokenizer (ResNet18, ImageNet
weights, 256 px -> 64 tokens of 32), and the same decoder block (cross attention of 16 query tokens to the condition
tokens, projection, attention pooling FinalHead). The CVAE's queries are the projected latent z; here they are learned
constants, and the head outputs the parameters of p(x | c) instead of x:
  K = 1: mean (3) and Cholesky factor (6) of a full-covariance Gaussian;
  K > 1: K means, K Cholesky factors and K mixture logits.
Everything is in the normalised target space of the paper CVAE (TARGET_MEAN / TARGET_STD of cvae_paper.py)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from CVAE_efficient_sampling.attn_cvae import FinalHead, ResNetTokenizer, TransformerBlock

SOFTPLUS_ONE = math.log(math.e - 1.0)  # softplus(SOFTPLUS_ONE) = 1: the initial std is 1 (the marginal's scale)


class CondGaussian(nn.Module):
    def __init__(self, n_components: int = 1, hidden_dim: int = 32, img_size: int = 256, seq_len: int = 16,
                 x_dim: int = 3, min_std: float = 1e-3, pretrained: bool = True):
        super().__init__()
        self.K, self.x_dim, self.min_std = n_components, x_dim, min_std
        self.cond = ResNetTokenizer(img_size=img_size, hidden_dim=hidden_dim, pretrained=pretrained)
        self.query = nn.Parameter(torch.randn(1, seq_len, hidden_dim))  # in place of the CVAE's latent tokens
        self.attn = TransformerBlock(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.n_tril = x_dim * (x_dim + 1) // 2
        self.head = FinalHead(hidden_dim, output_dim=n_components * (x_dim + self.n_tril + 1))
        rows, cols = torch.tril_indices(x_dim, x_dim)
        self.register_buffer("tril_rows", rows, persistent=False)
        self.register_buffer("tril_cols", cols, persistent=False)
        self.register_buffer("diag_mask", (rows == cols), persistent=False)
        flat = torch.zeros(x_dim * x_dim, dtype=torch.long)  # entry of raw for each (row-major) matrix entry
        flat[rows * x_dim + cols] = torch.arange(self.n_tril)
        self.register_buffer("tril_gather", flat, persistent=False)
        self.register_buffer("tril_mask", torch.tril(torch.ones(x_dim, x_dim)).flatten(), persistent=False)

    def encode(self, c: torch.Tensor) -> torch.Tensor:
        """(B, 64, hidden) condition tokens of the (B, 3, H, W) frames."""
        return self.cond(c)

    def params(self, tokens: torch.Tensor):
        """Mixture logits (B, K), means (B, K, 3) and Cholesky factors (B, K, 3, 3) for the condition tokens."""
        B = tokens.shape[0]
        h = self.attn(q=self.query.expand(B, -1, -1), k=tokens, v=tokens)
        out = self.head(self.proj(h)).view(B, self.K, self.x_dim + self.n_tril + 1)
        mean = out[..., :self.x_dim]
        raw = out[..., self.x_dim:self.x_dim + self.n_tril]
        logits = out[..., -1]
        raw = torch.where(self.diag_mask, F.softplus(raw + SOFTPLUS_ONE) + self.min_std, raw)
        # lower triangle filled by gather x mask (the same values as scale_tril[..., rows, cols] = raw; that index
        # assignment exports to an invalid ONNX graph, which TensorRT rejects)
        full = raw[..., self.tril_gather] * self.tril_mask
        scale_tril = full.view(B, self.K, self.x_dim, self.x_dim)
        return logits, mean, scale_tril

    def distribution(self, tokens: torch.Tensor):
        logits, mean, scale_tril = self.params(tokens)
        components = MultivariateNormal(mean, scale_tril=scale_tril)
        return MixtureSameFamily(Categorical(logits=logits), components)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Per-sample negative log likelihood (B,) of the normalised targets x (B, 3) under p(x | c)."""
        return -self.distribution(self.encode(c)).log_prob(x)

    def sample_from_noise(self, tokens: torch.Tensor, eps: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
        """(n, 3) normalised samples for one condition (tokens (1, L, hidden)) from given noise: eps (n, 3) standard
        normal, uniform (n, K) in (0, 1) for the component choice (Gumbel-max, ignored for K = 1). No host sync, so it
        can be captured in a CUDA graph (GaussianGraphSampler); the same distribution as distribution(tokens)."""
        logits, mean, scale_tril = (p.float() for p in self.params(tokens))
        if self.K == 1:
            k = torch.zeros(eps.shape[0], dtype=torch.long, device=eps.device)
        else:
            gumbel = -torch.log(-torch.log(uniform.float().clamp(1e-10, 1.0 - 1e-7)))
            k = torch.argmax(logits[0] + gumbel, dim=-1)
        return mean[0, k] + (scale_tril[0, k] @ eps.float().unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def generate(self, c: torch.Tensor, batch: int) -> torch.Tensor:
        """(batch, 3) normalised samples for one condition c (1, 3, H, W), the condition encoded once (as
        CVAE_EfficientPaper.generate)."""
        eps = torch.randn(batch, self.x_dim, device=c.device)
        uniform = torch.rand(batch, self.K, device=c.device)
        return self.sample_from_noise(self.encode(c), eps, uniform)

    @torch.no_grad()
    def sample_per_condition(self, c: torch.Tensor, n: int) -> torch.Tensor:
        """(B, n, 3) normalised samples, n per condition of the batch c (B, 3, H, W)."""
        return self.distribution(self.encode(c)).sample((n,)).transpose(0, 1)
