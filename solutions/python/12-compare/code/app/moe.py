"""Stages 06-08: a sigmoid top-k router, a mixture of experts, and router-bias load balancing."""

import math

import torch
import torch.nn as nn

from app.model import SwiGLU, softcap


class MoE(nn.Module):
    """Top-k mixture of experts with a sigmoid router.

    Each token scores every expert with sigmoid(router(x)), keeps the top_k, renormalizes their scores to sum to 1
    and scales them by route_scale (= top_k). `bias` is added to the scores only when CHOOSING experts, never when
    weighting them: it is the load-balancing knob that balance() nudges after every training step.
    The experts' weights are stacked (E, ...). At toy size it is faster to run every expert on every token and give
    the unchosen ones a weight of exactly 0; the result is the same as routing token by token."""

    def __init__(self, c):
        super().__init__()
        E, D, H = c.n_experts, c.n_embd, c.d_expert
        self.top_k, self.route_scale = c.top_k, float(c.top_k)
        self.router = nn.Linear(D, E, bias=False)
        self.w_gate, self.w_up = (nn.Parameter(torch.randn(E, D, H) / math.sqrt(D)) for _ in range(2))
        self.w_down = nn.Parameter(torch.randn(E, H, D) / math.sqrt(H))
        self.shared = SwiGLU(D, c.n_shared * H) if c.n_shared else None
        self.register_buffer("bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E))  # tokens per expert since the last balance()

    def route(self, flat):
        """(N, E) routing weights with top_k nonzero entries per token summing to route_scale, and the choices."""
        scores = torch.sigmoid(self.router(flat).float())
        idx = torch.topk(scores + self.bias, self.top_k, dim=-1).indices
        w = scores.gather(1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20) * self.route_scale
        return torch.zeros_like(scores).scatter(1, idx, w).to(flat.dtype), idx

    def forward(self, x):
        flat = x.reshape(-1, x.shape[-1])
        w, idx = self.route(flat)
        if self.training:
            self.load += torch.bincount(idx.flatten(), minlength=len(self.bias)).float()
        g = torch.einsum("nd,edh->neh", flat, self.w_gate)
        hid = softcap(g, 4.0) * torch.sigmoid(g) * softcap(torch.einsum("nd,edh->neh", flat, self.w_up), 25.0)
        out = torch.einsum("neh,ehd,ne->nd", hid, self.w_down, w)
        if self.shared is not None:
            out = out + self.shared(flat)
        return out.view_as(x)

    @torch.no_grad()
    def balance(self, rate):
        """Aux-loss-free balancing: raise the bias of under-used experts, lower it for over-used ones."""
        if self.load.sum() > 0:
            self.bias += rate * torch.sign(self.load.mean() - self.load)
        self.load.zero_()
