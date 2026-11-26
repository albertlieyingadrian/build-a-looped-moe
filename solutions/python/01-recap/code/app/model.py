"""Stage 01: the pieces every model here shares, and the plain looped GPT (one residual stream)."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    task: str = "chain"
    arch: str = "loopmoe"  # "loopmoe", "plainloop" (one residual stream) or "standard" (plainloop with K = 1)
    n_embd: int = 128
    n_head: int = 4
    block_layers: int = 2  # layers per block (Inkling 2 Mini: 6)
    prelude_blocks: int = 1
    core_blocks: int = 2
    coda_blocks: int = 1
    loops: int = 3  # K: core passes
    random_loops: bool = False
    mix: bool = True  # block mixing; False = read the plain sum of the blocks (a residual stream)
    messages: bool = True  # gated messages between core passes
    gate_rank: int = 16
    moe: bool = True  # top-k MoE; False = a dense SwiGLU with the same active width
    n_experts: int = 16
    top_k: int = 4
    n_shared: int = 0
    d_expert: int = 64
    bias_rate: float = 1e-3
    hops: int = 8
    block_size: int = 96
    batch_size: int = 128
    steps: int = 6000
    lr: float = 5e-4
    seed: int = 1337
    device: str = "auto"
    out: str = "out/model.pt"
    vocab: str = ""

    @property
    def n_layer(self):
        return self.block_layers * (self.prelude_blocks + self.core_blocks + self.coda_blocks)


def rms(x, eps=1e-6):
    """RMS norm with no learned scale: every vector is rescaled to length sqrt(dim)."""
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return rms(x) * self.weight


def rope(x):
    """Rotary positions: q.k depends on the distance between two tokens (a stand-in for Inkling's relative bias)."""
    T, D = x.shape[-2], x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device) / D))
    ang = torch.arange(T, device=x.device)[:, None] * freqs
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * ang.cos() - x2 * ang.sin(), x1 * ang.sin() + x2 * ang.cos()], -1).flatten(-2)


class Attention(nn.Module):
    """Causal multi-head attention whose queries, keys and values may read different inputs."""

    def __init__(self, c):
        super().__init__()
        self.n_head = c.n_head
        self.q, self.k, self.v, self.o = (nn.Linear(c.n_embd, c.n_embd, bias=False) for _ in range(4))

    def forward(self, xq, xk, xv):
        B, T, C = xq.shape
        q, k, v = (
            f(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            for f, x in ((self.q, xq), (self.k, xk), (self.v, xv))
        )
        y = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


def softcap(x, cap):
    return torch.tanh(x / cap) * cap


class SwiGLU(nn.Module):
    """down(softcap(g, 4) * sigmoid(g) * softcap(up, 25)): Inkling's softcapped SwiGLU."""

    def __init__(self, d, hidden):
        super().__init__()
        self.gate, self.up = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.down = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        g = self.gate(x)
        return self.down(softcap(g, 4.0) * torch.sigmoid(g) * softcap(self.up(x), 25.0))


def ffn(c):
    """The feed-forward sublayer: the MoE of stage 07, or a dense SwiGLU with the same active width."""
    if c.moe:
        from app.moe import MoE

        return MoE(c)
    return SwiGLU(c.n_embd, (c.top_k + c.n_shared) * c.d_expert)


def output(model, x, targets):
    """Final norm, head and a logit softcap of 30; the loss ignores positions whose target is -1."""
    logits = softcap(model.head(model.norm(x)), 30.0)
    if targets is None:
        return logits, None
    return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


class PlainLayer(nn.Module):
    """One residual stream: x = x + attn(norm(x)); x = x + ffn(norm(x))."""

    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
        self.attn, self.ffn = Attention(c), ffn(c)

    def forward(self, x):
        n = self.n1(x)
        x = x + self.attn(n, n, n)
        return x + self.ffn(self.n2(x))


class PlainLoop(nn.Module):
    """The looped course's model at this layout: prelude, core x K with input injection, coda, one stream."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.core_start = c.block_layers * c.prelude_blocks
        self.core_end = self.core_start + c.block_layers * c.core_blocks
        self.tok = nn.Embedding(len(c.vocab), c.n_embd)
        self.layers = nn.ModuleList(PlainLayer(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, len(c.vocab), bias=False)

    def visits(self, K):
        """(layer index, pass) for every layer visit, in the order they run."""
        core = range(self.core_start, self.core_end)
        return (
            [(i, 0) for i in range(self.core_start)]
            + [(i, p) for p in range(K) for i in core]
            + [(i, 0) for i in range(self.core_end, self.c.n_layer)]
        )

    def forward(self, idx, loops=None, targets=None, rec=None):
        K = loops or self.c.loops
        x = self.tok(idx)
        for layer in self.layers[: self.core_start]:
            x = layer(x)
        e, x = x, torch.zeros_like(x)
        for _ in range(K):
            x = x + e  # input injection
            for layer in self.layers[self.core_start : self.core_end]:
                if rec is not None:
                    rec.append({"act_in": x})
                x = layer(x)
        for layer in self.layers[self.core_end :]:
            if rec is not None:
                rec.append({"act_in": x})
            x = layer(x)
        if rec is not None:
            rec.append({"act_in": x})
        return output(self, x, targets)


def build(c):
    if c.arch == "loopmoe":
        from app.loopmoe import LoopMoE

        return LoopMoE(c)
    return PlainLoop(c)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def count_params(model):
    """(stored, active per token). Active counts only the top_k routed experts (plus shared)."""
    total, idle = n_params(model), 0
    for m in model.modules():
        if hasattr(m, "w_gate"):
            per_expert = (m.w_gate.numel() + m.w_up.numel() + m.w_down.numel()) // len(m.bias)
            idle += per_expert * (len(m.bias) - m.top_k)
    return total, total - idle
