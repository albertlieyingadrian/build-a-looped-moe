"""Stages 02-04: blocks, the active block, and depth attention (each sublayer's own mix of the blocks)."""

import math

import torch
import torch.nn as nn

from app.model import Attention, RMSNorm, ffn, rms


def depth_attention(query, stack):
    """Softmax over the block axis of (normalized block . query): the mix weights.
    stack: (B, T, N, D), already normalized; query: (Q, D). Returns (B, T, N, Q)."""
    return torch.softmax(stack @ query.float().T / math.sqrt(stack.shape[-1]), dim=-2)


class DepthMix(nn.Module):
    """The input a sublayer reads: a softmax-weighted average of the RMS-normalized blocks (finished + active),
    scored against a learned query. Queries start at zero, so every mix starts as a plain average.
    mix=False returns the plain SUM of the blocks instead, which is exactly an ordinary residual stream."""

    def __init__(self, c, n_query=1):
        super().__init__()
        self.mix = c.mix
        self.query = nn.Parameter(torch.zeros(n_query, c.n_embd))

    def weights(self, blocks, active):
        normed = rms(torch.cat([blocks, active.unsqueeze(-2)], dim=-2).float())  # (B, T, N+1, D)
        return normed, depth_attention(self.query, normed)

    def forward(self, blocks, active):
        if not self.mix:
            return [blocks.sum(-2) + active] * self.query.shape[0]
        normed, w = self.weights(blocks, active)
        return list(torch.einsum("btnq,btnd->btqd", w, normed).to(active.dtype).unbind(-2))


class Layer(nn.Module):
    """One decoder layer. `h` is the active block (the sum of sublayer outputs since the block started);
    `blocks` is (B, T, N, D), the finished blocks so far."""

    def __init__(self, c, idx, looped):
        super().__init__()
        self.boundary = idx % c.block_layers == 0  # this layer starts a new block
        self.attn_input = DepthMix(c, n_query=3) if idx > 0 else None
        self.mlp_input = DepthMix(c)
        self.norm_q, self.norm_k, self.norm_v, self.norm_mlp = (RMSNorm(c.n_embd) for _ in range(4))
        self.attn, self.ffn = Attention(c), ffn(c)
        self.gates = None
        if looped and c.messages:  # stage 11
            from app.loopmoe import MessageGate

            self.gates = nn.ModuleList(MessageGate(c) for _ in range(4))

    def forward(self, h, blocks, msgs=None, rec=None):
        start = h
        xq, xk, xv = self.attn_input(blocks, h) if self.attn_input else (h, h, h)
        if msgs is not None and self.gates is not None:
            xq, xk, xv = (g(x, m) for g, x, m in zip(self.gates[:3], (xq, xk, xv), msgs[:3]))
        a = self.attn(self.norm_q(xq), self.norm_k(xk), self.norm_v(xv))
        if self.boundary:  # freeze the block we were building; start a new one with this output
            blocks = torch.cat([blocks, start.unsqueeze(-2)], dim=-2)
            h = a
        else:
            h = start + a
        x = self.mlp_input(blocks, h)[0]
        if msgs is not None and self.gates is not None:
            x = self.gates[3](x, msgs[3])
        if rec is not None:
            rec.append({"act_in": start, "mlp_in": x, "blocks": blocks})
        return h + self.ffn(self.norm_mlp(x)), blocks
