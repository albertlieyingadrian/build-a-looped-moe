"""Stages 05 and 09-11: the whole model. Prelude, core x K (every pass restarts from the prelude's blocks and gets
a gated message from the previous pass), coda, and the output head's own mix of the blocks (final_input)."""

import torch
import torch.nn as nn

from app.blocks import DepthMix, Layer, depth_attention
from app.model import RMSNorm, output, rms


class MessageGate(nn.Module):
    """x <- (1 - g_forget) * x + g_message * message, both gates tanh of a rank-r projection of [norm(x), norm(msg)].
    Both projections start at zero, so the gate starts closed: every pass begins as a non-looped model."""

    def __init__(self, c):
        super().__init__()
        self.down = nn.Linear(2 * c.n_embd, c.gate_rank, bias=False)
        self.forget = nn.Linear(c.gate_rank, c.n_embd, bias=False)
        self.message = nn.Linear(c.gate_rank, c.n_embd, bias=False)
        nn.init.zeros_(self.forget.weight)
        nn.init.zeros_(self.message.weight)

    def forward(self, x, msg):
        z = self.down(torch.cat([rms(x), rms(msg)], dim=-1))
        return (1 - torch.tanh(self.forget(z))) * x + torch.tanh(self.message(z)) * msg


class MessageInput(nn.Module):
    """Four messages (q, k, v, MoE input) per core layer, read from the message bank (the previous pass's finished
    core blocks plus its last active block) with one learned query each."""

    def __init__(self, c, n_core_layers):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(n_core_layers, 4, c.n_embd))

    def forward(self, bank_blocks, bank_active, core_layer):
        stack = rms(torch.cat([bank_blocks, bank_active.unsqueeze(-2)], dim=-2).float())
        w = depth_attention(self.query[core_layer], stack)
        return list(torch.einsum("btnq,btnd->btqd", w, stack).to(bank_active.dtype).unbind(-2))


class LoopMoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        L, p, k = c.block_layers, c.prelude_blocks, c.core_blocks
        self.core_start, self.core_end = L * p, L * (p + k)
        self.tok = nn.Embedding(len(c.vocab), c.n_embd)
        self.layers = nn.ModuleList(Layer(c, i, self.core_start <= i < self.core_end) for i in range(c.n_layer))
        self.message_input = MessageInput(c, self.core_end - self.core_start)
        self.final_input = DepthMix(c)
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
        h = self.tok(idx)
        blocks = h.new_zeros(*h.shape[:-1], 0, h.shape[-1])
        for layer in self.layers[: self.core_start]:
            h, blocks = layer(h, blocks, rec=rec)
        pre_h, pre_blocks = h, blocks
        n_pre, bank = blocks.shape[-2], None
        for _ in range(K):
            h, blocks = pre_h, pre_blocks  # every pass restarts from the prelude
            for j, layer in enumerate(self.layers[self.core_start : self.core_end]):
                msgs = self.message_input(*bank, j) if bank is not None and self.c.messages else None
                h, blocks = layer(h, blocks, msgs, rec=rec)
            bank = (blocks[..., n_pre + 1 :, :], h)  # +1 skips the prelude's block, frozen by the first core layer
        for layer in self.layers[self.core_end :]:
            h, blocks = layer(h, blocks, rec=rec)
        if rec is not None:
            rec.append({"act_in": h, "blocks": blocks})
        return output(self, self.final_input(blocks, h)[0], targets)
