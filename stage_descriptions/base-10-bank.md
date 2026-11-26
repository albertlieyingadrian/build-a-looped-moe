In this stage, you'll keep what the previous pass built and read messages from it.

### The idea: a bank of the last draft

When a pass ends, its finished core blocks and its last active block go into the **message bank**. In Inkling 2 Mini
that is the two middle blocks (layers 6-11 and 12-17). In pass 2 and 3, **every core layer** reads **four messages**
from the bank, one each for its attention's query, key and value inputs and one for its MoE input, each with its own
learned query vector, using the same depth attention as stage 03.

```
bank (pass p-1):  [core block A, last active block B]
                         │  MessageInput: 4 learned queries per core layer
                         ▼
pass p, core layer j:   m_q, m_k, m_v, m_mlp
```

**Verified:** `message_input` holds one query per message per loop layer, shape `(12, 4, hidden)`, zero-initialized;
the bank is the previous pass's blocks after the prelude's block, plus its last hidden state.

One detail: the first core layer is a block boundary, so it freezes the *prelude's* active block into the list. The
bank skips that one (`blocks[..., n_pre + 1:, :]`): it belongs to the prelude, not to the pass.

### Worked example

Toy layout, K = 2: the core has 4 layers in 2 blocks. Pass 2's 4 core layers each read from a bank of 1 finished
core block + 1 active block, and each gets 4 messages. With zero queries each message is the plain average of the
two normalized bank entries.

```bash
$ ./your_program.sh bank
message reads per forward at K=2: 4 (one per core layer of pass 2)
bank: 1 finished core blocks + 1 active block
messages per core layer: 4, each of shape (2, 40, 32)
zero queries: every message is the bank's average: True
```

### What to build

`MessageInput(c, n_core_layers)` with `forward(bank_blocks, bank_active, core_layer)` returning 4 tensors, and the
bank bookkeeping in `LoopMoE.forward`. Pass the messages to each core `Layer`; stage 11 decides what the layer does
with them.

### Tests

The command must report 4 reads, 1 + 1 bank entries, 4 messages and `True`.

### Check yourself

1. Why does pass 1 get no messages?
2. In Inkling 2 Mini, how many message queries are there in total?
