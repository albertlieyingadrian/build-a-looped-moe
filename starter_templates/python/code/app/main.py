"""Your looped MoE. Run it with ./your_program.sh <command> [options].

Each stage adds one command. Start with stage 01 (`recap`): read stage_descriptions/base-01-recap.md, write the plain
looped GPT and the chain puzzle, and make it pass `make test STAGE=01 DIR=<your copy of this folder>`.

Suggested layout as you go (the reference solutions use it, but any layout passes the tests until stage 14, which
loads the one-file model's weights by parameter name):
    app/model.py   stage 01 (Config, attention, SwiGLU, PlainLoop)   app/chain.py   stage 01   app/train.py  stage 01
    app/blocks.py  stages 02-04 (DepthMix, Layer)                    app/loopmoe.py stages 05, 09-11
    app/moe.py     stages 06-08                                      app/lens.py    stage 13
"""

import sys


def main():
    # Debug output goes to stderr, so it never confuses the tests, which read stdout.
    print("Logs from your program will appear here!", file=sys.stderr)

    # TODO: stage 01. `./your_program.sh recap --steps 300` must print, for K in 1, 3 and 6,
    #
    #     K=1  parameters 60448  layer visits 8
    #
    # (the same parameter count at every K), then train for --steps steps on the chain task and print
    #
    #     loss over the first 20 steps: 2.891
    #     loss over the last 20 steps: 0.636
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    raise RuntimeError(f"Unknown command {command!r}")


if __name__ == "__main__":
    main()
