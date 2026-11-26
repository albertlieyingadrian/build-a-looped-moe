#!/bin/sh
# The commands behind the numbers in stage_descriptions/base-12-compare.md and base-13-lens.md, on a laptop CPU.
# Usage: ./run_all.sh [seed]
set -e
SEED=${1:-0}
P=solutions/python/14-layout/code/your_program.sh
OUT=runs/seed$SEED
mkdir -p "$OUT"
$P compare --seed "$SEED" --n-embd 64 --batch-size 64 --lr 1e-3 --steps 4000 --hops 3 --out-dir "$OUT" --verbose
$P lens --seed "$SEED" --n 500 --out-dir "$OUT"
