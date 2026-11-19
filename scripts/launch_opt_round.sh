#!/bin/bash
# Launch 8 pure Loop-Muon optimizer ideas in parallel across GPUs 0..7 on one 8xB300 box.
# Usage: launch_opt_round.sh ROUND_DIR BACKBONE STEPS BATCH MICRO idea0..idea7
set -euo pipefail
ROUND_DIR=$1; shift
BACKBONE=$1; shift
STEPS=$1; shift
BATCH=$1; shift
MICRO=$1; shift
IDEAS=("$@")
if [ "${#IDEAS[@]}" -ne 8 ]; then
  echo "Expected exactly 8 ideas, got ${#IDEAS[@]}" >&2
  exit 1
fi
mkdir -p "$ROUND_DIR"
for g in 0 1 2 3 4 5 6 7; do
  idea="${IDEAS[$g]}"
  CUDA_VISIBLE_DEVICES=$g NANOCHAT_BASE_DIR=/dev/shm/nt-runtime/resources \
    nohup /root/nt/.venv/bin/python /root/pn/loop-muon/muon_opt_arena.py \
      --idea "$idea" --backbone "$BACKBONE" \
      --steps "$STEPS" --batch-size "$BATCH" --micro-batch "$MICRO" \
      --out "$ROUND_DIR/gpu${g}_${idea}.json" \
      > "$ROUND_DIR/gpu${g}_${idea}.log" 2>&1 &
  echo "GPU $g -> $idea ($BACKBONE, pid $!)"
done
