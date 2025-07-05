#!/usr/bin/env bash
#
# run_batches.sh
#
# Run render.sh in small batches to avoid blowing up memory.

# how many times to call render.sh:
NUM_RUNS=10

for (( run=1; run<=NUM_RUNS; run++ )); do
  echo "===== Batch $run of $NUM_RUNS ====="
  . ./render.sh
  echo
done
