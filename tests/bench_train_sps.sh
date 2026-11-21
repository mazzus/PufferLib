#!/bin/bash
# Training throughput protocol: fixed step count, SPS = steps / wall time
# (includes startup), repeated. Run on an otherwise idle machine.
#   ./tests/bench_train_sps.sh BINARY STEPS REPEATS [--section.key=value ...]
# Prints one line per run and the median.
set -e
BIN=$1; STEPS=$2; REPEATS=$3; shift 3
results=()
for r in $(seq 1 "$REPEATS"); do
    t0=$(python3 -c 'import time; print(time.time())')
    "$BIN" train --train.total_timesteps="$STEPS" "$@" > /dev/null 2>&1
    t1=$(python3 -c 'import time; print(time.time())')
    sps=$(python3 -c "print(round($STEPS / ($t1 - $t0)))")
    results+=("$sps")
    echo "run $r: steps=$STEPS wall=$(python3 -c "print(round($t1 - $t0, 2))")s sps=$sps"
done
python3 -c "import statistics,sys; print('median_sps=%d' % statistics.median(map(int, sys.argv[1:])))" "${results[@]}"
