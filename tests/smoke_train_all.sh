#!/bin/bash
# Smoke-train every env whose trainer binary exists in build/envs/puffer_ENV:
# run up to SECONDS_EACH (default 45) and require at least one completed epoch
# (dashboard "Epoch" >= 1) and no crash. Prints PASS/FAIL per env.
#   ./tests/smoke_train_all.sh [SECONDS_EACH]
SECS=${1:-45}
for bin in build/envs/puffer_*; do
    [ -f "$bin" ] && [ -x "$bin" ] || continue  # skip .dSYM bundles
    env=${bin#build/envs/puffer_}
    log=build/envs/smoke_$env.log
    perl -e 'alarm shift; exec @ARGV' "$SECS" "$bin" train > "$log" 2>&1
    status=$?
    epoch=$(LC_ALL=C sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" "$log" | grep -aoE "Epoch +[0-9]+" | tail -1 | awk '{print $2}')
    sps=$(LC_ALL=C sed "s/\x1b\[[0-9;?]*[a-zA-Z]//g" "$log" | grep -aoE "SPS +[0-9.]+[KM]?" | tail -1 | awk '{print $2}')
    # 142 = SIGALRM (our timeout): fine if it trained
    if [ -n "$epoch" ] && [ "$epoch" -ge 1 ] && { [ $status -eq 0 ] || [ $status -eq 142 ]; }; then
        echo "PASS $env epochs=$epoch sps=$sps"
    else
        echo "FAIL $env status=$status epochs=${epoch:-0} :: $(grep -aE 'Assert|error|unsupported|abort|Segmentation' "$log" | head -1 | cut -c1-160)"
    fi
done
