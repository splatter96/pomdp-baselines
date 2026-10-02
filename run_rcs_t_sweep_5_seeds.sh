#!/usr/bin/env bash

VALUES=(0 5 10 15 20)

# Five independent evaluation seeds per (RCS, T) configuration.
# Change these values if you want to use a different seed set.
SEEDS=(42 43 44 45 46)

# Maximum number of evaluation processes running at the same time.
MAX_JOBS=5

OUTPUT_DIR="sweep_results"
mkdir -p "${OUTPUT_DIR}"

run_eval() {
    local RCS="$1"
    local T="$2"
    local SEED="$3"
    local OUTPUT_FILE="${OUTPUT_DIR}/RCS_${RCS}_T_${T}_seed_${SEED}.txt"

    echo "Starting RCS=${RCS}, T=${T}, seed=${SEED}"

    if PYTHONPATH="${PWD}:$PYTHONPATH" \
        python3 policies/enjoy.py \
            render=False \
            env.observation.RCS="${RCS}" \
            env.observation.T="${T}" \
            seed="${SEED}" \
            > "${OUTPUT_FILE}" 2>&1
    then
        echo "Finished RCS=${RCS}, T=${T}, seed=${SEED}"
    else
        status=$?
        echo "FAILED RCS=${RCS}, T=${T}, seed=${SEED} (exit code ${status})" >&2
    fi
}

for RCS in "${VALUES[@]}"; do
    for T in "${VALUES[@]}"; do
        for SEED in "${SEEDS[@]}"; do

            # Keep at most MAX_JOBS processes alive at once.
            while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
                wait -n
            done

            run_eval "${RCS}" "${T}" "${SEED}" &
        done
    done
done

# Wait for the final jobs to finish.
wait

echo "Sweep complete."
