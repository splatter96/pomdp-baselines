#!/bin/bash

VALUES=(0 5 10 15 20)
MAX_JOBS=7

mkdir -p sweep_results

for RCS in "${VALUES[@]}"; do
    for T in "${VALUES[@]}"; do

        # Wait until fewer than MAX_JOBS are running
        while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do
            wait -n
        done

        echo "Starting RCS=${RCS}, T=${T}..."

        (
            PYTHONPATH="${PWD}:$PYTHONPATH" \
            python3 policies/enjoy.py \
                render=False \
                env.observation.RCS="${RCS}" \
                env.observation.T="${T}" \
                > "sweep_results/RCS_${RCS}_T_${T}.txt" 2>&1

            echo "Finished RCS=${RCS}, T=${T}"
        ) &

    done
done

# Wait for the final jobs to finish
wait

echo "Sweep complete."
