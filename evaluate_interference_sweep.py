#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


EVAL_DUTYCYCLES = [0, 3, 14, 20, 30, 40, 50]

# Prevent output from multiple worker threads being interleaved.
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def find_latest_checkpoint(save_dir: Path) -> Path:
    """
    Select checkpoint with the highest iteration number.

    Expected filenames, for example:
        agent_100_perf83.506.pt
        agent_1900_perf99.749.pt
        agent_2000_perf68.373.pt
    """
    checkpoints = list(save_dir.glob("*.pt"))

    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {save_dir}")

    def checkpoint_key(path: Path):
        match = re.search(r"agent_(\d+)", path.name)

        if match:
            iteration = int(match.group(1))
        else:
            iteration = -1

        return iteration, path.stat().st_mtime_ns

    return max(checkpoints, key=checkpoint_key)


def load_run_info(variant_file: Path):
    with variant_file.open("r") as f:
        cfg = yaml.safe_load(f)

    train_seed = cfg["seed"]

    interference_enabled = (
        cfg.get("env", {})
        .get("observation", {})
        .get("enable_interference", True)
    )

    if interference_enabled:
        train_dutycycle = cfg["env"]["dutycycle"]
    else:
        train_dutycycle = 0

    return train_seed, train_dutycycle


def evaluate_run(
    variant_file: Path,
    project_root: Path,
    output_dir: Path,
    eval_seed: int,
    overwrite: bool,
    dry_run: bool,
    env: dict,
):
    """
    Evaluate one trained run for all configured evaluation duty cycles.
    """

    run_dir = variant_file.parent
    save_dir = run_dir / "save"

    try:
        train_seed, train_dutycycle = load_run_info(variant_file)
        checkpoint = find_latest_checkpoint(save_dir)

    except Exception as e:
        safe_print(
            f"[SKIP] {run_dir}\n"
            f"       {e}"
        )
        return

    safe_print(
        "\n" + "=" * 80 + "\n"
        f"Run:                {run_dir}\n"
        f"Training dutycycle: {train_dutycycle}\n"
        f"Training seed:      {train_seed}\n"
        f"Checkpoint:         {checkpoint.name}\n"
        + "=" * 80
    )

    main_script = project_root / "policies" / "enjoy.py"

    for eval_dutycycle in EVAL_DUTYCYCLES:

        output_name = (
            f"trained_on_{train_dutycycle}"
            f"_train_seed_{train_seed}"
            f"_eval_duty_{eval_dutycycle}.txt"
        )

        output_file = output_dir / output_name

        if output_file.exists() and not overwrite:
            safe_print(
                f"[SKIP] train duty={train_dutycycle:2d}, "
                f"train seed={train_seed}, "
                f"eval duty={eval_dutycycle:2d} "
                f"(output exists)"
            )
            continue

        cmd = [
            "python3",
            str(main_script),
            f"env.dutycycle={eval_dutycycle}",
            f"seed={eval_seed}",
            f"chkpt_path={checkpoint.resolve()}",
        ]

        safe_print(
            f"[RUN ] train duty={train_dutycycle:2d}, "
            f"train seed={train_seed}, "
            f"eval duty={eval_dutycycle:2d}\n"
            f"       -> {output_file.name}"
        )

        if dry_run:
            safe_print("       " + " ".join(cmd))
            continue

        with output_file.open("w") as f:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

        if result.returncode != 0:
            safe_print(
                f"[FAIL] train duty={train_dutycycle:2d}, "
                f"train seed={train_seed}, "
                f"eval duty={eval_dutycycle:2d}, "
                f"exit code={result.returncode}"
            )
        else:
            safe_print(
                f"[ OK ] train duty={train_dutycycle:2d}, "
                f"train seed={train_seed}, "
                f"eval duty={eval_dutycycle:2d}"
            )

    safe_print(
        f"[DONE] train duty={train_dutycycle}, "
        f"train seed={train_seed}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained agents over multiple radar duty cycles."
    )

    parser.add_argument(
        "runs_dir",
        type=Path,
        help="Root directory containing the training runs.",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Root of the project containing policies/enjoy.py. "
             "Default: current directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory for evaluation output files.",
    )

    parser.add_argument(
        "--eval-seed",
        type=int,
        default=21,
        help="Seed used during evaluation. Default: 21.",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of training runs to evaluate in parallel. Default: 1.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite evaluation files that already exist.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    runs_dir = args.runs_dir.resolve()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    main_script = project_root / "policies" / "enjoy.py"

    if not main_script.exists():
        raise FileNotFoundError(
            f"Could not find policies/enjoy.py in {project_root}"
        )

    # Find all variant YAML files recursively.
    variant_files = sorted(
        list(runs_dir.rglob("variant_*.yml"))
        + list(runs_dir.rglob("variant_*.yaml"))
    )

    if not variant_files:
        raise RuntimeError(
            f"No variant_*.yml or variant_*.yaml files found under {runs_dir}"
        )

    print(f"Found {len(variant_files)} training runs.")
    print(f"Evaluation duty cycles: {EVAL_DUTYCYCLES}")
    print(f"Parallel training runs: {args.jobs}")
    print()

    # Equivalent to:
    #
    # PYTHONPATH=${PWD}:$PYTHONPATH python3 policies/enjoy.py ...
    #
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "")

    if existing_pythonpath:
        env["PYTHONPATH"] = f"{project_root}:{existing_pythonpath}"
    else:
        env["PYTHONPATH"] = str(project_root)

    # Each worker handles one complete training run:
    # duty 0 -> 5 -> 10 -> ... -> 50.
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:

        futures = [
            executor.submit(
                evaluate_run,
                variant_file,
                project_root,
                output_dir,
                args.eval_seed,
                args.overwrite,
                args.dry_run,
                env,
            )
            for variant_file in variant_files
        ]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                safe_print(f"[ERROR] Worker failed: {e}")

    print()
    print("Evaluation sweep finished.")


if __name__ == "__main__":
    main()
