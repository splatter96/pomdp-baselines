#!/usr/bin/env python3
"""Plot binary radar-interference signals and summarize interference statistics."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t


RADAR_COLUMNS = [f"radar_{i}" for i in range(4)]
RADAR_NAMES = {
                "radar_0": "Front Right",
                "radar_1": "Rear Right",
                "radar_2": "Rear Left",
                "radar_3": "Front Left",
               }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot radar_0 ... radar_3 as four subplots in a single figure and "
            "print statistics for all rows in the CSV."
        )
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the radar-interference CSV file.",
    )
    parser.add_argument(
        "--max-row",
        type=int,
        default=None,
        help=(
            "Last zero-based CSV data row to include in the plots (inclusive). "
            "If omitted, the entire file is plotted. This does not affect statistics."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output image path, e.g. radar_interference.png. "
            "If omitted, the figure is shown interactively."
        ),
    )
    return parser.parse_args()


def one_run_lengths(values: np.ndarray) -> np.ndarray:
    """Return lengths of all contiguous runs where values == 1."""
    binary = np.asarray(values) == 1
    if binary.size == 0:
        return np.array([], dtype=int)

    padded = np.concatenate(([False], binary, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    return ends - starts


def run_lengths_with_boundaries(df: pd.DataFrame, radar_col: str) -> np.ndarray:
    """
    Calculate 1-run lengths without allowing runs to cross independent sequence
    boundaries. If task_idx / episode_idx exist, they define those boundaries.
    """
    group_cols = [c for c in ("task_idx", "episode_idx") if c in df.columns]

    if not group_cols:
        return one_run_lengths(df[radar_col].to_numpy())

    runs = []
    # sort=False preserves the ordering in the input file.
    for _, group in df.groupby(group_cols, sort=False, dropna=False):
        group_runs = one_run_lengths(group[radar_col].to_numpy())
        if group_runs.size:
            runs.append(group_runs)

    if not runs:
        return np.array([], dtype=int)
    return np.concatenate(runs)


def print_statistics(df: pd.DataFrame) -> None:
    print("\nRadar interference statistics (computed over the entire file)")
    print("-" * 96)
    print(
        f"{'Radar':<10} {'Fraction of 1s':>15} {'Percent 1s':>12} "
        f"{'Mean 1-run':>14} {'95% CI mean 1-run':>24} {'# runs':>9}"
    )
    print("-" * 96)

    for radar_col in RADAR_COLUMNS:
        values = df[radar_col].to_numpy()
        fraction_ones = np.mean(values == 1) if values.size else np.nan
        runs = run_lengths_with_boundaries(df, radar_col)

        if runs.size:
            mean_run = float(np.mean(runs))

            if runs.size > 1:
                sem_run = float(np.std(runs, ddof=1) / np.sqrt(runs.size))
                t_crit = float(t.ppf(0.975, df=runs.size - 1))
                ci_low = mean_run - t_crit * sem_run
                ci_high = mean_run + t_crit * sem_run
            else:
                # A confidence interval cannot be estimated from a single run.
                ci_low = np.nan
                ci_high = np.nan
        else:
            mean_run = np.nan
            ci_low = np.nan
            ci_high = np.nan

        print(
            f"{radar_col:<10} "
            f"{fraction_ones:>15.6f} "
            f"{100.0 * fraction_ones:>11.2f}% "
            f"{mean_run:>14.3f} "
            f"{f'[{ci_low:.3f}, {ci_high:.3f}]':>24} "
            f"{runs.size:>9d}"
        )

    print("-" * 96)
    print("Run lengths are measured in consecutive CSV rows / simulation steps.")
    print("Runs are not allowed to continue across task_idx/episode_idx boundaries.")
    print("The 95% CI is a Student-t confidence interval for the mean 1-run length.\n")


def plot_radars(df: pd.DataFrame, max_row: int, output: Path) -> None:
    if max_row is not None:
        if max_row < 0:
            raise ValueError("--max-row must be >= 0")
        plot_df = df.iloc[: max_row + 1].copy()
    else:
        plot_df = df.copy()

    # Continuous sample order is preferable to `step`, because `step` resets
    # for each task in this file.
    plot_df["row"] = np.arange(len(plot_df))

    sns.set_theme()
    sns.set_context("paper")
    sns.set(font_scale=1.2)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(10, 10),
        sharex=True,
        constrained_layout=True,
    )

    for ax, radar_col in zip(axes, RADAR_COLUMNS):
        sns.lineplot(
            data=plot_df,
            x="row",
            y=radar_col,
            # drawstyle="steps-post",
            color="steelblue",
            linewidth=1.5,
            estimator=None,
            ax=ax,
        )
        ax.set_title(f"{RADAR_NAMES[radar_col]} Radar")
        ax.set_ylabel("Interference")
        ax.set_yticks([0, 1], ["no", "yes"])
        ax.set_ylim(-0.1, 1.1)
        ax.margins(x=0)

    axes[-1].set_xlabel("Frame")
    #fig.suptitle("Radar interference signals", fontsize=14)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        print(f"Saved figure to: {output}")
    else:
        plt.show()



def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv_file)

    missing = [col for col in RADAR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required radar columns: " + ", ".join(missing)
        )

    print_statistics(df)
    plot_radars(df, args.max_row, args.output)


if __name__ == "__main__":
    main()
