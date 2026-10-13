#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

sns.set_theme()
sns.set_context("paper")
sns.set(font_scale=1.2)

# Expected filename:
#
# trained_on_10_train_seed21_eval_duty_0.txt
#
FILENAME_RE = re.compile(
    r"trained_on_(?P<train_duty>-?\d+(?:\.\d+)?)"
    r"_train_seed_(?P<train_seed>\d+)"
    r"_eval_duty_(?P<eval_duty>-?\d+(?:\.\d+)?)"
    r"\.txt$"
)


METRIC_PATTERNS = {
    "total_merges": re.compile(r"Total merges:\s*(\d+)"),
    "total_crashes": re.compile(r"Total crashes:\s*(\d+)"),
    "total_episodes": re.compile(r"Total episodes:\s*(\d+)"),
    "avg_speed": re.compile(
        r"Ego speed:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    ),
}


def parse_number(value):
    """Convert duty cycles to int if possible, otherwise float."""
    value = float(value)
    if value.is_integer():
        return int(value)
    return value


def parse_result_file(path: Path):
    """
    Parse metadata from filename and metrics from file.
    """

    match = FILENAME_RE.match(path.name)

    if match is None:
        return None

    train_duty = parse_number(match.group("train_duty"))
    train_seed = int(match.group("train_seed"))
    eval_duty = parse_number(match.group("eval_duty"))

    # Reading the entire file is fine, but rsplit keeps the regex work
    # focused on the end of what may be a fairly large tqdm log.
    text = path.read_text(errors="replace")

    values = {}

    for name, pattern in METRIC_PATTERNS.items():
        matches = pattern.findall(text)

        if not matches:
            raise ValueError(
                f"Could not find metric '{name}' in {path}"
            )

        # Use the last occurrence, just in case something similar was
        # printed previously.
        values[name] = matches[-1]

    total_merges = int(values["total_merges"])
    total_crashes = int(values["total_crashes"])
    total_episodes = int(values["total_episodes"])
    avg_speed = float(values["avg_speed"])

    if total_episodes <= 0:
        raise ValueError(
            f"Invalid number of episodes in {path}: {total_episodes}"
        )

    return {
        "file": str(path),
        "train_duty": train_duty,
        "train_seed": train_seed,
        "eval_duty": eval_duty,
        "total_merges": total_merges,
        "total_crashes": total_crashes,
        "total_episodes": total_episodes,
        "collision_rate": total_crashes / total_episodes,
        "merge_rate": total_merges / total_episodes,
        "avg_speed": avg_speed,
    }


def confidence_interval(values, confidence=0.95):
    """
    Calculate mean and Student-t confidence interval.

    The samples here are the different training seeds.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    mean = np.mean(values)

    # Cannot estimate a CI from one seed.
    if n == 1:
        return mean, np.nan, np.nan

    sem = stats.sem(values)

    if sem == 0:
        return mean, mean, mean

    alpha = 1.0 - confidence
    critical = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    half_width = critical * sem

    return mean, mean - half_width, mean + half_width


def aggregate_metric(df, metric, confidence):
    rows = []

    groups = df.groupby(["train_duty", "eval_duty"])

    for (train_duty, eval_duty), group in groups:
        mean, lower, upper = confidence_interval(
            group[metric].values,
            confidence=confidence,
        )

        rows.append(
            {
                "train_duty": train_duty,
                "eval_duty": eval_duty,
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_seeds": len(group),
            }
        )

    return pd.DataFrame(rows)


def duty_label(duty):
    if duty == 0:
        return "No Interference"

    if float(duty).is_integer():
        return str(int(duty))

    return str(duty)


def plot_metric(
    df,
    metric,
    ylabel,
    output_file,
    confidence=0.95,
):
    aggregated = aggregate_metric(
        df,
        metric,
        confidence,
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))

    trained_duties = sorted(aggregated["train_duty"].unique())

    for train_duty in trained_duties:
        data = aggregated[
            aggregated["train_duty"] == train_duty
        ].sort_values("eval_duty")

        x = data["eval_duty"].to_numpy()
        mean = data["mean"].to_numpy()
        lower = data["ci_lower"].to_numpy()
        upper = data["ci_upper"].to_numpy()

        # plot() returns the chosen color so the confidence band
        # automatically gets exactly the same color.
        line, = ax.plot(
            x,
            mean,
            linewidth=2,
            marker="o",
            markersize=4,
            label=duty_label(train_duty),
        )

        # Only shade points for which a CI could actually be estimated.
        valid = np.isfinite(lower) & np.isfinite(upper)

        if np.any(valid):
            ax.fill_between(
                x,
                lower,
                upper,
                where=valid,
                color=line.get_color(),
                alpha=0.15,
                interpolate=True,
            )

    ax.set_xlabel("Evaluated Duty Cycle [%]")
    ax.set_ylabel(ylabel)

    ax.legend(
        title="Trained on duty cycle [%]",
        frameon=False,
    )

    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot collision rate, merge rate and average speed "
            "over evaluated radar duty cycles."
        )
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing evaluation .txt files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_plots"),
        help="Directory in which plots are saved.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level. Default: 0.95.",
    )

    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output figure format. Default: png.",
    )

    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []

    # Recursive search allows the files to be arranged in subdirectories
    # if desired.
    result_files = sorted(
        results_dir.rglob("trained_on_*_train_seed*_eval_duty_*.txt")
    )

    if not result_files:
        raise RuntimeError(
            f"No evaluation output files found in {results_dir}"
        )

    print(f"Found {len(result_files)} candidate files.")

    for path in result_files:
        try:
            result = parse_result_file(path)

            if result is None:
                skipped.append((path, "filename did not match"))
                continue

            rows.append(result)

        except Exception as e:
            # This is particularly useful while evaluations are still
            # running: unfinished output files simply get skipped.
            skipped.append((path, str(e)))

    if not rows:
        raise RuntimeError("No completed evaluation files could be parsed.")

    df = pd.DataFrame(rows)

    df = df.sort_values(
        ["train_duty", "train_seed", "eval_duty"]
    ).reset_index(drop=True)

    print()
    print(f"Successfully parsed {len(df)} evaluations.")

    if skipped:
        print(f"Skipped {len(skipped)} incomplete/invalid files:")
        for path, reason in skipped:
            print(f"  {path.name}: {reason}")

    # Check how many training seeds are available.
    print()
    print("Number of seeds per trained/evaluated duty-cycle combination:")

    seed_counts = (
        df.groupby(["train_duty", "eval_duty"])["train_seed"]
        .nunique()
        .unstack()
    )

    print(seed_counts.to_string())

    # Save raw extracted results. This is useful for checking individual
    # values and later statistical analysis.
    csv_path = output_dir / "evaluation_metrics.csv"
    df.to_csv(csv_path, index=False)

    print()
    print(f"Saved extracted metrics to {csv_path}")

    # Produce the three requested figures.
    plots = [
        (
            "collision_rate",
            "Collision rate",
            f"collision_rate.{args.format}",
        ),
        (
            "merge_rate",
            "Merge rate",
            f"merge_rate.{args.format}",
        ),
        (
            "avg_speed",
            "Average speed [m/s]",
            f"average_speed.{args.format}",
        ),
    ]

    for metric, ylabel, filename in plots:
        output_file = output_dir / filename

        plot_metric(
            df,
            metric,
            ylabel,
            output_file,
            confidence=args.confidence,
        )

        print(f"Saved {output_file}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
