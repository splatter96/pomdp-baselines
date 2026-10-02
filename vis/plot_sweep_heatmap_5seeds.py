#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# Expected filename:
#   RCS_10_T_10_seed_42.txt
FILENAME_RE = re.compile(
    rf"^RCS_(?P<rcs>{FLOAT_RE})_T_(?P<t>{FLOAT_RE})_seed_(?P<seed>-?\d+)\.txt$"
)

RATE_RE = re.compile(
    rf"Crashrate\s+(?P<crash>{FLOAT_RE})\s+"
    rf"Mergerate\s+(?P<merge>{FLOAT_RE})"
)

EGO_SPEED_RE = re.compile(
    rf"Ego speed:\s*(?P<speed>{FLOAT_RE})"
)


METRICS = {
    "crashrate": {
        "title": "Collision Rate",
        "colorbar": "Mean collision rate",
        "fmt": ".3f",
    },
    "mergerate": {
        "title": "Merge Rate",
        "colorbar": "Mean merge rate",
        "fmt": ".3f",
    },
    "ego_speed": {
        "title": "Average Ego Speed",
        "colorbar": "Mean average ego speed",
        "fmt": ".2f",
    },
}


def parse_file(path: Path):
    """Extract RCS, T, seed and evaluation metrics from one run."""
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")

    rcs = float(match.group("rcs"))
    t = float(match.group("t"))
    seed = int(match.group("seed"))

    text = path.read_text(errors="replace")

    # Crashrate and Mergerate are printed repeatedly during evaluation.
    # The final occurrence is the final result of the run.
    rate_matches = list(RATE_RE.finditer(text))
    if not rate_matches:
        raise ValueError(f"No Crashrate/Mergerate found in {path}")

    final_rate = rate_matches[-1]
    crashrate = float(final_rate.group("crash"))
    mergerate = float(final_rate.group("merge"))

    speed_matches = list(EGO_SPEED_RE.finditer(text))
    if not speed_matches:
        raise ValueError(f"No 'Ego speed:' value found in {path}")

    ego_speed = float(speed_matches[-1].group("speed"))

    return {
        "rcs": rcs,
        "t": t,
        "seed": seed,
        "crashrate": crashrate,
        "mergerate": mergerate,
        "ego_speed": ego_speed,
    }


def format_tick(value):
    return f"{value:g}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the mean and standard deviation across seeds for an "
            "RCS/T evaluation sweep."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("sweep_results"),
        help=(
            "Directory containing RCS_<value>_T_<value>_seed_<seed>.txt files "
            "(default: sweep_results)"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=METRICS.keys(),
        default="crashrate",
        help="Metric to plot (default: crashrate)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: heatmap_<metric>.png",
    )
    parser.add_argument(
        "--ascending-y",
        action="store_true",
        help=(
            "Show T increasing from top to bottom. "
            "By default the largest T is at the top."
        ),
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional fixed lower color scale limit.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional fixed upper color scale limit.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=5,
        help=(
            "Expected number of seeds per parameter configuration. "
            "Used only for consistency warnings (default: 5)."
        ),
    )
    parser.add_argument(
        "--population-std",
        action="store_true",
        help=(
            "Use population standard deviation (ddof=0). "
            "By default, sample standard deviation (ddof=1) is used."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output resolution (default: 300 dpi).",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("RCS_*_T_*_seed_*.txt"))
    if not files:
        raise FileNotFoundError(
            "No files matching RCS_*_T_*_seed_*.txt found in "
            f"{args.input_dir}"
        )

    results = []
    seen_runs = set()

    for path in files:
        try:
            result = parse_file(path)

            run_key = (result["rcs"], result["t"], result["seed"])
            if run_key in seen_runs:
                print(
                    "Warning: duplicate run for "
                    f"RCS={result['rcs']:g}, T={result['t']:g}, "
                    f"seed={result['seed']}. Using all matching files."
                )
            seen_runs.add(run_key)

            results.append(result)
            print(
                f"{path.name}: "
                f"RCS={result['rcs']:g}, T={result['t']:g}, "
                f"seed={result['seed']}, "
                f"crashrate={result['crashrate']:.6f}, "
                f"mergerate={result['mergerate']:.6f}, "
                f"ego_speed={result['ego_speed']:.6f}"
            )
        except ValueError as exc:
            print(f"Warning: {exc}")

    if not results:
        raise RuntimeError("No valid evaluation result files could be parsed.")

    rcs_values = sorted({result["rcs"] for result in results})
    t_values = sorted(
        {result["t"] for result in results},
        reverse=not args.ascending_y,
    )

    rcs_index = {value: i for i, value in enumerate(rcs_values)}
    t_index = {value: i for i, value in enumerate(t_values)}

    # Collect all seed values for each (RCS, T) configuration.
    grouped_values = defaultdict(list)
    grouped_seeds = defaultdict(list)

    for result in results:
        key = (result["rcs"], result["t"])
        grouped_values[key].append(result[args.metric])
        grouped_seeds[key].append(result["seed"])

    mean_data = np.full((len(t_values), len(rcs_values)), np.nan)
    std_data = np.full((len(t_values), len(rcs_values)), np.nan)
    count_data = np.zeros((len(t_values), len(rcs_values)), dtype=int)

    ddof = 0 if args.population_std else 1

    for (rcs, t), values in grouped_values.items():
        values = np.asarray(values, dtype=float)

        row = t_index[t]
        col = rcs_index[rcs]

        mean_data[row, col] = np.mean(values)
        count_data[row, col] = len(values)

        if len(values) > ddof:
            std_data[row, col] = np.std(values, ddof=ddof)

        seeds = grouped_seeds[(rcs, t)]
        if len(values) != args.expected_seeds:
            print(
                "Warning: "
                f"RCS={rcs:g}, T={t:g} has {len(values)} valid run(s), "
                f"expected {args.expected_seeds}. Seeds found: {sorted(seeds)}"
            )

    metric_cfg = METRICS[args.metric]

    # Scale the figure slightly with the number of sweep points.
    fig_width = max(7.0, 1.35 * len(rcs_values) + 2.0)
    fig_height = max(6.0, 1.15 * len(t_values) + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    finite_values = mean_data[np.isfinite(mean_data)]
    if finite_values.size == 0:
        raise RuntimeError(f"No values available for metric '{args.metric}'.")

    vmin = args.vmin if args.vmin is not None else float(np.min(finite_values))
    vmax = args.vmax if args.vmax is not None else float(np.max(finite_values))

    # Avoid a degenerate color scale if all cells have the same value.
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-6)
        vmin -= pad
        vmax += pad

    # Cell color represents the mean over seeds.
    image = ax.imshow(
        mean_data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xticks(np.arange(len(rcs_values)))
    ax.set_xticklabels([format_tick(v) for v in rcs_values])
    ax.set_yticks(np.arange(len(t_values)))
    ax.set_yticklabels([format_tick(v) for v in t_values])

    # Preserve the naming/style conventions from the supplied plotting script.
    ax.set_xlabel(r"$\sigma_c$ [dBm]", fontsize=15)
    ax.set_ylabel(r"$T_{dB}$ [dB]", fontsize=15)
    ax.set_title(metric_cfg["title"], fontsize=20, pad=12)

    ax.tick_params(axis="both", labelsize=13, length=0)

    # White grid lines between cells.
    ax.set_xticks(np.arange(-0.5, len(rcs_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(t_values), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell as:
    #
    #   mean
    #   ± std
    #
    # The standard deviation is calculated across the evaluation seeds.
    norm = image.norm
    for row in range(mean_data.shape[0]):
        for col in range(mean_data.shape[1]):
            mean_value = mean_data[row, col]
            std_value = std_data[row, col]

            if np.isnan(mean_value):
                label = "N/A"
                text_color = "black"
            else:
                mean_label = format(mean_value, metric_cfg["fmt"])

                if np.isnan(std_value):
                    std_label = "N/A"
                else:
                    std_label = format(std_value, metric_cfg["fmt"])

                label = f"{mean_label}\n± {std_label}"

                rgba = cmap(norm(mean_value))
                # Relative luminance for readable black/white annotation text.
                luminance = (
                    0.2126 * rgba[0]
                    + 0.7152 * rgba[1]
                    + 0.0722 * rgba[2]
                )
                text_color = "black" if luminance > 0.55 else "white"

            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=13,
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, pad=0.05)
    colorbar.set_label(metric_cfg["colorbar"], fontsize=15)
    colorbar.ax.tick_params(labelsize=12)

    output = args.output
    if output is None:
        output = Path(f"heatmap_{args.metric}.png")

    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")

    print(f"Saved {output}")


if __name__ == "__main__":
    main()
