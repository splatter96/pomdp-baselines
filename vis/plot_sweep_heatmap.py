#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

FILENAME_RE = re.compile(
    rf"^RCS_(?P<rcs>{FLOAT_RE})_T_(?P<t>{FLOAT_RE})\.txt$"
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
        "colorbar": "Collision rate",
        "fmt": ".3f",
    },
    "mergerate": {
        "title": "Merge Rate",
        "colorbar": "Merge rate",
        "fmt": ".3f",
    },
    "ego_speed": {
        "title": "Average Ego Speed",
        "colorbar": "Average ego speed",
        "fmt": ".2f",
    },
}


def parse_file(path: Path):
    """Extract RCS, T, crash rate, merge rate and ego speed from one run."""
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected filename: {path.name}")

    rcs = float(match.group("rcs"))
    t = float(match.group("t"))

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
        "crashrate": crashrate,
        "mergerate": mergerate,
        "ego_speed": ego_speed,
    }


def format_tick(value):
    return f"{value:g}"


def main():
    parser = argparse.ArgumentParser(
        description="Plot an RCS/T evaluation sweep as a heatmap."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=Path("sweep_results"),
        help="Directory containing RCS_<value>_T_<value>.txt files "
             "(default: sweep_results)",
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
        help="Show T increasing from top to bottom. "
             "By default the largest T is at the top, matching the example plot.",
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
        "--dpi",
        type=int,
        default=300,
        help="Output resolution (default: 300 dpi).",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("RCS_*_T_*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No files matching RCS_*_T_*.txt found in {args.input_dir}"
        )

    results = []
    for path in files:
        try:
            result = parse_file(path)
            results.append(result)
            print(
                f"{path.name}: "
                f"RCS={result['rcs']:g}, T={result['t']:g}, "
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

    data = np.full((len(t_values), len(rcs_values)), np.nan)

    for result in results:
        row = t_index[result["t"]]
        col = rcs_index[result["rcs"]]
        data[row, col] = result[args.metric]

    metric_cfg = METRICS[args.metric]

    # Scale the figure slightly with the number of sweep points.
    fig_width = max(7.0, 1.35 * len(rcs_values) + 2.0)
    fig_height = max(6.0, 1.15 * len(t_values) + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")

    finite_values = data[np.isfinite(data)]
    if finite_values.size == 0:
        raise RuntimeError(f"No values available for metric '{args.metric}'.")

    vmin = args.vmin if args.vmin is not None else float(np.min(finite_values))
    vmax = args.vmax if args.vmax is not None else float(np.max(finite_values))

    # Avoid a degenerate color scale if all cells have the same value.
    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1e-6)
        vmin -= pad
        vmax += pad

    image = ax.imshow(
        data,
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

    ax.set_xlabel("$\sigma_c$ [dBm]", fontsize=15)
    ax.set_ylabel("$T_{dB}$ [dB]", fontsize=15)
    ax.set_title(metric_cfg["title"], fontsize=20, pad=12)

    ax.tick_params(axis="both", labelsize=13, length=0)

    # White grid lines between cells, similar to the example heatmap.
    ax.set_xticks(np.arange(-0.5, len(rcs_values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(t_values), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate every heatmap cell.
    norm = image.norm
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]

            if np.isnan(value):
                label = "N/A"
                text_color = "black"
            else:
                label = format(value, metric_cfg["fmt"])
                rgba = cmap(norm(value))
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
                fontsize=14,
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
