#!/usr/bin/env python3
"""Template-based statistical plot generator for PaperBanana.

Generates publication-ready matplotlib/seaborn plots from JSON configurations.
Supports: bar, line, scatter, box, violin, heatmap, confusion_matrix,
histogram, radar.

Usage:
    python plot_generator.py --config config.json --output figure.pdf
    python plot_generator.py --type bar --data '{"A": 92, "B": 88}' --output fig.png

Requirements:
    pip install matplotlib seaborn numpy
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
STYLES_DIR = SKILL_DIR / "assets" / "matplotlib_styles"
PALETTES_DIR = SKILL_DIR / "assets" / "palettes"

DEFAULT_STYLE = STYLES_DIR / "academic_default.mplstyle"
DEFAULT_PALETTE = PALETTES_DIR / "colorblind_safe.json"


def load_palette(palette_path: Path = DEFAULT_PALETTE) -> list:
    """Load a color palette from a JSON file."""
    if palette_path.exists():
        with open(palette_path, "r") as f:
            data = json.load(f)
        return data.get("colors", data.get("categorical", []))
    # Fallback: Okabe-Ito
    return ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377", "#66CCEE", "#BBBBBB"]


def apply_style(style_path: Path = DEFAULT_STYLE):
    """Apply matplotlib style if available."""
    if style_path.exists():
        plt.style.use(str(style_path))
    else:
        # Fallback defaults
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        })


def plot_bar(config: dict, ax: plt.Axes, colors: list):
    """Generate a bar chart."""
    data = config["data"]
    labels = list(data.keys()) if isinstance(data, dict) else config.get("labels", [])
    values = list(data.values()) if isinstance(data, dict) else data

    # Handle grouped bars
    if isinstance(values[0], (list, dict)):
        # Grouped bar chart
        series_names = config.get("series_names", [])
        n_groups = len(labels)
        n_series = len(values[0]) if isinstance(values[0], list) else len(values[0])
        bar_width = 0.8 / n_series
        x = np.arange(n_groups)

        for i in range(n_series):
            if isinstance(values[0], dict):
                series_vals = [v[list(v.keys())[i]] for v in values]
                name = list(values[0].keys())[i]
            else:
                series_vals = [v[i] for v in values]
                name = series_names[i] if i < len(series_names) else f"Series {i+1}"

            bars = ax.bar(x + i * bar_width, series_vals, bar_width,
                         label=name, color=colors[i % len(colors)],
                         edgecolor="gray", linewidth=0.5)

            # Value labels for small datasets
            if n_groups <= 10:
                for bar, val in zip(bars, series_vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           f"{val:.1f}" if isinstance(val, float) else str(val),
                           ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x + bar_width * (n_series - 1) / 2)
        ax.set_xticklabels(labels)
        ax.legend()
    else:
        # Simple bar chart
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=[colors[i % len(colors)] for i in range(len(values))],
                     edgecolor="gray", linewidth=0.5)

        # Value labels for small datasets
        if len(labels) <= 10:
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       f"{val:.1f}" if isinstance(val, float) else str(val),
                       ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)

    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    if config.get("title"):
        ax.set_title(config["title"])

    # Horizontal grid for bar charts
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def plot_line(config: dict, ax: plt.Axes, colors: list):
    """Generate a line chart."""
    markers = ["o", "s", "^", "D", "v", "p", "*", "h"]
    linestyles = ["-", "--", "-.", ":"]
    series = config.get("series", {})

    if not series and "data" in config:
        series = {"Data": config["data"]}

    for i, (name, points) in enumerate(series.items()):
        if isinstance(points, dict):
            x = list(points.keys())
            y = list(points.values())
        elif isinstance(points, list) and points and isinstance(points[0], (list, tuple)):
            x = [p[0] for p in points]
            y = [p[1] for p in points]
        else:
            x = list(range(len(points)))
            y = points

        ax.plot(x, y,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)] if i > 0 else "-",
                linewidth=1.8,
                markersize=6,
                label=name)

    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    if config.get("title"):
        ax.set_title(config["title"])
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def plot_scatter(config: dict, ax: plt.Axes, colors: list):
    """Generate a scatter plot."""
    series = config.get("series", {})

    if not series and "data" in config:
        series = {"Data": config["data"]}

    for i, (name, points) in enumerate(series.items()):
        if isinstance(points, list) and points and isinstance(points[0], (list, tuple)):
            x = [p[0] for p in points]
            y = [p[1] for p in points]
        elif isinstance(points, dict):
            x = list(points.keys())
            y = list(points.values())
        else:
            continue

        sizes = config.get("sizes", 40)
        alpha = config.get("alpha", 0.7)
        ax.scatter(x, y, c=colors[i % len(colors)], s=sizes, alpha=alpha,
                  label=name, edgecolors="gray", linewidth=0.5)

    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    if config.get("title"):
        ax.set_title(config["title"])
    if len(series) > 1:
        ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)


def plot_heatmap(config: dict, ax: plt.Axes, colors: list):
    """Generate a heatmap."""
    data = np.array(config["data"])
    xlabels = config.get("xlabels", [str(i) for i in range(data.shape[1])])
    ylabels = config.get("ylabels", [str(i) for i in range(data.shape[0])])
    cmap = config.get("cmap", "viridis")
    annot = config.get("annotate", True)
    fmt = config.get("fmt", ".2f")

    if HAS_SEABORN:
        sns.heatmap(data, ax=ax, annot=annot, fmt=fmt, cmap=cmap,
                   xticklabels=xlabels, yticklabels=ylabels,
                   linewidths=0.5, linecolor="white")
    else:
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels)
        plt.colorbar(im, ax=ax)

        if annot:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, f"{data[i, j]:{fmt[1:]}}",
                           ha="center", va="center",
                           color="white" if data[i, j] > data.mean() else "black",
                           fontsize=7)

    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    if config.get("title"):
        ax.set_title(config["title"])


def plot_box(config: dict, ax: plt.Axes, colors: list):
    """Generate a box plot."""
    data = config["data"]
    labels = config.get("labels", [f"Group {i+1}" for i in range(len(data))])

    if HAS_SEABORN:
        bp = sns.boxplot(data=data, ax=ax, palette=colors[:len(data)])
    else:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)

    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    if config.get("title"):
        ax.set_title(config["title"])


def plot_violin(config: dict, ax: plt.Axes, colors: list):
    """Generate a violin plot."""
    data = config["data"]
    labels = config.get("labels", [f"Group {i+1}" for i in range(len(data))])

    if HAS_SEABORN:
        sns.violinplot(data=data, ax=ax, palette=colors[:len(data)], inner="quartile")
    else:
        parts = ax.violinplot(data, showmedians=True, showquartiles=True)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1) if not HAS_SEABORN else range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    if config.get("title"):
        ax.set_title(config["title"])


def plot_histogram(config: dict, ax: plt.Axes, colors: list):
    """Generate a histogram."""
    data = config["data"]
    bins = config.get("bins", "auto")

    if isinstance(data, dict):
        for i, (name, values) in enumerate(data.items()):
            ax.hist(values, bins=bins, alpha=0.7, label=name,
                   color=colors[i % len(colors)], edgecolor="white", linewidth=0.5)
        ax.legend(frameon=False)
    else:
        ax.hist(data, bins=bins, color=colors[0], edgecolor="white", linewidth=0.5)

    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", "Frequency"))
    if config.get("title"):
        ax.set_title(config["title"])
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")
    ax.set_axisbelow(True)


def plot_radar(config: dict, ax_placeholder: plt.Axes, colors: list):
    """Generate a radar/spider chart. Requires polar projection."""
    data = config["data"]
    categories = config.get("categories", [])

    # Close the figure from placeholder and create polar
    fig = ax_placeholder.get_figure()
    ax_placeholder.remove()
    ax = fig.add_subplot(111, polar=True)

    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    for i, (name, values) in enumerate(data.items()):
        vals = list(values) + [values[0]]  # Close
        ax.plot(angles, vals, "o-", linewidth=1.5, label=name,
                color=colors[i % len(colors)], markersize=4)
        ax.fill(angles, vals, alpha=0.2, color=colors[i % len(colors)])

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False)
    if config.get("title"):
        ax.set_title(config["title"], pad=20)


PLOT_TYPES = {
    "bar": plot_bar,
    "line": plot_line,
    "scatter": plot_scatter,
    "heatmap": plot_heatmap,
    "confusion_matrix": plot_heatmap,
    "box": plot_box,
    "violin": plot_violin,
    "histogram": plot_histogram,
    "radar": plot_radar,
}


def generate_plot(config: dict, output_path: str):
    """Generate a plot from a configuration dictionary.

    Args:
        config: Plot configuration with keys: type, data, and optional
                formatting parameters.
        output_path: Path to save the generated plot.
    """
    apply_style()

    plot_type = config.get("type", "bar")
    if plot_type not in PLOT_TYPES:
        print(f"Error: Unknown plot type '{plot_type}'.")
        print(f"Supported types: {', '.join(PLOT_TYPES.keys())}")
        sys.exit(1)

    # Load palette
    palette_name = config.get("palette", None)
    if palette_name:
        palette_path = PALETTES_DIR / palette_name
        if not palette_path.suffix:
            palette_path = palette_path.with_suffix(".json")
    else:
        palette_path = DEFAULT_PALETTE
    colors = load_palette(palette_path)

    # Figure size
    figsize = config.get("figsize", (6.875, 4.5))
    if isinstance(figsize, list):
        figsize = tuple(figsize)

    fig, ax = plt.subplots(figsize=figsize)

    # Generate the plot
    plot_fn = PLOT_TYPES[plot_type]
    plot_fn(config, ax, colors)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready statistical plots"
    )
    parser.add_argument("--config", type=str, help="JSON config file path")
    parser.add_argument("--type", type=str, choices=list(PLOT_TYPES.keys()),
                        help="Plot type (used with --data)")
    parser.add_argument("--data", type=str,
                        help="JSON data string (used with --type)")
    parser.add_argument("--output", type=str, default="output/figure.pdf",
                        help="Output file path (default: output/figure.pdf)")

    args = parser.parse_args()

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        with open(config_path, "r") as f:
            config = json.load(f)
    elif args.type and args.data:
        parsed = json.loads(args.data)
        if isinstance(parsed, dict) and any(k != "data" for k in parsed):
            # Merge top-level keys (series, xlabel, etc.) into config
            config = {"type": args.type, **parsed}
        else:
            config = {"type": args.type, "data": parsed}
    else:
        parser.error("Either --config or both --type and --data are required")
        return

    generate_plot(config, args.output)


if __name__ == "__main__":
    main()
