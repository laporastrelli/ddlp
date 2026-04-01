"""
Visualize nominal vs recentered latent-alignment metrics for:
- linear probe: uniform_forward
- nonlinear probe: p_only.train, p_s.train, p_s_d_t.train

Generates 3 bar charts (+/- std error bars) and 3 CSV summaries for:
- mean_pearson_pos_mean
- r2_mean
- rmse_mean
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Avoid matplotlib config/cache permission warnings in shared environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    source_kind: str  # "linear" or "nonlinear"
    linear_transform: str | None = None
    input_type: str | None = None
    nonlinear_split: str | None = None

    def selector_for_dataset(self, dataset: str) -> str:
        if self.source_kind == "linear":
            return f"{dataset}.{self.linear_transform}"
        return f"{dataset}.{self.input_type}.{self.nonlinear_split}"


@dataclass(frozen=True)
class MetricSpec:
    mean_key: str
    std_key: str
    slug: str
    title: str
    ylabel: str


METHOD_SPECS: list[MethodSpec] = [
    MethodSpec(
        key="linear_uniform_forward",
        label="linear\nuniform_forward",
        source_kind="linear",
        linear_transform="uniform_forward",
    ),
    MethodSpec(
        key="nonlinear_p_only_train",
        label="nonlinear\np_only.train",
        source_kind="nonlinear",
        input_type="p_only",
        nonlinear_split="train",
    ),
    MethodSpec(
        key="nonlinear_p_s_train",
        label="nonlinear\np_s.train",
        source_kind="nonlinear",
        input_type="p_s",
        nonlinear_split="train",
    ),
    MethodSpec(
        key="nonlinear_p_s_d_t_train",
        label="nonlinear\np_s_d_t.train",
        source_kind="nonlinear",
        input_type="p_s_d_t",
        nonlinear_split="train",
    ),
]

METRICS: list[MetricSpec] = [
    MetricSpec(
        mean_key="mean_pearson_pos_mean",
        std_key="mean_pearson_pos_std",
        slug="mean_pearson_pos",
        title="Latent Alignment: Mean Pearson Position",
        ylabel="mean_pearson_pos_mean (+/- std)",
    ),
    MetricSpec(
        mean_key="r2_mean",
        std_key="r2_std",
        slug="r2",
        title="Latent Alignment: R2",
        ylabel="r2_mean (+/- std)",
    ),
    MetricSpec(
        mean_key="rmse_mean",
        std_key="rmse_std",
        slug="rmse",
        title="Latent Alignment: RMSE",
        ylabel="rmse_mean (+/- std)",
    ),
]

DATASETS = ["nominal", "recentered"]
COLORS = {
    "nominal": "#4C72B0",
    "recentered": "#DD8452",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot nominal vs recentered latent-alignment metrics for linear/nonlinear probes."
        )
    )
    p.add_argument(
        "--linear-json",
        type=Path,
        default=Path(
            "/data2/users/lr4617/ddlp/outputs/290126_164237_twobody_ddlp_minimal_off_cnt/"
            "extraction_evaluation/best/latent_alignment_eval_comparison/"
            "latent_alignment_metrics_latent_valid_variant_both_source_patch_alpha_nms_nominal_"
            "reorder_smallest_consecutive_distance_hungarian_on.json"
        ),
        help="Linear probe comparison JSON path.",
    )
    p.add_argument(
        "--nonlinear-json",
        type=Path,
        default=Path(
            "/data2/users/lr4617/ddlp/outputs/290126_164237_twobody_ddlp_minimal_off_cnt/"
            "extraction_evaluation/best/latent_alignment_eval_comparison_nonlinear_no_features/"
            "latent_alignment_metrics_nonlinear_latent_train_valid_variant_both_source_patch_alpha_"
            "nms_nominal_reorder_smallest_consecutive_distance_hungarian_on.json"
        ),
        help="Nonlinear probe comparison JSON path.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/data2/users/lr4617/ddlp/outputs/290126_164237_twobody_ddlp_minimal_off_cnt/"
            "extraction_evaluation/best/results"
        ),
        help="Output directory for figures and CSV summaries.",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        default="latent_alignment_nominal_vs_recentered",
        help="Prefix for output filenames.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_metric_block(
    method: MethodSpec,
    dataset: str,
    linear_payload: dict[str, Any],
    nonlinear_payload: dict[str, Any],
) -> dict[str, Any] | None:
    if method.source_kind == "linear":
        node = linear_payload.get(dataset, {})
        if not isinstance(node, dict):
            return None
        block = node.get(method.linear_transform or "", None)
        return block if isinstance(block, dict) else None

    node = nonlinear_payload.get(dataset, {})
    if not isinstance(node, dict):
        return None
    input_node = node.get(method.input_type or "", None)
    if not isinstance(input_node, dict):
        return None
    block = input_node.get(method.nonlinear_split or "", None)
    return block if isinstance(block, dict) else None


def collect_metric_rows(
    method_specs: list[MethodSpec],
    metric: MetricSpec,
    linear_payload: dict[str, Any],
    nonlinear_payload: dict[str, Any],
    linear_json_path: Path,
    nonlinear_json_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in method_specs:
        for dataset in DATASETS:
            block = get_metric_block(
                method=method,
                dataset=dataset,
                linear_payload=linear_payload,
                nonlinear_payload=nonlinear_payload,
            )
            mean_val = maybe_float(block.get(metric.mean_key)) if block else None
            std_val = maybe_float(block.get(metric.std_key)) if block else None
            available = (mean_val is not None) and (std_val is not None)
            source_file = (
                linear_json_path if method.source_kind == "linear" else nonlinear_json_path
            )
            rows.append(
                {
                    "method_key": method.key,
                    "method_label": method.label.replace("\n", " "),
                    "source_kind": method.source_kind,
                    "selector": method.selector_for_dataset(dataset),
                    "dataset": dataset,
                    "available": available,
                    metric.mean_key: mean_val,
                    metric.std_key: std_val,
                    "source_file": str(source_file),
                }
            )
    return rows


def write_metric_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_key",
        "method_label",
        "source_kind",
        "selector",
        "dataset",
        "available",
        "source_file",
    ]

    # Add metric columns in stable order based on first row extras.
    if rows:
        extra = [k for k in rows[0].keys() if k not in fieldnames]
        for k in extra:
            if k not in fieldnames:
                fieldnames.append(k)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_metric_bars(
    metric: MetricSpec,
    rows: list[dict[str, Any]],
    output_path: Path,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    row_map = {(r["method_key"], r["dataset"]): r for r in rows}
    group_gap = 1.6
    bar_width = 0.34
    centers = [i * group_gap for i in range(len(METHOD_SPECS))]

    fig, ax = plt.subplots(figsize=(13, 6.5))

    low_candidates: list[float] = []
    high_candidates: list[float] = []

    for idx, method in enumerate(METHOD_SPECS):
        center = centers[idx]
        offsets = {"nominal": -bar_width / 2.0, "recentered": +bar_width / 2.0}
        for dataset in DATASETS:
            row = row_map.get((method.key, dataset), None)
            x = center + offsets[dataset]
            if row is None or not row["available"]:
                ax.text(
                    x,
                    0.0,
                    "N/A",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#7A7A7A",
                )
                continue

            mean_val = float(row[metric.mean_key])
            std_val = float(row[metric.std_key])
            low_candidates.append(mean_val - std_val)
            high_candidates.append(mean_val + std_val)

            ax.bar(
                x,
                mean_val,
                width=bar_width,
                color=COLORS[dataset],
                edgecolor="black",
                linewidth=0.9,
            )
            ax.errorbar(
                x,
                mean_val,
                yerr=std_val,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=4,
                capthick=1.0,
            )

    if low_candidates and high_candidates:
        lower = min(low_candidates)
        upper = max(high_candidates)
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
        if upper <= lower:
            delta = 0.5 if upper == 0 else abs(upper) * 0.2
            lower -= delta
            upper += delta
        pad = 0.12 * (upper - lower)
        y_min = lower - pad
        y_max = upper + pad
    else:
        y_min, y_max = -1.0, 1.0

    ax.set_ylim(y_min, y_max)

    na_level = y_min + 0.06 * (y_max - y_min)
    for txt in ax.texts:
        if txt.get_text() == "N/A":
            txt.set_position((txt.get_position()[0], na_level))

    ax.set_xticks(centers)
    ax.set_xticklabels([m.label for m in METHOD_SPECS], fontsize=10)
    ax.set_ylabel(metric.ylabel)
    ax.set_title(metric.title)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        handles=[
            Patch(facecolor=COLORS["nominal"], edgecolor="black", label="nominal"),
            Patch(facecolor=COLORS["recentered"], edgecolor="black", label="recentered"),
        ],
        loc="upper right",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.linear_json.exists():
        raise FileNotFoundError(f"Linear JSON not found: {args.linear_json}")
    if not args.nonlinear_json.exists():
        raise FileNotFoundError(f"Nonlinear JSON not found: {args.nonlinear_json}")

    linear_payload = load_json(args.linear_json)
    nonlinear_payload = load_json(args.nonlinear_json)

    print("Building latent alignment visualizations for nominal vs recentered.")
    print(f"  linear source:    {args.linear_json}")
    print(f"  nonlinear source: {args.nonlinear_json}")

    output_paths: list[Path] = []
    for metric in METRICS:
        rows = collect_metric_rows(
            method_specs=METHOD_SPECS,
            metric=metric,
            linear_payload=linear_payload,
            nonlinear_payload=nonlinear_payload,
            linear_json_path=args.linear_json,
            nonlinear_json_path=args.nonlinear_json,
        )
        fig_path = args.output_dir / f"{args.output_prefix}_{metric.slug}_bar.png"
        csv_path = args.output_dir / f"{args.output_prefix}_{metric.slug}_summary.csv"

        plot_metric_bars(metric=metric, rows=rows, output_path=fig_path, dpi=args.dpi)
        write_metric_csv(csv_path=csv_path, rows=rows)

        output_paths.extend([fig_path, csv_path])

    print("\nWrote outputs:")
    for p in output_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
