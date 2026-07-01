"""
CLI — Command-line interface for SyncPipe (distribution: syncpipe).

Usage:
    python -m multisync analyze -i neural.csv,bio.csv,behavior.csv \
           -n neural,bio,behavior --hz 1.0 -o results.json

    python -m multisync demo --output demo_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from .__about__ import __version__
from .core import Dyad, DynamicAnalyzer
from .dataset import SynchronyDataset
from .design_controls import design_control_audit, synchrony_existence_audit
from .feature_status import feature_status_latex, feature_status_table
from .io import load_csv
from .synthetic import generate_ground_truth_dyad


def cmd_analyze(args: argparse.Namespace) -> None:
    """Run analysis on user-provided CSV files."""
    input_files = args.input.split(",")
    names = args.names.split(",") if args.names else [
        f"modality_{i}" for i in range(len(input_files))
    ]

    if len(input_files) != len(names):
        print("Error: number of input files must match number of names.", file=sys.stderr)
        sys.exit(1)

    hz = float(args.hz)
    modalities = {}
    for name, path in zip(names, input_files):
        print(f"  Loading {name}: {path}")
        modalities[name] = load_csv(path)

    # Create dyad and run pipeline
    dyad = Dyad(**modalities, hz=hz)

    # Add context labels if provided
    if args.contexts:
        for ctx_str in args.contexts:
            parts = ctx_str.split(",")
            if len(parts) >= 3:
                dyad.add_context(
                    start=float(parts[0]),
                    end=float(parts[1]),
                    label=parts[2],
                    score=float(parts[3]) if len(parts) > 3 else 0.0,
                )

    dyad.align(target_hz=hz)
    dyad.zscore()

    analyzer = DynamicAnalyzer(
        window_size=args.window_size,
        surrogate_n=args.surrogates,
        max_lag_sec=args.max_lag,
        seed=args.seed,
    )

    print("  Running analysis...")
    results = analyzer.fit_transform(dyad)

    output_path = args.output or "results.json"
    results.export_viewer_json(output_path)
    print(f"  Results exported to: {output_path}")

    # Summary
    if results.prediction:
        print("\n  Prediction results:")
        for key, pred in results.prediction.items():
            print(
                f"    {key}: delta-AUC = {pred.get('mean_delta_auc', 0):.3f}, "
                f"dynamic AUC = {pred.get('mean_dynamic_auc', 0.5):.3f}, "
                f"baseline AUC = {pred.get('mean_baseline_auc', 0.5):.3f}"
            )


def _json_ready(obj: Any) -> Any:
    """Convert numpy/pandas objects and NaN values to JSON-safe values."""
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_ready(obj.tolist())
    if isinstance(obj, float):
        return None if not np.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _make_demo_cohort(n_dyads: int, *, hz: float, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Small synthetic cohort used only for design-control demonstration."""
    cohort = {}
    for i in range(n_dyads):
        ds_i = generate_ground_truth_dyad(
            lead_modality="behavior",
            lag_modality="neural",
            true_lag_sec=0.0,
            noise_ratio=0.35,
            duration_sec=240,
            hz=hz,
            seed=seed + i,
            gap_prob=0.0,
            coupling=0.65,
        )
        df = ds_i.modalities["behavior"]
        cohort[f"dyad_{i + 1:02d}"] = (
            df["person_a"].to_numpy(dtype=float),
            df["person_b"].to_numpy(dtype=float),
        )
    return cohort


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a complete synthetic demo with viewer JSON and audit reports."""
    print("  Generating synthetic dyad (single-modality coupling, 30% noise)...")
    ds = generate_ground_truth_dyad(
        lead_modality="behavior",
        lag_modality="neural",
        true_lag_sec=0.0,
        noise_ratio=0.3,
        duration_sec=300,
        hz=1.0,
        seed=42,
    )

    ds.align(target_hz=1.0)
    ds.zscore()

    ds.add_context(start=0, end=150, label="PreTask")
    ds.add_context(start=150, end=300, label="Task")

    analyzer = DynamicAnalyzer(
        window_size=10,
        surrogate_n=args.surrogates,
        max_lag_sec=30.0,
        seed=42,
        enable_prediction=not getattr(args, "no_prediction", False),
    )

    print("  Running analysis...")
    results = analyzer.fit_transform(ds)

    output_arg = args.output or "demo_results.json"
    output_path = Path(output_arg)
    output_is_dir = output_path.suffix.lower() != ".json"
    demo_dir = output_path if output_is_dir else output_path.parent
    demo_dir.mkdir(parents=True, exist_ok=True)
    viewer_path = demo_dir / "viewer_results.json" if output_is_dir else output_path
    results.export_viewer_json(str(viewer_path))
    print(f"  Viewer results exported to: {viewer_path}")

    feature_rows = []
    for pair, feats in results.dynamic_features.items():
        row = {"scope": "global", "label": "all", "pair": pair}
        row.update(feats)
        feature_rows.append(row)
    for label, pairs in results.dynamic_features_segmented.items():
        for pair, feats in pairs.items():
            row = {"scope": "segment", "label": label, "pair": pair}
            row.update(feats)
            feature_rows.append(row)
    feature_table = pd.DataFrame(feature_rows)
    feature_table_path = demo_dir / "feature_table.csv"
    feature_table.to_csv(feature_table_path, index=False)

    feature_status_path = demo_dir / "feature_status_table.csv"
    feature_status_table().to_csv(feature_status_path, index=False)
    feature_status_tex_path = demo_dir / "TABLE1_FEATURE_STATUS.tex"
    feature_status_latex(str(feature_status_tex_path))

    behavior = ds.modalities["behavior"]
    existence = synchrony_existence_audit(
        behavior["person_a"].to_numpy(dtype=float),
        behavior["person_b"].to_numpy(dtype=float),
        hz=1.0,
        window_size=10,
        surrogate_n=getattr(args, "audit_surrogates", 99),
        seed=42,
    )
    existence_path = demo_dir / "synchrony_existence_audit.json"
    _write_json(existence_path, existence)

    cohort = _make_demo_cohort(getattr(args, "demo_dyads", 6), hz=1.0, seed=100)
    design = design_control_audit(
        cohort,
        hz=1.0,
        window_size=10,
        n_pseudo_per_dyad=2,
        shift_lags_sec=(-60.0, -30.0, 30.0, 60.0),
        seed=42,
    )
    design_path = demo_dir / "design_control_audit.json"
    _write_json(design_path, design)

    gt = ds._ground_truth
    report_path = demo_dir / "DEMO_REPORT.md"
    lines = [
        "# SyncPipe demo report",
        "",
        "This demo illustrates SyncPipe as single-modality synchrony measurement infrastructure: WCC trace construction, descriptor export, synchrony-existence audit, design-control audit, and viewer-ready output.",
        "",
        "## Ground truth",
        f"- Synthetic dyadic coupling, noise ratio: {gt['noise_ratio']}.",
        "",
        "## Outputs",
        f"- Viewer JSON: `{viewer_path.name}`",
        f"- Feature table: `{feature_table_path.name}`",
        f"- Feature status table: `{feature_status_path.name}`",
        f"- Table 1 LaTeX: `{feature_status_tex_path.name}`",
        f"- Synchrony-existence audit: `{existence_path.name}`",
        f"- Design-control audit: `{design_path.name}`",
        "",
        "## Synchrony-existence audit",
        "Signal-level IAAFT asks whether the observed WCC exceeds independent autocorrelated signals. It is necessary but not sufficient evidence for interpersonal coupling.",
        "",
        "```json",
        json.dumps(_json_ready(existence.get("p_values", {})), indent=2),
        "```",
        "",
        "## Feature status table",
        "`feature_status_table.csv` is the Table 1 draft: source level, incremental information, paradigm restriction, default audit/test, status, and risk. It separates descriptor usefulness from confirmatory status.",
        "",
        "## Design controls",
        "Pseudo-pair and time-shift controls are design-level audits for dyad-specificity and temporal-alignment dependence. They do not solve all ISC/co-presence problems, but they make those alternatives visible.",
        "",
        "| feature | real median | pseudo median | time-shift median | p(real>pseudo) | p(real>shift) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for feature, summary in design["feature_summary"].items():
        lines.append(
            f"| {feature} | "
            f"{summary['real_median']:.3f} | "
            f"{summary['pseudo_pair_median']:.3f} | "
            f"{summary['time_shift_median']:.3f} | "
            f"{summary['p_real_gt_pseudo']:.4f} | "
            f"{summary['p_real_gt_time_shift']:.4f} |"
        )
    lines.extend([
        "",
        "## Caution",
        "This demo is synthetic. Passing signal-level IAAFT does not prove dyad-specific coupling. For real event-locked or shared-stimulus designs, add pseudo-pair, time-shift, and when possible across-stimulus shuffle controls.",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"  Feature table exported to: {feature_table_path}")
    print(f"  Feature status table exported to: {feature_status_path}")
    print(f"  Table 1 LaTeX exported to: {feature_status_tex_path}")
    print(f"  Audit report exported to: {report_path}")

    n_sig = sum(1 for p in existence.get("p_values", {}).values() if isinstance(p, (int, float)) and p < 0.05)
    print(f"\n  Synchrony-existence audit: {n_sig} feature(s) exceed the IAAFT null (p < 0.05).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multisync",
        description="SyncPipe: Dynamic process analysis for multimodal synchrony.",
    )
    parser.add_argument(
        "--version", action="version", version=f"syncpipe {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze multi-modal dyadic data.")
    p_analyze.add_argument("-i", "--input", required=True, help="Comma-separated CSV paths.")
    p_analyze.add_argument("-n", "--names", help="Comma-separated modality names.")
    p_analyze.add_argument("--hz", default=1.0, help="Target sampling rate.")
    p_analyze.add_argument("-o", "--output", default="results.json", help="Output JSON path.")
    p_analyze.add_argument("--window-size", type=int, default=10, help="WCC window size.")
    p_analyze.add_argument("--surrogates", type=int, default=500, help="Number of surrogates.")
    p_analyze.add_argument("--max-lag", type=float, default=30.0, help="Max CCF lag (sec).")
    p_analyze.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_analyze.add_argument(
        "--contexts", nargs="*", help="Context labels: start,end,label[,score]."
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # demo
    p_demo = sub.add_parser("demo", help="Run complete synthetic demo + audit reports.")
    p_demo.add_argument(
        "-o", "--output", default="demo_results.json",
        help="Output JSON path, or a directory for all demo artifacts.",
    )
    p_demo.add_argument("--surrogates", type=int, default=500, help="Number of CCF surrogates.")
    p_demo.add_argument(
        "--audit-surrogates", type=int, default=99,
        help="Number of signal-level IAAFT surrogates for the existence audit.",
    )
    p_demo.add_argument(
        "--demo-dyads", type=int, default=6,
        help="Synthetic dyads used for pseudo-pair/time-shift design controls.",
    )
    p_demo.add_argument(
        "--no-prediction", action="store_true",
        help="Skip rolling-origin prediction CV for a faster methods demo.",
    )
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
