"""
CLI — Command-line interface for SyncPipe (distribution: syncpipe).

Usage (``syncpipe`` is the command/import namespace):

    # Canonical v1 audited evidence chain (manifest + config -> report bundle)
    python -m syncpipe analyze -m manifest.csv -c config.toml -o results/

    # Design-agnostic descriptor path on ad-hoc CSVs (exploratory)
    python -m syncpipe describe -i neural.csv,bio.csv -n neural,bio --hz 1.0 -o out.json

    python -m syncpipe demo --output demo_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from .__about__ import __version__
from .batch import _bh_fdr_correction  # verified-equivalent BH-FDR helper (no 4th impl)
from .core import Dyad, DynamicAnalyzer
from .dataset import SynchronyDataset
from .design_controls import design_control_audit, synchrony_existence_audit
from .qc import format_qc_report, run_quality_check
from .feature_status import feature_status_latex, feature_status_table
from .io import load_csv
from .synthetic import generate_ground_truth_dyad


def cmd_describe(args: argparse.Namespace) -> None:
    """Describe one aligned pair without the full study-level checks."""
    input_files = [p.strip() for p in args.input.split(",") if p.strip()]
    names = [n.strip() for n in args.names.split(",") if n.strip()] if args.names else [
        f"modality_{i}" for i in range(len(input_files))
    ]

    if len(input_files) != len(names):
        print("Error: number of input files must match number of names.", file=sys.stderr)
        sys.exit(1)

    hz = args.hz  # already validated as positive float by argparse type
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

    # ponytail: capture align() warnings so they appear as controlled Notes
    # after QC, not as raw Python warnings on stderr before QC says PASS.
    with warnings.catch_warnings(record=True) as _align_warnings:
        warnings.simplefilter("always")
        dyad.align(target_hz=hz)
    dyad.zscore()

    # P1-D strict input contract: every modality must carry at least one
    # numeric signal column, or the result is meaningless. Fail loud.
    empty_mods = [m for m, cols in dyad.feature_columns.items() if not cols]
    if empty_mods:
        print(
            f"Error: modality(ies) {empty_mods} have no numeric feature "
            f"columns (only 'time'). Provide at least one signal column per "
            f"modality.",
            file=sys.stderr,
        )
        sys.exit(2)

    qc_report = run_quality_check(dyad, raise_on_fail=False)
    print(format_qc_report(qc_report))
    for w in _align_warnings:
        print(f"  Note: {w.message}")
    if not qc_report.passed:
        print("Analysis stopped because QC failed. Fix the issues above or use the Python API with qc_raise_on_fail=False for exploratory inspection.", file=sys.stderr)
        sys.exit(2)

    # A4 routing: DynamicAnalyzer is the default DESCRIPTOR / CLI path.
    # (The scientific canonical is the audited evidence chain:
    #  pipeline_bridge + InferencePipeline.run_audited_evidence_chain.)
    analyzer = DynamicAnalyzer(
        window_size=args.window_size,
        surrogate_n=args.surrogates,
        max_lag_sec=args.max_lag,
        seed=args.seed,
        run_qc=False,
        cross_modal=getattr(args, "cross_modal", False),
    )

    print("  Running analysis...")
    results = analyzer.fit_transform(dyad)

    # P1-D strict input contract: refuse to emit a meaningless empty result.
    if not results.dynamic_features:
        print(
            "Error: no same-modality dyad pairs could be formed from the "
            "input. Provide a same-modality dyad (one modality with "
            "person_a/person_b columns, or two single-column files for the "
            "two dyad members). Use --cross-modal to pair across modalities.",
            file=sys.stderr,
        )
        sys.exit(2)

    results.parameters["qc"] = qc_report.to_dict()
    results.parameters["full_family_fdr"] = bool(
        getattr(args, "full_family_fdr", False)
    )
    if qc_report.overall_verdict != "PASS":
        results.diagnostics.append({
            "stage": "qc",
            "pair": "all",
            "reason": qc_report.overall_verdict,
            "detail": qc_report.to_dict(),
        })

    output_path = args.output or "results.json"
    results.export_viewer_json(output_path)
    print(f"  Results exported to: {output_path}")


def cmd_external_kit(args: argparse.Namespace) -> None:
    """Create a self-contained external usability-validation kit."""
    from .external_validation import create_external_validation_kit

    paths = create_external_validation_kit(
        args.output, seed=args.seed, n_dyads=args.n_dyads
    )
    print(f"External validation kit: {paths['root']}")
    print(f"Runbook: {paths['runbook']}")
    print("This kit is not evidence of construct validity.")


def cmd_external_check(args: argparse.Namespace) -> None:
    """Audit the structure and claim fields of an external result bundle."""
    from .external_validation import audit_external_bundle

    report = audit_external_bundle(args.input)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    if not report["structural_pass"]:
        raise SystemExit(1)


def cmd_migrate(args: argparse.Namespace) -> None:
    """Migrate legacy v1 canonical manifest/config inputs to v2 contracts."""
    from .migration import migrate_v1_project

    report = migrate_v1_project(
        manifest=args.manifest,
        config=args.config,
        output_dir=args.output,
        signal_type=args.signal_type,
        unit=args.unit,
        preprocessing_path=args.preprocessing_path,
        primary_modalities=args.primary_modalities,
        primary_endpoint=args.primary_endpoint,
    )
    print(f"Migrated manifest: {report['manifest']['destination']}")
    print(f"Migrated config: {report['config']['destination']}")
    print(f"Migration report: {report['report_path']}")
    print("Review all user-supplied scientific assumptions before analysis.")


def cmd_analyze(args: argparse.Namespace) -> None:
    """Check a dyadic study and write a plain-language report plus details."""
    from .canonical_runner import run_canonical

    try:
        res = run_canonical(args.manifest, args.config, args.output)
    except (ValueError, FileNotFoundError, OSError) as e:
        # Manifest/config/eligibility contract violations are expected,
        # user-facing errors — print cleanly and exit non-zero (fail-loud,
        # no traceback).
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    graph = res.chain.get("evidence_graph", {})
    decision = graph.get("decision", {})
    print("SyncPipe analysis complete.")
    print(f"  Read first: {Path(res.output_dir) / 'REPORT.md'}")
    print(
        f"  Data rows: {res.qc.get('total_rows')} listed | "
        f"{res.qc.get('included')} analyzed | {len(res.exclusions)} excluded"
    )
    print(
        "  Strongest conclusion supported: "
        f"{decision.get('permitted_claim', 'not available')}"
    )
    print(
        "  Condition comparison: "
        f"{decision.get('condition_claim', 'not available')}"
    )
    unresolved = decision.get("unresolved_rivals") or []
    if unresolved:
        print("  Still not ruled out: " + ", ".join(map(str, unresolved)))
    if res.exclusions:
        print("  Excluded observations:")
        for item in res.exclusions:
            print(
                f"    - {item.dyad_id}/{item.modality}/{item.condition}: "
                f"{item.detail}"
            )


def _json_ready(payload: Any) -> Any:
    """Convert demo values to ordinary JSON-compatible values."""
    from .export.runtime import json_safe

    return json_safe(payload)


def _write_json(path: Path, payload: Any) -> None:
    """Write strict JSON for demo artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _make_demo_cohort(
    n_dyads: int, *, hz: float, seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Create a small synthetic cohort for the design-check demonstration."""
    cohort = {}
    for i in range(n_dyads):
        dataset = generate_ground_truth_dyad(
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
        frame = dataset.modalities["behavior"]
        cohort[f"dyad_{i + 1:02d}"] = (
            frame["person_a"].to_numpy(dtype=float),
            frame["person_b"].to_numpy(dtype=float),
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

    # Surface alignment warnings as Notes (same policy as `analyze`) instead
    # of blanket-suppressing them. The synthetic demo is co-started by
    # construction, so this usually prints nothing — but if a future code path
    # emits a "device not co-started" warning, first-time users will see it
    # explained here rather than having it silently swallowed.
    with warnings.catch_warnings(record=True) as _align_warnings:
        warnings.simplefilter("always")
        ds.align(target_hz=1.0)
    ds.zscore()
    for w in _align_warnings:
        print(f"  Note: {w.message}")

    ds.add_context(start=0, end=150, label="PreTask")
    ds.add_context(start=150, end=300, label="Task")

    # A4 routing: default DESCRIPTOR / CLI DynamicAnalyzer path.
    # (Scientific canonical = pipeline_bridge + InferencePipeline.run_audited_evidence_chain.)
    analyzer = DynamicAnalyzer(
        window_size=10,
        surrogate_n=args.surrogates,
        max_lag_sec=0.0,
        seed=42,
    )

    print("  Running analysis...")
    results = analyzer.fit_transform(ds)

    output_arg = args.output or "demo_results.json"
    output_path = Path(output_arg)
    output_is_dir = output_path.suffix.lower() != ".json"
    if output_is_dir:
        # A non-.json path is treated as a DIRECTORY for the whole demo bundle.
        demo_dir = output_path
        demo_dir.mkdir(parents=True, exist_ok=True)
        viewer_path = demo_dir / "viewer_results.json"
        print(f"  Output directory: {demo_dir}/  (all demo artifacts written here)")
    else:
        # A .json path is the single viewer JSON file; the other demo
        # artifacts (feature table, audits, report) go alongside it.
        demo_dir = output_path.parent
        demo_dir.mkdir(parents=True, exist_ok=True)
        viewer_path = output_path
        print(f"  Viewer JSON: {viewer_path}")
        print(f"  Other demo artifacts written to: {demo_dir}/")
    results.export_viewer_json(str(viewer_path))

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
        n_pseudo_per_dyad=3,  # documented demo minimum; publication-grade audits use >=10
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
        "The p-values above are **raw (uncorrected)**. Because several features are "
        "tested on the same pair, apply Benjamini-Hochberg FDR across them before "
        "reporting significance. The terminal summary prints the FDR-corrected count.",
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

    p_values = existence.get("p_values", {})
    raw_p = [
        v for v in p_values.values()
        if isinstance(v, (int, float)) and np.isfinite(v)
    ]
    n_raw = int(sum(1 for p in raw_p if p < 0.05))
    if raw_p:
        # Apply BH-FDR across the tested features — the package's own stance
        # is that uncorrected p<0.05 is not reportable. Lead with the corrected
        # count; the raw count is shown only for transparency.
        _, fdr_sig = _bh_fdr_correction(raw_p, alpha=0.05)
        n_fdr = int(sum(fdr_sig))
    else:
        n_fdr = 0

    # Lead with the PRE-REGISTERED primary endpoint (peak_amplitude), not an
    # any-of-k "some feature was significant" count — the latter is a hidden
    # multiple-comparison. The remaining features are reported alongside as
    # reference, mirroring the per-modality primary-endpoint gate used in the
    # canonical evidence chain.
    from .feature_definitions import PRIMARY_EXISTENCE_ENDPOINT
    primary_p = p_values.get(PRIMARY_EXISTENCE_ENDPOINT)
    primary_sig = (
        isinstance(primary_p, (int, float))
        and np.isfinite(primary_p)
        and primary_p < 0.05
    )
    print("\n  Synchrony-existence audit (signal-level IAAFT):")
    if primary_p is not None and np.isfinite(primary_p):
        verdict = "SIGNIFICANT" if primary_sig else "not significant"
        print(
            f"    Pre-registered primary endpoint '{PRIMARY_EXISTENCE_ENDPOINT}': "
            f"p = {primary_p:.4f} ({verdict} at α=0.05)."
        )
    print(
        f"    Reference: {n_fdr} feature(s) significant after BH-FDR correction "
        f"(α=0.05); {n_raw} raw p < 0.05 before correction."
    )
    if n_raw and not n_fdr:
        print(
            "    Note: uncorrected p<0.05 findings did not survive multiple-comparison correction."
        )


def _positive_float(value: str) -> float:
    """argparse type: a float that must be > 0 (friendly error, no traceback)."""
    try:
        f = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}")
    if f <= 0:
        raise argparse.ArgumentTypeError(f"value must be > 0, got {f}")
    return f


def _min_int(lo: int):
    """argparse type factory: an int that must be >= lo."""
    def _check(value: str) -> int:
        try:
            i = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"invalid int value: {value!r}")
        if i < lo:
            raise argparse.ArgumentTypeError(f"value must be >= {lo}, got {i}")
        return i
    return _check


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # Pin the program name so `--help` reports "syncpipe" consistently
        # regardless of entry form. argparse's own default is unusable: under
        # `python -m` it derives prog from sys.argv[0] and prints "__main__.py".
        prog="syncpipe",
        description=(
            "SyncPipe checks co-movement between two aligned signals, tests "
            "common alternative explanations, and writes a readable report."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"syncpipe {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # analyze — canonical locked-protocol evidence chain
    p_analyze = sub.add_parser(
        "analyze",
        help="Analyze a multi-dyad study and write a report with checks and limits.",
    )
    p_analyze.add_argument(
        "-m", "--manifest", required=True,
        help="Strict v2 manifest CSV including signal_type, unit and "
             "preprocessing_path.",
    )
    p_analyze.add_argument(
        "-c", "--config", required=True,
        help="TOML file describing the conditions, main measure, and settings.",
    )
    p_analyze.add_argument(
        "-o", "--output", default="canonical_results",
        help="Output directory. Start with REPORT.md after the run.",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    # describe — design-agnostic descriptor path (exploratory / measurement core)
    p_describe = sub.add_parser(
        "describe",
        help="Describe one aligned pair without running the full study checks.",
    )
    p_describe.add_argument("-i", "--input", required=True, help="Comma-separated CSV paths.")
    p_describe.add_argument("-n", "--names", help="Comma-separated modality names.")
    p_describe.add_argument("--hz", type=_positive_float, default=1.0, help="Target sampling rate.")
    p_describe.add_argument("-o", "--output", default="results.json", help="Output JSON path.")
    p_describe.add_argument("--window-size", type=_min_int(2), default=10, help="WCC window size.")
    p_describe.add_argument("--surrogates", type=_min_int(1), default=500, help="Number of surrogates.")
    p_describe.add_argument(
        "--max-lag", type=float, default=0.0,
        help="v1 supports zero-lag WCC only; non-zero values fail loudly.",
    )
    p_describe.add_argument(
        "--cross-modal", action="store_true",
        help="OPT-IN: pair ACROSS modalities (legacy exploratory cross-modal "
             "description) instead of the v1 default SAME-MODALITY dyad "
             "pairing (person_a vs person_b within one modality).",
    )
    p_describe.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_describe.add_argument(
        "--contexts", nargs="*", help="Context labels: start,end,label[,score]."
    )
    p_describe.add_argument(
        "--full-family-fdr", action="store_true",
        help="Advanced: include every implemented measure in multiple-test correction.",
    )
    p_describe.set_defaults(func=cmd_describe)

    # demo
    p_demo = sub.add_parser("demo", help="Run complete synthetic demo + audit reports.")
    p_demo.add_argument(
        "-o", "--output", default="demo_results.json",
        help="Output location. A path ending in '.json' is the single viewer JSON "
             "file (other demo artifacts go alongside it). Any other path is treated "
             "as a DIRECTORY that receives the whole demo bundle.",
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
    p_demo.set_defaults(func=cmd_demo)

    # Utilities for migration and independent testing
    # external validation — independent usability/reproduction scaffold
    p_external_kit = sub.add_parser(
        "external-kit", help="Create an example project for independent testing."
    )
    p_external_kit.add_argument("-o", "--output", required=True)
    p_external_kit.add_argument("--seed", type=int, default=20260819)
    p_external_kit.add_argument("--n-dyads", type=_min_int(4), default=4)
    p_external_kit.set_defaults(func=cmd_external_kit)

    p_external_check = sub.add_parser(
        "external-check", help="Check that a result folder is complete and readable."
    )
    p_external_check.add_argument("-i", "--input", required=True)
    p_external_check.add_argument("-o", "--output")
    p_external_check.set_defaults(func=cmd_external_check)

    # migrate — explicit v1 canonical input migration
    p_migrate = sub.add_parser(
        "migrate",
        help="Convert older project files to the current format.",
    )
    p_migrate.add_argument("-m", "--manifest", required=True)
    p_migrate.add_argument("-c", "--config", required=True)
    p_migrate.add_argument("-o", "--output", required=True)
    p_migrate.add_argument("--signal-type", required=True)
    p_migrate.add_argument("--unit", required=True)
    p_migrate.add_argument("--preprocessing-path", required=True)
    p_migrate.add_argument(
        "--main-modalities", "--primary-modalities", dest="primary_modalities",
        nargs="+", required=True, help="Signal types used for the main result."
    )
    p_migrate.add_argument(
        "--main-measure", "--primary-endpoint", dest="primary_endpoint",
        default="peak_amplitude", help="Main value to compare between conditions."
    )
    p_migrate.set_defaults(func=cmd_migrate)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
