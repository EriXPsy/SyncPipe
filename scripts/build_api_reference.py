"""build_api_reference.py — regenerate docs/API_REFERENCE.md from the code.

Dependency-free introspection of the ``syncpipe`` package's public surface:
one section per curated public module, listing classes and functions with
their signature and docstring summary line. The generated file is a mirror of
the code, not an independent document — after any public-API change, re-run:

    python scripts/build_api_reference.py

The reference is launch-level: no internal jargon, neutral positioning.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PACKAGE = "syncpipe"
OUT = ROOT / "docs" / "API_REFERENCE.md"

# Curated public modules in stable, user-facing order (data flow: data ->
# computation -> inference -> evidence -> reporting).
MODULES = [
    "syncpipe.core",
    "syncpipe.dataset",
    "syncpipe.pipeline_bridge",
    "syncpipe.batch",
    "syncpipe.dynamic_features",
    "syncpipe.feature_definitions",
    "syncpipe.inference_pipeline",
    "syncpipe.design_controls",
    "syncpipe.morphology",
    "syncpipe.session_threshold",
    "syncpipe.feature_status",
    "syncpipe.qc",
    "syncpipe.io",
    "syncpipe.synthetic",
    "syncpipe.external_validation",
    "syncpipe.migration",
]

MODULE_TITLES = {
    "syncpipe.core": "Core objects",
    "syncpipe.dataset": "Dataset assembly",
    "syncpipe.pipeline_bridge": "Data-to-pipeline bridge",
    "syncpipe.batch": "Batch computation",
    "syncpipe.dynamic_features": "WCC computation and features",
    "syncpipe.feature_definitions": "Feature contracts (registry)",
    "syncpipe.inference_pipeline": "Inference (L0/L1/L2 evidence chain)",
    "syncpipe.design_controls": "Design controls",
    "syncpipe.morphology": "Morphology analysis",
    "syncpipe.session_threshold": "Session-pooled thresholds",
    "syncpipe.feature_status": "Feature status tables",
    "syncpipe.qc": "Quality control",
    "syncpipe.io": "Data loading",
    "syncpipe.synthetic": "Synthetic ground truth",
    "syncpipe.external_validation": "External validation kit",
    "syncpipe.migration": "Version migration",
}


def _summary(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    for line in doc.splitlines():
        line = line.strip()
        # Skip decorative separator lines (====/----) and box art.
        if line and not set(line) <= set("=-_─═—*#~+"):
            return line
    return ""


def _signature(obj) -> str:
    try:
        sig = str(inspect.signature(obj))
        return sig
    except (TypeError, ValueError):
        return "(...)"


def _public_members(module):
    classes, functions = [], []
    for name in sorted(dir(module)):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        obj_module = getattr(obj, "__module__", "")
        if not str(obj_module).startswith(PACKAGE):
            continue
        if inspect.isclass(obj):
            classes.append((name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append((name, obj))
    return classes, functions


def main() -> None:
    lines = [
        "# API reference",
        "",
        "Generated from the installed package by "
        "`scripts/build_api_reference.py` — do not edit by hand; re-run the "
        "script after public-API changes. Version: see `syncpipe.__version__`.",
        "",
        "Top-level entry points (see README): `syncpipe.analyze()` for a full "
        "study from a manifest, `syncpipe.make_example()` for a runnable "
        "example project.",
        "",
    ]
    for mod_name in MODULES:
        try:
            module = importlib.import_module(mod_name)
        except ImportError as e:
            print(f"skip {mod_name}: {e}")
            continue
        title = MODULE_TITLES.get(mod_name, mod_name)
        lines += [f"## {title}", "", f"Module: `{mod_name}`", ""]
        classes, functions = _public_members(module)
        if classes:
            lines += ["### Classes", ""]
            for name, obj in classes:
                summary = _summary(obj)
                lines += [f"- **`{name}{_signature(obj)}`** — {summary}"]
            lines.append("")
        if functions:
            lines += ["### Functions", ""]
            for name, obj in functions:
                summary = _summary(obj)
                lines += [f"- **`{name}{_signature(obj)}`** — {summary}"]
            lines.append("")
        if not classes and not functions:
            lines += ["_(no public members)_", ""]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
