"""Optional ``trial_id`` / ``stimulus_key`` manifest columns.

The governing rule is that a manifest without these columns behaves exactly as before; trial rows are
an addition. The end-to-end test makes that concrete: the same data described as trial rows and as a
hand-concatenated file with a ``mask_path`` must produce an identical report.
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil

import numpy as np
import pandas as pd
import pytest

from syncpipe.canonical_runner import run_canonical
from syncpipe.contracts import MANIFEST_COLUMNS, OPTIONAL_TRIAL_COLUMNS, ManifestRecord, parse_manifest
from syncpipe.external_validation import create_external_validation_kit
from syncpipe.preparation import (
    contiguous_segments,
    iter_record_groups,
    load_record_group,
    validate_trial_alignment,
)

HZ = 1.0


# ----------------------------------------------------------------- helpers ----
@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    """A valid preprocessing provenance file and config, borrowed from the external validation kit."""
    root = tmp_path_factory.mktemp("kit")
    create_external_validation_kit(root / "kit", n_dyads=10)
    return root / "kit"


def _write_signal(path: pathlib.Path, values: np.ndarray) -> None:
    pd.DataFrame({"time": np.arange(values.size) / HZ, "value": values}).to_csv(path, index=False)


def _trial_signals(rng: np.random.Generator, length: int, coupling: float):
    shared = np.convolve(rng.normal(size=length + 6), np.ones(4) / 4, mode="same")[3:3 + length]
    a = shared * coupling + np.convolve(rng.normal(size=length + 6), np.ones(4) / 4, mode="same")[3:3 + length]
    b = shared * coupling + np.convolve(rng.normal(size=length + 6), np.ones(4) / 4, mode="same")[3:3 + length]
    return a, b


def _make_study(root: pathlib.Path, kit: pathlib.Path, *, n_dyads=10, n_trials=3, trial_len=60, concatenated=False):
    """Write a small two-condition study; ``concatenated`` chooses the manifest form, never the data."""
    rng = np.random.default_rng(7)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(kit / "preprocessing.json", root / "preprocessing.json")
    rows = []
    for dyad in range(n_dyads):
        for condition, coupling in (("condition_x", 0.4), ("condition_y", 0.9)):
            trials = [_trial_signals(rng, trial_len, coupling) for _ in range(n_trials)]
            base = {"dyad_id": f"d{dyad:02d}", "modality": "SYNTH", "condition": condition, "hz": HZ,
                    "signal_type": "synthetic_continuous_envelope", "unit": "z_score",
                    "preprocessing_path": "preprocessing.json"}
            if concatenated:
                a = np.concatenate([t[0] for t in trials])
                b = np.concatenate([t[1] for t in trials])
                stem = f"d{dyad:02d}_{condition}"
                _write_signal(root / f"{stem}_a.csv", a)
                _write_signal(root / f"{stem}_b.csv", b)
                mask = np.ones(a.size, dtype=int)
                mask[trial_len::trial_len] = 0                       # the seam convention: first sample of each later trial
                pd.DataFrame({"mask": mask}).to_csv(root / f"{stem}_mask.csv", index=False)
                rows.append({**base, "person_a_path": f"{stem}_a.csv", "person_b_path": f"{stem}_b.csv",
                             "mask_path": f"{stem}_mask.csv"})
            else:
                for k, (a, b) in enumerate(trials):
                    stem = f"d{dyad:02d}_{condition}_t{k}"
                    _write_signal(root / f"{stem}_a.csv", a)
                    _write_signal(root / f"{stem}_b.csv", b)
                    rows.append({**base, "person_a_path": f"{stem}_a.csv", "person_b_path": f"{stem}_b.csv",
                                 "mask_path": "", "trial_id": f"t{k}", "stimulus_key": f"stim{k}"})
    columns = list(MANIFEST_COLUMNS) + ([] if concatenated else list(OPTIONAL_TRIAL_COLUMNS))
    pd.DataFrame(rows)[columns].to_csv(root / "manifest.csv", index=False)
    config = (kit / "config.toml").read_text(encoding="utf-8").replace("surrogate_n = 99", "surrogate_n = 29") \
        .replace("n_permutations = 100", "n_permutations = 49")
    (root / "config.toml").write_text(config, encoding="utf-8")
    return root / "manifest.csv", root / "config.toml"


def _record(**over) -> ManifestRecord:
    fields = dict(dyad_id="d01", modality="EDA", condition="rest", person_a_path="a.csv", person_b_path="b.csv", hz=1.0,
                  signal_type="eda", unit="z_score", preprocessing_path="p.json")
    return ManifestRecord(**{**fields, **over})


def _write_manifest(path: pathlib.Path, rows: list[dict], optional=True) -> pathlib.Path:
    base = {"person_a_path": "a.csv", "person_b_path": "b.csv", "hz": 1.0, "signal_type": "eda", "unit": "z_score",
            "preprocessing_path": "p.json", "mask_path": ""}
    frame = pd.DataFrame([{**base, **r} for r in rows])
    cols = list(MANIFEST_COLUMNS) + [c for c in OPTIONAL_TRIAL_COLUMNS if optional and c in frame.columns]
    frame[cols].to_csv(path, index=False)
    return path


# ------------------------------------------------- guards: the old behaviour ----
def test_manifest_columns_are_unchanged_because_migration_writes_with_them():
    assert list(MANIFEST_COLUMNS) == ["dyad_id", "modality", "condition", "person_a_path", "person_b_path", "hz",
                                      "signal_type", "unit", "preprocessing_path", "mask_path"]
    assert OPTIONAL_TRIAL_COLUMNS == ("trial_id", "stimulus_key")


def test_a_manifest_without_trial_columns_parses_as_before(tmp_path):
    path = _write_manifest(tmp_path / "m.csv", [{"dyad_id": "d01", "modality": "EDA", "condition": "rest"},
                                                {"dyad_id": "d02", "modality": "EDA", "condition": "rest"}], optional=False)
    records = parse_manifest(path)
    assert [r.dyad_id for r in records] == ["d01", "d02"]
    assert all(r.trial_id is None and r.stimulus_key is None for r in records)


def test_plain_duplicate_rows_still_fail_with_the_original_message(tmp_path):
    path = _write_manifest(tmp_path / "m.csv", [{"dyad_id": "d01", "modality": "EDA", "condition": "rest"}] * 2, optional=False)
    with pytest.raises(ValueError, match=r"duplicate \(dyad_id, modality, condition\) rows"):
        parse_manifest(path)


def test_plain_duplicates_are_never_merged_into_a_group():
    groups = iter_record_groups([_record(), _record()])
    assert [len(g) for g in groups] == [1, 1]                  # so the downstream duplicate check still fires


# ------------------------------------------------------------ parsing rules ----
def test_trial_rows_share_a_series_and_blank_cells_mean_none(tmp_path):
    rows = [{"dyad_id": "d01", "modality": "EDA", "condition": "rest", "trial_id": "t1", "stimulus_key": "s1"},
            {"dyad_id": "d01", "modality": "EDA", "condition": "rest", "trial_id": "t2", "stimulus_key": "s2"},
            {"dyad_id": "d02", "modality": "EDA", "condition": "rest", "trial_id": "t1", "stimulus_key": ""}]
    # d02 has no stimulus keys at all while d01 has: allowed (the order rule applies only among keyed dyads)
    records = parse_manifest(_write_manifest(tmp_path / "m.csv", rows))
    assert [(r.dyad_id, r.trial_id, r.stimulus_key) for r in records] == [
        ("d01", "t1", "s1"), ("d01", "t2", "s2"), ("d02", "t1", None)]
    assert [len(g) for g in iter_record_groups(records)] == [2, 1]


@pytest.mark.parametrize("rows, message", [
    ([{"trial_id": "t1"}, {"trial_id": ""}], "mixes rows with and without trial_id"),
    ([{"trial_id": "t1"}, {"trial_id": "t1"}], "duplicate trial_id"),
    ([{"trial_id": "", "stimulus_key": "s1"}], "stimulus_key requires trial_id"),
    ([{"trial_id": "t1", "stimulus_key": "s1"}, {"trial_id": "t2", "stimulus_key": ""}], "for every trial or for none"),
    ([{"trial_id": "t1", "stimulus_key": "s1"}, {"trial_id": "t2", "stimulus_key": "s1"}], "repeats a stimulus_key"),
])
def test_invalid_trial_structure_fails_loudly(tmp_path, rows, message):
    base = {"dyad_id": "d01", "modality": "EDA", "condition": "rest"}
    with pytest.raises(ValueError, match=message):
        parse_manifest(_write_manifest(tmp_path / "m.csv", [{**base, **r} for r in rows]))


def test_stimulus_order_must_match_across_dyads(tmp_path):
    rows = [{"dyad_id": "d01", "trial_id": "t1", "stimulus_key": "a"}, {"dyad_id": "d01", "trial_id": "t2", "stimulus_key": "b"},
            {"dyad_id": "d02", "trial_id": "t1", "stimulus_key": "b"}, {"dyad_id": "d02", "trial_id": "t2", "stimulus_key": "a"}]
    with pytest.raises(ValueError, match="stimulus order differs between dyads"):
        parse_manifest(_write_manifest(tmp_path / "m.csv", [{"modality": "EDA", "condition": "rest", **r} for r in rows]))


def test_json_schema_accepts_the_optional_properties():
    schema = json.loads((pathlib.Path(__file__).resolve().parents[2] / "syncpipe" / "schemas" / "manifest-record.schema.json")
                        .read_text(encoding="utf-8"))
    assert {"trial_id", "stimulus_key"} <= set(schema["properties"]) and schema["additionalProperties"] is False
    assert {"trial_id", "stimulus_key"}.isdisjoint(schema["required"])


# ---------------------------------------------------------------- stacking ----
def _trial_group(tmp_path, lengths, *, keys=True, masks=None, hz=1.0):
    rng = np.random.default_rng(1)
    shutil.copy(pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "bsync_golden" / "pair.csv", tmp_path / "unused.csv")
    group = []
    for k, n in enumerate(lengths):
        a, b = _trial_signals(rng, n, 0.5)
        _write_signal(tmp_path / f"a{k}.csv", a)
        _write_signal(tmp_path / f"b{k}.csv", b)
        mask_path = None
        if masks and masks.get(k) is not None:
            pd.DataFrame({"m": masks[k]}).to_csv(tmp_path / f"m{k}.csv", index=False)
            mask_path = f"m{k}.csv"
        group.append(_record(person_a_path=f"a{k}.csv", person_b_path=f"b{k}.csv", mask_path=mask_path, hz=hz,
                             trial_id=f"t{k}", stimulus_key=f"s{k}" if keys else None))
    return group


def test_trials_are_stacked_with_a_continuous_clock_and_seam_markers(tmp_path):
    group = _trial_group(tmp_path, [60, 60, 90])
    rec = load_record_group(group, tmp_path)
    assert len(rec.person_a) == 210 and rec.trial_lengths == (60, 60, 90)
    assert rec.trial_ids == ("t0", "t1", "t2") and rec.stimulus_keys == ("s0", "s1", "s2")
    assert np.allclose(np.diff(rec.person_a["time"].to_numpy()), 1.0) and np.array_equal(rec.person_a["time"], rec.person_b["time"])
    assert np.flatnonzero(~rec.discontinuity_mask).tolist() == [60, 120]
    assert [tuple(seg) for seg in contiguous_segments(rec.discontinuity_mask)] == [(0, 60), (61, 120), (121, 210)]
    first = pd.read_csv(tmp_path / "a0.csv")["value"].to_numpy()
    assert np.array_equal(rec.person_a["value"].to_numpy()[:60], first)


def test_a_trials_own_mask_is_combined_with_the_seams(tmp_path):
    own = np.ones(60, dtype=int)
    own[10:13] = 0
    rec = load_record_group(_trial_group(tmp_path, [60, 60], masks={1: own}), tmp_path)
    assert np.flatnonzero(~rec.discontinuity_mask).tolist() == [60, 70, 71, 72]


def test_a_single_plain_row_loads_exactly_as_before(tmp_path):
    group = _trial_group(tmp_path, [80])
    plain = ManifestRecord(**{**group[0].__dict__, "trial_id": None, "stimulus_key": None})
    rec = load_record_group([plain], tmp_path)
    assert rec.trial_ids is None and rec.trial_lengths is None and rec.discontinuity_mask is None


def test_trials_with_different_columns_are_refused(tmp_path):
    group = _trial_group(tmp_path, [60, 60])
    frame = pd.read_csv(tmp_path / "a1.csv").rename(columns={"value": "other"})
    frame.to_csv(tmp_path / "a1.csv", index=False)
    with pytest.raises(ValueError, match="same columns"):
        load_record_group(group, tmp_path)


def test_equal_trial_lengths_are_required_across_dyads_when_stimuli_are_keyed(tmp_path):
    same = [load_record_group(_trial_group(tmp_path, [60, 60]), tmp_path) for _ in range(2)]
    for i, r in enumerate(same):
        r.dyad_label = f"d{i}"
    validate_trial_alignment(same)                                       # equal: fine
    other = load_record_group(_trial_group(tmp_path, [60, 75]), tmp_path)
    other.dyad_label = "d9"
    with pytest.raises(ValueError, match="trial lengths differ between dyads"):
        validate_trial_alignment([*same, other])
    unkeyed = load_record_group(_trial_group(tmp_path, [60, 75], keys=False), tmp_path)
    validate_trial_alignment([same[0], unkeyed])                         # no stimulus_key, no claim, no check


# -------------------------------------------------------------- end to end ----
def test_trial_rows_and_a_hand_concatenated_file_give_an_identical_report(tmp_path, kit):
    manifest_t, config_t = _make_study(tmp_path / "trial_rows", kit, concatenated=False)
    manifest_c, config_c = _make_study(tmp_path / "concatenated", kit, concatenated=True)
    run_canonical(manifest_t, config_t, tmp_path / "out_t")
    run_canonical(manifest_c, config_c, tmp_path / "out_c")
    report_t = (tmp_path / "out_t" / "REPORT.md").read_text(encoding="utf-8")
    report_c = (tmp_path / "out_c" / "REPORT.md").read_text(encoding="utf-8")
    # The one legitimate difference: the report counts manifest ROWS, and the trial form has one row per
    # trial (3 trials x 20 series = 60) where the concatenated form has one row per series (20).
    row_counts = lambda text: re.findall(r"- Rows (?:listed|analyzed): (\d+)", text)  # noqa: E731
    assert row_counts(report_t) == ["60", "60"] and row_counts(report_c) == ["20", "20"]
    normalise = lambda text: re.sub(r"(- Rows (?:listed|analyzed): )\d+", r"\1N", text)  # noqa: E731
    assert normalise(report_t) == normalise(report_c)
    assert "peak_amplitude" in report_t
