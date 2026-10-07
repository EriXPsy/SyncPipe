"""Bizzego 入口真实执行回归；合成信号不冒充原始研究数据。"""

from __future__ import annotations

from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_bizzego_replication.py"

pytestmark = pytest.mark.skipif(
    not _SCRIPT.exists(),
    reason=("run_bizzego_replication.py is untracked in the published repository; "
            "these contracts apply only to full working checkouts"),
)

import builtins
import json
import sys

import numpy as np
import pandas as pd
import pytest

import importlib

if _SCRIPT.exists():
    runner = importlib.import_module("scripts.run_bizzego_replication")
else:  # untracked in published checkouts; module-level skipif handles it
    runner = None


def manifest(tmp_path):
    t = np.arange(240) / 2
    rng = np.random.default_rng(12)
    for member in ('a', 'b'):
        pd.DataFrame({'time': t, 'value': np.sin(t / 5) + rng.normal(0, .2, t.size)}).to_csv(tmp_path / f'{member}.csv', index=False)
    (tmp_path / 'provenance.json').write_text(json.dumps({
        'schema_version': '1.0.0', 'signal_type': 'synthetic', 'output_unit': 'arbitrary',
        'software': {'name': 'numpy', 'version': np.__version__},
        'steps': [{'name': 'synthetic_test_generator', 'parameters': {'seed': 12}}],
    }), encoding='utf-8')
    rows = [dict(dyad_id=d, modality='IBI', condition='HS', person_a_path='a.csv',
                 person_b_path='b.csv', hz=2, signal_type='synthetic', unit='arbitrary',
                 preprocessing_path='provenance.json') for d in ('F01', 'F02')]
    path = tmp_path / 'manifest.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def invoke(monkeypatch, *args):
    monkeypatch.setattr(sys, 'argv', ['run_bizzego_replication', *map(str, args)])
    return runner.main()


def test_manifest_full_execution_without_raw_module(tmp_path, monkeypatch):
    path = manifest(tmp_path)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'syncpipe.realtest.bizzego_2020':
            raise AssertionError('raw loader must not be imported')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    out = tmp_path / 'out'
    assert invoke(monkeypatch, '--from-manifest', path, '--out-dir', out,
                  '--limit-dyads', 1, '--surrogate-n', 2) == 0
    summary = json.loads((out / 'replication_summary.json').read_text())
    assert summary['hz'] == 2
    assert summary['n_features'] == 1
    assert summary['excluded'][0]['reason'] == 'limit_dyads'
    assert len(json.loads((out / 'l1_structure.json').read_text())['results']) == 1


def test_legacy_schema_error_precedes_inference(tmp_path, monkeypatch):
    path = tmp_path / 'old.csv'
    pd.DataFrame([{'dyad_id': 'F01', 'modality': 'IBI', 'condition': 'HS',
                   'person_a_path': 'a.csv', 'person_b_path': 'b.csv', 'hz': 2}]).to_csv(path, index=False)
    out = tmp_path / 'out'
    assert invoke(monkeypatch, '--from-manifest', path, '--out-dir', out) == 2
    reason = json.loads((out / 'diagnostics.json').read_text())['reason']
    assert "['signal_type', 'unit', 'preprocessing_path']" in reason


def test_invalid_provenance_and_missing_signal_diagnosed(tmp_path):
    path = manifest(tmp_path)
    (tmp_path / 'provenance.json').write_text('{}')
    loaded, excluded, seen = runner._load_from_manifest(path, None)
    assert not loaded and len(excluded) == seen == 2
    assert all('provenance missing' in row['reason'] for row in excluded)
    path = manifest(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[0, 'person_a_path'] = 'missing.csv'
    frame.to_csv(path, index=False)
    loaded, excluded, _ = runner._load_from_manifest(path, None)
    assert len(loaded) == len(excluded) == 1
    assert 'missing.csv' in excluded[0]['reason']


def test_raw_blocker_and_no_overwrite(tmp_path, monkeypatch):
    out = tmp_path / 'out'
    assert invoke(monkeypatch, '--data-root', tmp_path, '--out-dir', out) == 3
    before = (out / 'diagnostics.json').read_bytes()
    assert invoke(monkeypatch, '--data-root', tmp_path, '--out-dir', out) == 1
    assert (out / 'diagnostics.json').read_bytes() == before


def test_negative_limit_rejected(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as exc:
        invoke(monkeypatch, '--data-root', tmp_path, '--limit-dyads', -1)
    assert exc.value.code == 2
