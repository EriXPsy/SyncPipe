# `dwell_time` NaN — what it is, why it happens, and how to answer a reviewer

> 中文要点：dwell_time 的 NaN 不是设计缺陷，而是"持续同步片段"这一构念在弱耦合/短轨迹下的**空值（null）**。框架已内置可定义性审计（definedness + `[WARN]` + `p_definedness`）透明处理。只要如实报告可定义率，审稿人不会认为存在严重操作化问题；反之若隐瞒 NaN 或把 dwell 当作无条件主终点，才会成为问题。

---

## 1. The question

> "dwell_time 目前出现的是什么问题？为什么这么多 dyad 无法定义（NaN）？如果是这样的话，会不会被 Reviewer 认为设计、操作化存在严重问题？"

## 2. What is actually happening (code-level)

`dwell_time` = mean duration (seconds) of *sustained above-threshold WCC runs*.
In `multisync/feature_definitions.py::compute_dwell_time`:

```python
above = _binarize_with_hysteresis(wcc, threshold, hysteresis_delta)
if not above.any():
    return float("nan")          # ← the NaN source
run_lengths = ends - starts
if run_lengths.size == 0:
    return float("nan")
return float(np.mean(run_lengths)) / hz
```

So `dwell_time` is **undefined whenever the WCC trace contains zero sustained
above-threshold runs**. That happens when the trace never crosses and *stays*
above the onset threshold long enough to form a run.

`switching_rate` shares the same precondition (it counts transitions of the same
binary above/below state), so it too is `NaN` when no run exists — though in
practice it is more often defined than dwell because a single crossing already
yields one transition.

## 3. Why so many are undefined — it is the construct's null, not a bug

Whether a "sustained above-threshold run" exists is a direct function of two
physical quantities:

1. **Coupling strength** — stronger synchrony → higher, more persistent WCC →
   more runs. Under weak coupling (e.g. rest, or non-interacting dyads) the WCC
   hovers near zero and rarely breaches threshold.
2. **Trace length / WCC resolution** — a run needs enough consecutive samples
   above threshold. A trace that is only 18–22 WCC points long (Gordon) simply
   cannot contain a *sustained* run, no matter how coupled.

Real-data definedness (from the five OSF datasets, `realdata_l2_audit.py` probe):

| Dataset | WCC length context | dwell_time defined |
|---|---|---|
| Han (long traces) | long | **98.4%** |
| Lerique (rest/task, 1 Hz) | long | **59.7%** |
| Gordon (18–22 WCC points) | **very short** | **24.3%** |
| Andersen | HR, moderate | n/a in extracted CSV (run without dwell) |
| Bizzego | IBI 2 Hz | n/a in extracted CSV (run without dwell) |

The gradient — Han 98% → Lerique 60% → Gordon 24% — is exactly what the
construct predicts. Gordon's 24% is not a defect; it is the *null of sustained
synchrony* manifesting in data that is too short to exhibit it. The morphology
report corroborates this independently: only **12 of 345** Gordon WCC traces
ever cross the conventional 0.5 threshold.

## 4. Would a reviewer see a serious design flaw? — No, *if framed correctly*

A reviewer's concern would be legitimate only under two failures, neither of
which SyncPipe commits:

- **(a) Hiding the NaN** — presenting dwell as a primary endpoint while
  silently dropping undefined dyads. That would be survivor bias.
- **(b) Treating NaN as missing-at-random** — imputing or pooling undefined
  values without testing whether missingness is informative.

SyncPipe does the opposite, by design:

1. `dwell_time` and `switching_rate` are declared **conditional / L1-structure**
   descriptors in the SSoT (`feature_definitions.py`) and in the README feature
   table — *not* unconditional primary endpoints. `peak_amplitude` (L0,
   intensity) is the unconditional workhorse and the only FDR-family member that
   is defined in ~100% of real dyads.
2. The L2 layer (`validation/l2_between_condition.py`) runs the contrast **only
   on dyads where the feature is finite in BOTH conditions** (`defined_a`,
   `defined_b`), so undefined dyads never enter the test silently.
3. It reports a **definedness audit**: `p_definedness` via permutation tests
   whether definedness differs across conditions (i.e. is the missingness
   informative / MNAR?). If it does, the pipeline emits
   **`[WARN] dwell_time definedness diff`** in `summarize()`.

So the NaN is *evidence of methodological honesty*, not a flaw. The correct
narrative is: "dwell_time measures the temporal structure of sustained
synchrony episodes; under weak coupling / short recordings no such episodes
exist, and the framework reports this transparently rather than forcing a
number."

## 5. Ready-to-use reviewer response (draft)

> *"dwell_time and switching_rate are conditional descriptors defined only when
> a WCC trace contains at least one sustained above-threshold run. They are
> intentionally not treated as unconditional primary endpoints; peak_amplitude
> (defined in ~100% of dyads) carries the primary intensity claim. Under weak
> coupling or very short recordings (e.g. Gordon's 18–22 WCC-point traces) no
> sustained run exists, so the descriptors are undefined by construction — this
> is the null of 'sustained synchrony', not a computation error. We report
> per-condition definedness rates for every conditional descriptor and run the
> between-condition test only on dyads where the feature is finite in both
> conditions; a permutation-based definedness test (p_definedness) flags any
> informative missingness. This transparency is a designed safeguard against
> survivor bias, not a limitation of the operationalization."*

## 6. What NOT to do

- Do not impute dwell_time / switching_rate to 0 or to the median.
- Do not enter undefined values into the FDR family without the definedness
  guard.
- Do not present dwell as a standalone confirmatory endpoint in datasets where
  definedness is low (e.g. Gordon) without explicitly reporting the rate.

## 7. Where this is enforced

- `multisync/feature_definitions.py` — `compute_dwell_time` / `compute_switching_rate` return `NaN` when no run.
- `multisync/validation/l2_between_condition.py` — `defined_a` / `defined_b` / `p_definedness`; skips undefined dyads.
- `multisync/inference_pipeline.py::summarize()` — emits `[WARN] … definedness diff`.
- `README.md` (SOP) — "Handling undefined descriptors" rule.
