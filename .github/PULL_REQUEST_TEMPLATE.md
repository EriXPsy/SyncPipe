## Summary

What does this PR change, and why?

## What changed

- [ ] Code (which modules)
- [ ] Tests (which new/updated tests)
- [ ] Documentation (which pages)

## Verification

- [ ] `pytest tests/ -m "not slow" -q` passes (fast layer)
- [ ] Full suite (`pytest tests/ -q`) passes if locked behavior changed
- [ ] `tests/test_suite_health.py` counts updated **with a stated reason** if
      tests were added/removed
- [ ] New statistical endpoints have an H0-calibration row in
      `tests/test_h0_calibration_endpoints.py`

## Governance checks (if applicable)

- [ ] If this changes a locked default, threshold, or feature definition:
      a `DECISION_LOG.md` entry is included.
- [ ] If this changes what reports may claim: the claim-ceiling wording in
      `README.md` / `docs/LIMITATIONS.md` was updated consistently.
- [ ] No new hard dependency in the measurement core (extras only).
