---
name: Bug report
about: Something in SyncPipe produced a wrong number, a crash, or a silent contract violation.
title: "[bug] "
labels: bug
assignees: ""

---

**What happened?**
A clear description of the wrong behavior. If a number looks statistically
wrong (e.g. a p-value that seems impossible), say what you expected instead.

**Minimal reproduction**
The smallest code + input shape that triggers it. SyncPipe fails loud by
design, so please include the exact error message or, for silent issues,
the evidence/audit JSON snippet.

```python
# your code here
```

**Environment**
- SyncPipe version (`pip show syncpipe`):
- Python version:
- OS:
- Optional extras in use (`[ecg]`, `[rqa]`, `[xls]`):

**Anything else?**
Surrogate seeds, dataset identities, or configuration that would help
reproduce the run. Do not paste confidential participant data.
