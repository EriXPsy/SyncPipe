# Plain-language policy

SyncPipe keeps technical names inside code and machine-readable files. Users
should not need those names to run an analysis or understand `REPORT.md`.

## Start with questions

Prefer:

- Is the result larger than expected from independent signals?
- Are real partners stronger than mismatched partners?
- Does the result depend on the original timing?
- What explanations are still possible?

Avoid starting with:

- L0/L1/L2;
- endpoint specification;
- null-family governance;
- claimability propagation;
- evidence-chain stage status.

## Preferred words

| Internal term | User-facing wording |
|---|---|
| endpoint | main measure |
| primary modalities | signal types used for the main result |
| null / surrogate | randomized comparison |
| evidence profile | results of the separate checks |
| claim ceiling | strongest conclusion supported |
| provenance | processing record |
| eligibility | enough usable data to answer the question |
| definedness | whether the value could be calculated |
| FDR | correction for testing several values |
| WCC | sliding-window Pearson correlation |

A technical term may appear once in parentheses when users need it to read a
method section or JSON field.

## Report order

1. strongest supported conclusion;
2. condition comparison;
3. explanations still possible;
4. checks and results;
5. data used and excluded;
6. important limits;
7. technical and reproducibility details.

## Error messages

Every user-facing error should answer:

1. what is wrong;
2. why the analysis cannot continue;
3. what the user should change.

## Scientific restraint

Plain language is not marketing simplification. Do not replace “insufficient
information” with “no effect,” and do not replace “co-movement” with “coupling”
unless the relevant alternative explanations were tested.
