"""Dyad pairing policy and pair iteration.

This module is the single source of truth for pair selection and pair keys.
Feature extraction, segmented analysis, and WCC caching should reuse it.
"""


def pairing_policy(dataset, cross_modal: bool = False) -> str:
    """Return the effective dyad-pairing policy for the result manifest."""
    if cross_modal:
        return "cross_modal"
    feat_cols = dataset.feature_columns
    if any(len(cols) >= 2 for cols in feat_cols.values()):
        return "same_modality"
    names = dataset.modality_names
    if len(names) == 2 and all(len(feat_cols[name]) == 1 for name in names):
        return "two_file_dyad_fallback"
    return "same_modality_no_pair"


def iter_dyad_pairs(dataset, cross_modal: bool = False):
    """Yield valid dyad pairs using the canonical pairing policy."""
    feat_cols = dataset.feature_columns
    names = dataset.modality_names

    if cross_modal:
        for i, name_a in enumerate(names):
            for name_b in names[i + 1:]:
                for col_a in feat_cols[name_a]:
                    for col_b in feat_cols[name_b]:
                        x = dataset.get_aligned_array(name_a, col_a)
                        y = dataset.get_aligned_array(name_b, col_b)
                        if x is not None and y is not None:
                            yield (
                                f"{name_a}_{col_a}__{name_b}_{col_b}",
                                name_a, name_b, col_a, col_b, x, y,
                            )
        return

    yielded = 0
    for name in names:
        cols = feat_cols[name]
        for i, col_a in enumerate(cols):
            for col_b in cols[i + 1:]:
                x = dataset.get_aligned_array(name, col_a)
                y = dataset.get_aligned_array(name, col_b)
                if x is not None and y is not None:
                    yielded += 1
                    yield (
                        f"{name}_{col_a}__{name}_{col_b}",
                        name, name, col_a, col_b, x, y,
                    )

    if yielded == 0 and len(names) == 2:
        ca, cb = feat_cols[names[0]], feat_cols[names[1]]
        if len(ca) == 1 and len(cb) == 1:
            x = dataset.get_aligned_array(names[0], ca[0])
            y = dataset.get_aligned_array(names[1], cb[0])
            if x is not None and y is not None:
                yield (
                    f"{names[0]}_{ca[0]}__{names[1]}_{cb[0]}",
                    names[0], names[1], ca[0], cb[0], x, y,
                )
