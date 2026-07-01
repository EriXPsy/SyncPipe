"""
SyncPipe Validation Suite
=====================

Ground-truth recovery experiments for the 6 dynamic features.

Levels
------
- Level 1: Coupling x seed grid (recovery, monotonicity, reliability,
           Type-I error). See ``recovery.run_level1_grid``.
- Level 2: SNR robustness curves (noise_ratio x coupling x seed).
           See ``snr.run_level2_grid``.
- Level 3: Surrogate-based significance (Type-I error + power).
           See ``pgt1_intensity.run_level3_grid``.

Each level exposes a ``run_*`` function that returns a tidy
:class:`pandas.DataFrame` plus a small set of summary statistics.
The functions are deterministic (seeded) so the resulting tables
can be regenerated bit-for-bit by anyone.
"""
from .recovery import (
    Level1Config,
    run_level1_grid,
    summarise_level1,
    summarise_definedness,
    split_half_icc,
)
from .snr import (
    Level2Config,
    run_level2_grid,
    summarise_level2,
    robustness_curves,
)
from .pgt1_intensity import (
    Level3Config,
    run_level3_grid,
    apply_bh_fdr_within_noise,
    summarise_level3,
    ft_surrogate,
    prtf_surrogate,
    iaaft_surrogate,
    phipson_smyth_p,
    bh_fdr,
    FEATURE_TAILS,
    REFERENCE_TAILS,
    FEATURE_P_COLUMNS,
    REFERENCE_P_COLUMNS,
    # Surrogate-derived threshold (DECISION-01 revised 2026-06-21)
    compute_dyad_surrogate_threshold,
    # Legacy aliases
    DIAGNOSTIC_TAILS,
    DIAGNOSTIC_P_COLUMNS,
)
from .across_stim_shuffle import (
    across_stim_shuffle,
    across_stim_shuffle_test,
)
from .pgt2_structure import (
    PGT2Config,
    run_pgt2_grid,
    summarise_pgt2,
    test_pgt2_hypotheses,
)
from .pgt3_temporal import (
    PGT3Config,
    run_pgt3_grid,
    summarise_pgt3,
    test_pgt3_hypotheses,
)
from .pgt3_extended import (
    PGT3ExtendedConfig,
    run_pgt3_extended_grid,
    shape_robustness_table,
    ideal_baseline_metrics,
    degradation_summary,
)

from .egt4_emergent import (
    EGT4Config,
    run_egt4_matrix,
    summarise_egt4,
    eg4_generalisation_gap,
)
from .l2_between_condition import (
    L2Result,
    between_condition_fdr,
    between_condition_by_modality,
)

__all__ = [
    # Level 1
    "Level1Config",
    "run_level1_grid",
    "summarise_level1",
    "summarise_definedness",
    "split_half_icc",
    # Level 2
    "Level2Config",
    "run_level2_grid",
    "summarise_level2",
    "robustness_curves",
    # Level 3
    "Level3Config",
    "run_level3_grid",
    "apply_bh_fdr_within_noise",
    "summarise_level3",
    "ft_surrogate",
    "prtf_surrogate",
    "iaaft_surrogate",
    "phipson_smyth_p",
    "bh_fdr",
    "FEATURE_TAILS",
    "REFERENCE_TAILS",
    "FEATURE_P_COLUMNS",
    "REFERENCE_P_COLUMNS",
    # Surrogate-derived threshold
    "compute_dyad_surrogate_threshold",
    # Legacy aliases
    "DIAGNOSTIC_TAILS",
    "DIAGNOSTIC_P_COLUMNS",
    # Across-stim shuffle
    "across_stim_shuffle",
    "across_stim_shuffle_test",
    # PGT-2 Structure Recovery
    "PGT2Config",
    "run_pgt2_grid",
    "summarise_pgt2",
    "test_pgt2_hypotheses",
    # PGT-3 Temporal Recovery (Core)
    "PGT3Config",
    "run_pgt3_grid",
    "summarise_pgt3",
    "test_pgt3_hypotheses",
    # PGT-3 Extended (Shape Robustness Diagnostic)
    "PGT3ExtendedConfig",
    "run_pgt3_extended_grid",
    "shape_robustness_table",
    "ideal_baseline_metrics",
    "degradation_summary",
    # EGT-4 Emergent Dynamics
    "EGT4Config",
    "run_egt4_matrix",
    "summarise_egt4",
    "eg4_generalisation_gap",
    # L2 Between-Condition Null
    "L2Result",
    "between_condition_fdr",
    "between_condition_by_modality",
]
