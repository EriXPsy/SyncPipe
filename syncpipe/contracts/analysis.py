"""Internal definitions for one reproducible analysis.

Users normally edit a small TOML file. These immutable classes keep those
settings consistent across calculation and reporting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional, Tuple, Union

from ..feature_definitions import PRIMARY_EXISTENCE_ENDPOINT


@dataclass(frozen=True)
class NullSpec:
    """How randomized comparison data are made for the main measure."""

    name: str
    level: str
    sampling_unit: str
    tail: str
    preserves: Tuple[str, ...]
    destroys: Tuple[str, ...]
    missingness_policy: str


@dataclass(frozen=True)
class EndpointSpec:
    """Definition, comparison method, and interpretation limits for one measure."""

    name: str
    estimand: str
    unit: str
    fdr_family: str
    duration_policy: str
    null: NullSpec
    permitted_claim: str
    forbidden_claims: Tuple[str, ...]


@dataclass(frozen=True)
class ModalitySpec:
    """A signal label and whether it is used for the study's main result."""

    label: str
    role: str  # primary | comparator

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("modality label must not be empty")
        if self.role not in {"primary", "comparator"}:
            raise ValueError("modality role must be 'primary' or 'comparator'")


SIGNAL_LEVEL_IAAFT_PEAK = NullSpec(
    name="signal_level_segmentwise_iaaft",
    level="L0",
    sampling_unit="raw_signal_segment",
    tail="two_sided",
    preserves=("marginal_distribution", "approximate_power_spectrum", "segment_boundaries"),
    destroys=("cross_person_dependence",),
    missingness_policy="finite_contiguous_segments",
)

ENDPOINT_SPECS = {
    PRIMARY_EXISTENCE_ENDPOINT: EndpointSpec(
        name=PRIMARY_EXISTENCE_ENDPOINT,
        estimand="median between-condition difference in the maximum 3-point-smoothed zero-lag sliding-window Pearson correlation",
        unit="correlation",
        fdr_family="primary",
        duration_policy="equal_observation_required_and_duration_audited",
        null=SIGNAL_LEVEL_IAAFT_PEAK,
        permitted_claim="condition difference in maximum local co-fluctuation under the declared procedure",
        forbidden_claims=(
            "universal interpersonal synchrony biomarker",
            "causal interpersonal coupling",
            "lead-lag direction",
        ),
    ),
}


def resolve_endpoint_spec(name: str) -> EndpointSpec:
    """Resolve a registered endpoint or fail before inference begins."""
    try:
        return ENDPOINT_SPECS[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"analysis setting 'main_measure' must be one of {sorted(ENDPOINT_SPECS)}; "
            f"got {name!r}"
        ) from exc


@dataclass(frozen=True)
class AnalysisSpec:
    """Validated settings for one study analysis.

    ``SyncPipeConfig`` is kept as an older name for compatibility.
    """

    window_size: int = 30
    window_type: str = "rect"
    contrast: Optional[Tuple[str, str]] = None
    fdr_scope: str = "global"
    undefined_policy: str = "gate"
    observation_policy: str = "raise"
    eligibility_policy: str = "raise"
    n_min_dyads: int = 10
    onset_threshold: Union[float, str] = "session_pooled"
    n_permutations: int = 10000
    seed: int = 42
    surrogate_n: int = 1000
    design_threshold: float = 0.5
    design_condition: Optional[str] = None
    primary_endpoint: Optional[str] = None
    primary_modalities: Optional[Tuple[str, ...]] = None
    existence_alpha: float = 0.05
    n_workers: int = 1

    def __post_init__(self) -> None:
        if int(self.window_size) != self.window_size or self.window_size < 2:
            raise ValueError("window_size must be an integer >= 2")
        allowed_windows = {
            "rect", "boxcar", "rectangular", "hann", "hanning", "hamming",
            "triang", "gaussian",
        }
        if self.window_type not in allowed_windows:
            raise ValueError(f"unsupported window_type: {self.window_type!r}")
        if self.contrast is not None:
            contrast = tuple(str(x).strip() for x in self.contrast)
            if len(contrast) != 2 or not all(contrast) or contrast[0] == contrast[1]:
                raise ValueError("contrast must contain two different non-empty conditions")
            object.__setattr__(self, "contrast", contrast)
        if self.fdr_scope not in {"global", "within_modality"}:
            raise ValueError("multiple-test scope must be 'global' or 'within_modality'")
        if self.undefined_policy not in {"flag", "gate"}:
            raise ValueError("missing-result policy must be 'flag' or 'gate'")
        if self.observation_policy not in {"ignore", "warn", "raise"}:
            raise ValueError("unequal-observation policy must be 'ignore', 'warn', or 'raise'")
        if self.eligibility_policy not in {"ignore", "warn", "raise"}:
            raise ValueError("small-sample policy must be 'ignore', 'warn', or 'raise'")
        if self.n_min_dyads < 4:
            raise ValueError("minimum dyads must be >= 4")
        if self.n_permutations < 1 or self.surrogate_n < 1:
            raise ValueError("n_permutations and surrogate_n must be >= 1")
        if isinstance(self.onset_threshold, str):
            if self.onset_threshold != "session_pooled":
                raise ValueError(
                    "onset_threshold string must be 'session_pooled'"
                )
        elif not -1.0 <= float(self.onset_threshold) <= 1.0:
            raise ValueError("onset_threshold must lie in [-1, 1]")
        if not -1.0 <= self.design_threshold <= 1.0:
            raise ValueError("design_threshold must lie in [-1, 1]")
        if not 0.0 < self.existence_alpha < 1.0:
            raise ValueError("significance level must lie in (0, 1)")
        if int(self.n_workers) != self.n_workers or self.n_workers < 1:
            raise ValueError("n_workers must be an integer >= 1")
        if self.primary_endpoint is not None:
            endpoint = str(self.primary_endpoint).strip()
            resolve_endpoint_spec(endpoint)
            object.__setattr__(self, "primary_endpoint", endpoint)
        if self.primary_modalities is not None:
            modalities = tuple(str(x).strip() for x in self.primary_modalities)
            if not modalities or any(not x for x in modalities):
                raise ValueError("main_modalities must contain non-empty labels")
            if len(set(modalities)) != len(modalities):
                raise ValueError("main_modalities must not contain duplicates")
            object.__setattr__(self, "primary_modalities", modalities)
        if self.design_condition is not None:
            value = str(self.design_condition).strip()
            object.__setattr__(self, "design_condition", value or None)

    def resolved_contrast(self) -> Tuple[str, str]:
        if self.contrast is None:
            raise ValueError(
                "contrast is required and must list exactly two "
                "pre-specified conditions, e.g. ['rest', 'task']"
            )
        return self.contrast

    def resolved_endpoint_spec(self) -> EndpointSpec:
        if self.primary_endpoint is None:
            raise ValueError(
                "main_measure is required for the canonical path"
            )
        return resolve_endpoint_spec(self.primary_endpoint)

    def resolved_primary_endpoint(self) -> str:
        return self.resolved_endpoint_spec().name

    def resolved_primary_modalities(self) -> Tuple[str, ...]:
        if not self.primary_modalities:
            raise ValueError(
                "main_modalities is required for the canonical path; "
                "declare the pre-specified modality set explicitly"
            )
        return self.primary_modalities

    def resolved_design_condition(self) -> str:
        return self.design_condition or self.resolved_contrast()[1]

    def modality_specs(self, available_modalities: Tuple[str, ...]) -> Tuple[ModalitySpec, ...]:
        primary = set(self.resolved_primary_modalities())
        return tuple(
            ModalitySpec(label=label, role="primary" if label in primary else "comparator")
            for label in available_modalities
        )

    def to_dict(self) -> dict:
        return asdict(self)


SyncPipeConfig = AnalysisSpec


def analysis_spec_from_mapping(
    mapping: Mapping[str, Any], *, require_declarations: bool = True
) -> AnalysisSpec:
    """Parse a mapping through the one typed AnalysisSpec contract."""
    values = dict(mapping)
    aliases = {
        "main_measure": "primary_endpoint",
        "main_modalities": "primary_modalities",
    }
    for plain_name, internal_name in aliases.items():
        if plain_name in values:
            if internal_name in values:
                raise ValueError(
                    f"use either {plain_name!r} or legacy {internal_name!r}, not both"
                )
            values[internal_name] = values.pop(plain_name)

    allowed = {field.name for field in fields(AnalysisSpec)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown analysis settings: {unknown}")
    if require_declarations:
        missing = [
            key for key in ("contrast", "primary_endpoint", "primary_modalities")
            if key not in values
        ]
        if missing:
            plain = {
                "primary_endpoint": "main_measure",
                "primary_modalities": "main_modalities",
            }.get(missing[0], missing[0])
            raise ValueError(f"analysis setting {plain!r} is required")

    if "contrast" in values:
        raw = values["contrast"]
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError("contrast must be a list/tuple of exactly two labels")
        values["contrast"] = tuple(str(x) for x in raw)
    if "primary_modalities" in values:
        raw = values["primary_modalities"]
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError("main_modalities must be a non-empty list")
        values["primary_modalities"] = tuple(str(x) for x in raw)

    integer_fields = {
        "window_size", "n_min_dyads", "n_permutations", "seed",
        "surrogate_n", "n_workers",
    }
    float_fields = {"design_threshold", "existence_alpha"}
    for name in integer_fields & values.keys():
        values[name] = int(values[name])
    for name in float_fields & values.keys():
        values[name] = float(values[name])
    if "onset_threshold" in values and not isinstance(values["onset_threshold"], str):
        values["onset_threshold"] = float(values["onset_threshold"])
    return AnalysisSpec(**values)


__all__ = [
    "NullSpec", "EndpointSpec", "ModalitySpec", "AnalysisSpec",
    "SyncPipeConfig", "SIGNAL_LEVEL_IAAFT_PEAK", "ENDPOINT_SPECS",
    "resolve_endpoint_spec", "analysis_spec_from_mapping",
]
