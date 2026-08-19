"""Version information."""

# Distribution release version (the pip-installable package).
__version__ = "2.0.0"

# Gate 2 — three-way version contract. External consumers pin and bump these
# independently; they must NOT be collapsed back into a single __version__.
PACKAGE_VERSION = __version__        # distribution release
ANALYSIS_SCHEMA_VERSION = "0.3.0"       # viewer WCC/feature JSON structure
CONFIG_SCHEMA_VERSION = "2.0.0"         # AnalysisSpec/TOML contract
MANIFEST_SCHEMA_VERSION = "2.0.0"       # canonical manifest columns
PREPROCESSING_SCHEMA_VERSION = "1.0.0"  # preprocessing provenance document
EVIDENCE_SCHEMA_VERSION = "1.0.0"       # typed E0-E5 evidence graph
