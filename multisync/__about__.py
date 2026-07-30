"""Version information."""

# Distribution release version (the pip-installable package).
__version__ = "1.0.0"

# Gate 2 — three-way version contract. External consumers pin and bump these
# independently; they must NOT be collapsed back into a single __version__.
PACKAGE_VERSION = __version__        # distribution release
ANALYSIS_SCHEMA_VERSION = "0.3.0"    # serialized WCC/feature JSON STRUCTURE
CONFIG_SCHEMA_VERSION = "1.0.0"     # manifest + SyncPipeConfig contract
