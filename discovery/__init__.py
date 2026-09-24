"""Schema discovery: profile CSVs in DuckDB, detect grain, discover joins.

Deterministic first pass over raw data. No LLM. See docs/SCHEMA_DISCOVERY.md.
"""

from .pipeline import run_discovery, DiscoveryResult

__all__ = ["run_discovery", "DiscoveryResult"]
