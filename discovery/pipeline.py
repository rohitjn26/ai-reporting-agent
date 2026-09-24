"""Orchestrate discovery and shape the result for review + graph rendering.

profile -> grain -> joins -> buckets + a node/edge graph. Deterministic; no LLM.
See docs/SCHEMA_DISCOVERY.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from .grain import GrainResult, detect_all
from .joins import JoinCandidate, discover_joins
from .profile import TableProfile, load_folder, profile_all

# a data-only match with no name signal is suspicious (e.g. quantity ⊆ id)
ACCEPT_MIN_CONFIDENCE = 0.75


@dataclass
class DiscoveryResult:
    profiles: dict[str, TableProfile]
    grains: dict[str, GrainResult]
    accepted: list[JoinCandidate]
    uncertain: list[JoinCandidate]

    def bridges(self) -> list[str]:
        """Tables whose whole grain is made of foreign-key columns = junctions."""
        fk_cols_by_table: dict[str, set[str]] = {}
        for j in self.accepted + self.uncertain:
            fk_cols_by_table.setdefault(j.fk_table, set()).add(j.fk_column)
        out = []
        for t, g in self.grains.items():
            if g.key and len(g.key) >= 2 and set(g.key) <= fk_cols_by_table.get(t, set()):
                out.append(t)
        return out

    def graph(self) -> dict:
        """Node/edge graph for the UI (Cytoscape-friendly)."""
        bridges = set(self.bridges())
        nodes = []
        for t, p in self.profiles.items():
            g = self.grains[t]
            nodes.append({
                "id": t,
                "rows": p.rows,
                "grain": g.key,
                "grain_kind": g.kind,
                "role": "bridge" if t in bridges else None,
            })
        edges = []
        for status, joins in (("accepted", self.accepted), ("uncertain", self.uncertain)):
            for j in joins:
                edges.append({
                    "source": j.fk_table,
                    "target": j.pk_table,
                    "column": j.fk_column,
                    "relationship": j.relationship,
                    "confidence": round(j.confidence, 3),
                    "containment": round(j.containment, 4),
                    "status": status,
                })
        return {"nodes": nodes, "edges": edges}

    def to_dict(self) -> dict:
        return {
            "tables": {t: p.to_dict() for t, p in self.profiles.items()},
            "grains": {t: g.to_dict() for t, g in self.grains.items()},
            "joins": {
                "accepted": [j.to_dict() for j in self.accepted],
                "uncertain": [j.to_dict() for j in self.uncertain],
            },
            "bridges": self.bridges(),
            "graph": self.graph(),
        }


def run_discovery(files: list[str | Path],
                  con: duckdb.DuckDBPyConnection | None = None) -> DiscoveryResult:
    con, tables = load_folder(files, con)
    profiles = profile_all(con, tables)
    grains = detect_all(con, profiles)
    candidates = discover_joins(con, profiles, grains)

    accepted, uncertain = [], []
    for j in candidates:
        if j.name_signal != "none" and j.confidence >= ACCEPT_MIN_CONFIDENCE:
            accepted.append(j)
        else:
            uncertain.append(j)  # data-only or low-confidence -> human decides

    # a FK column with a confident accepted join is "explained" — drop its
    # no-name-signal coincidental matches to other tables.
    explained = {(j.fk_table, j.fk_column) for j in accepted}
    uncertain = [j for j in uncertain if (j.fk_table, j.fk_column) not in explained]

    return DiscoveryResult(profiles, grains, accepted, uncertain)
