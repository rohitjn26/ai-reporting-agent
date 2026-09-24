"""CLI: point at CSVs (a folder or explicit files), print discovered schema.

    python -m discovery <folder-or-files...> [--json]

With a folder, every *.csv inside is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import run_discovery


def _collect_files(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            files.extend(sorted(path.glob("*.csv")))
        elif path.exists():
            files.append(path)
        else:
            print(f"skip (not found): {p}", file=sys.stderr)
    return files


def _print_report(result) -> None:
    d = result.to_dict()
    print("\n=== Tables & grain ===")
    for t, tbl in d["tables"].items():
        g = d["grains"][t]
        key = "+".join(g["key"]) if g["key"] else "?"
        bridge = "  [bridge]" if t in d["bridges"] else ""
        print(f"  {t:<14} rows={tbl['rows']:<7} grain={key} ({g['kind']}){bridge}")

    print("\n=== Accepted joins ===")
    for j in d["joins"]["accepted"]:
        print(f"  {j['fk']['table']}.{j['fk']['column']} -> "
              f"{j['pk']['table']}.{j['pk']['column']}  "
              f"{j['relationship']}  conf={j['confidence']} "
              f"containment={j['containment']} signal={j['name_signal']}")

    if d["joins"]["uncertain"]:
        print("\n=== Uncertain (needs review) ===")
        for j in d["joins"]["uncertain"]:
            print(f"  {j['fk']['table']}.{j['fk']['column']} -> "
                  f"{j['pk']['table']}.{j['pk']['column']}  "
                  f"conf={j['confidence']} containment={j['containment']} "
                  f"signal={j['name_signal']}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(prog="discovery")
    ap.add_argument("paths", nargs="+", help="folder(s) or CSV file(s)")
    ap.add_argument("--json", action="store_true", help="emit full result as JSON")
    args = ap.parse_args()

    files = _collect_files(args.paths)
    if not files:
        print("No CSV files found.", file=sys.stderr)
        sys.exit(1)

    result = run_discovery(files)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        _print_report(result)


if __name__ == "__main__":
    main()
