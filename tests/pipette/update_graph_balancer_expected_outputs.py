###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

#!/usr/bin/env python3
"""
Add a rxn smile's expected output from graph balancer to the test data files

Modes:
  - default: fill missing expected_output fields or missing snapshot files
  - --redo-all: refresh every entry
  - --rxn-smis '...' ['...']: refresh specific reactions, adding new entries if needed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
import flask_tools.pipette.graph_rxn_mapper


TESTS_DIR = Path(__file__).resolve().parent
DATA_FILE = TESTS_DIR / "data" / "atom_map_rxns.jsonl"
EXPECTED_DIR = TESTS_DIR / "expected_atom_map_res"
SCRIPT_DIR = Path(str(flask_tools.pipette.graph_rxn_mapper.__file__)).resolve().parent
MAPPER_SCRIPT = SCRIPT_DIR / "subtractive_reaction_mapper_v3.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate missing expected_output fields, refresh all snapshots, "
            "or refresh selected rxn_smiles entries."
        )
    )
    parser.add_argument(
        "--redo-all",
        action="store_true",
        help="Refresh every expected output in tests/data/rxns.jsonl.",
    )
    parser.add_argument(
        "--rxn-smis",
        "--rxn-smiles",
        nargs="*",
        default=None,
        dest="rxn_smis",
        help="Refresh only these reactions; new reactions are appended to tests/data/rxns.jsonl.",
    )
    return parser.parse_args()


def load_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for raw_line in DATA_FILE.read_text().splitlines():
        if raw_line.strip():
            entries.append(json.loads(raw_line))
    return entries


def write_entries(entries: list[dict[str, object]]) -> None:
    serialized = "\n".join(
        json.dumps(entry, separators=(",", ":")) for entry in entries
    )
    DATA_FILE.write_text(f"{serialized}\n")


def build_snapshot_path(rxn_smiles: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", rxn_smiles.lower()).strip("_")
    slug = slug or "reaction"
    digest = hashlib.sha256(rxn_smiles.encode("utf-8")).hexdigest()[:8]
    return f"{slug[:60]}__{digest}.json"


def build_entry_id(rxn_smiles: str) -> str:
    snapshot_name = Path(build_snapshot_path(rxn_smiles)).stem
    return snapshot_name.replace("__", "_")


def run_mapper(rxn_smiles: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(MAPPER_SCRIPT), rxn_smiles],
        cwd=SCRIPT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def should_refresh(entry: dict[str, object], args: argparse.Namespace) -> bool:
    rxn_smiles = str(entry["rxn_smiles"])
    expected_output = entry.get("expected_output")
    if args.redo_all:
        return True
    if args.rxn_smis is not None:
        return rxn_smiles in args.rxn_smis
    if not expected_output:
        return True
    return not (EXPECTED_DIR / str(expected_output)).exists()


def append_missing_entries(
    entries: list[dict[str, object]], rxn_smis: list[str] | None
) -> set[str]:
    if not rxn_smis:
        return set()

    existing = {str(entry["rxn_smiles"]) for entry in entries}
    added: set[str] = set()
    for rxn_smiles in rxn_smis:
        if rxn_smiles in existing:
            continue
        entries.append(
            {
                "id": build_entry_id(rxn_smiles),
                "rxn_smiles": rxn_smiles,
                "expected_output": build_snapshot_path(rxn_smiles),
            }
        )
        existing.add(rxn_smiles)
        added.add(rxn_smiles)
    return added


def main() -> int:
    args = parse_args()
    entries = load_entries()
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)

    added_entries = append_missing_entries(entries, args.rxn_smis)

    for entry in entries:
        if not should_refresh(entry, args):
            continue

        rxn_smiles = str(entry["rxn_smiles"])
        expected_output = str(
            entry.get("expected_output") or build_snapshot_path(rxn_smiles)
        )
        entry["expected_output"] = expected_output

        snapshot_path = EXPECTED_DIR / expected_output
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(run_mapper(rxn_smiles))

        if rxn_smiles in added_entries:
            print(f"added {entry['id']} {rxn_smiles} -> {expected_output}")
        else:
            print(f"updated {entry['id']} {rxn_smiles} -> {expected_output}")

    write_entries(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
