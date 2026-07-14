#!/usr/bin/env python3
"""Benchmark subtractive_reaction_mapper_v3 against mapped RDF reactions."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from rdfreader import RDFParser
from rdkit import Chem, RDLogger
from tqdm import tqdm

from subtractive_reaction_mapper_v3 import MapperConfig, subtractive_map_reaction


RDLogger.DisableLog("rdApp.warning")


@dataclass
class ReactionTask:
    index: int
    rdf_line: Optional[int]
    smiles: str


@dataclass
class BenchmarkRecord:
    index: int
    rdf_line: Optional[int]
    source_smiles: str
    unmapped_smiles: str
    expected_smiles: str
    predicted_smiles: str
    expected_normalized: str
    predicted_normalized: str
    expected_reactants_normalized: str
    predicted_reactants_normalized: str
    expected_products_normalized: str
    predicted_products_normalized: str
    reactants_matched: bool
    products_matched: bool
    matched: bool
    mapper_status: str
    elapsed_seconds: float
    topology_counts: Dict[str, Any]
    error: Optional[str] = None


def split_reaction_smiles(reaction_smiles: str) -> Tuple[str, str, str]:
    parts = reaction_smiles.strip().split(">")
    if len(parts) != 3:
        raise ValueError(
            f"Expected reaction SMILES with three '>'-separated parts: {reaction_smiles!r}"
        )
    return parts[0], parts[1], parts[2]


def mol_from_side(side: str) -> Chem.Mol:
    if not side:
        return Chem.Mol()
    mol = Chem.MolFromSmiles(side, sanitize=True)
    if mol is None:
        raise ValueError(f"Could not parse reaction side: {side!r}")
    return mol


def clear_atom_maps_from_side(side: str) -> str:
    mol = mol_from_side(side)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return canonical_side_smiles(mol)


def clear_atom_maps_from_reaction(
    reaction_smiles: str, keep_agents: bool = False
) -> str:
    reactants, agents, products = split_reaction_smiles(reaction_smiles)
    cleared_reactants = clear_atom_maps_from_side(reactants)
    cleared_products = clear_atom_maps_from_side(products)
    if keep_agents:
        cleared_agents = clear_atom_maps_from_side(agents)
        return f"{cleared_reactants}>{cleared_agents}>{cleared_products}"
    return f"{cleared_reactants}>>{cleared_products}"


def reaction_without_agents(reaction_smiles: str) -> str:
    reactants, _agents, products = split_reaction_smiles(reaction_smiles)
    return f"{reactants}>>{products}"


def map_numbers_on_side(mol: Chem.Mol) -> List[int]:
    return sorted(
        {atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0}
    )


def canonical_atom_order_for_map_assignment(mol: Chem.Mol) -> List[int]:
    """Return product atoms in map-independent canonical fragment order."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)

    ranks = list(Chem.CanonicalRankAtoms(copy, breakTies=True))
    fragments = []
    for atoms in Chem.GetMolFrags(copy, asMols=False, sanitizeFrags=True):
        atom_list = list(atoms)
        fragment_smiles = Chem.MolFragmentToSmiles(
            copy,
            atomsToUse=atom_list,
            canonical=True,
            isomericSmiles=True,
        )
        fragments.append(
            (
                fragment_smiles,
                len(atom_list),
                tuple(sorted(ranks[atom_idx] for atom_idx in atom_list)),
                atom_list,
            )
        )

    ordered_atoms: List[int] = []
    for _fragment_smiles, _size, _rank_key, atom_list in sorted(fragments):
        ordered_atoms.extend(
            sorted(atom_list, key=lambda atom_idx: (ranks[atom_idx], atom_idx))
        )
    return ordered_atoms


def canonical_side_smiles(mol: Chem.Mol) -> str:
    """Canonicalize a reaction side as a sorted multiset of mapped fragments."""
    if mol.GetNumAtoms() == 0:
        return ""
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    smiles = [
        Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        for fragment in fragments
    ]
    return ".".join(sorted(smiles))


def normalize_mapped_reaction_parts(reaction_smiles: str) -> Tuple[str, str, str]:
    """Return normalized reactant, agent, and product sides.

    Atom maps are assigned in canonical product-atom order.  The same old map
    number receives the same new number on both sides, so only the numbering
    scheme changes, not the mapping relationship.  Each side is then rendered as
    sorted canonical fragments so reactant/product order cannot affect accuracy.
    """
    reactants, agents, products = split_reaction_smiles(reaction_smiles)
    reactant_mol = mol_from_side(reactants)
    agent_mol = mol_from_side(agents)
    product_mol = mol_from_side(products)

    old_to_new: Dict[int, int] = {}
    for atom_idx in canonical_atom_order_for_map_assignment(product_mol):
        old_map = product_mol.GetAtomWithIdx(atom_idx).GetAtomMapNum()
        if old_map > 0 and old_map not in old_to_new:
            old_to_new[old_map] = len(old_to_new) + 1

    # Include reactant-only maps after product maps.  They usually indicate an
    # incomplete mapping, but keeping them deterministic makes debug output sane.
    for mol in (reactant_mol, agent_mol):
        for old_map in map_numbers_on_side(mol):
            if old_map not in old_to_new:
                old_to_new[old_map] = len(old_to_new) + 1

    for mol in (reactant_mol, agent_mol, product_mol):
        for atom in mol.GetAtoms():
            old_map = atom.GetAtomMapNum()
            atom.SetAtomMapNum(old_to_new.get(old_map, 0))

    return (
        canonical_side_smiles(reactant_mol),
        canonical_side_smiles(agent_mol),
        canonical_side_smiles(product_mol),
    )


def renumber_atom_maps_deterministically(reaction_smiles: str) -> str:
    reactants, agents, products = normalize_mapped_reaction_parts(reaction_smiles)
    if agents:
        return f"{reactants}>{agents}>{products}"
    return f"{reactants}>>{products}"


def reaction_map_counts(reaction_smiles: str) -> Dict[str, int]:
    reactants, agents, products = split_reaction_smiles(reaction_smiles)
    return {
        "reactant_maps": len(map_numbers_on_side(mol_from_side(reactants))),
        "agent_maps": len(map_numbers_on_side(mol_from_side(agents))),
        "product_maps": len(map_numbers_on_side(mol_from_side(products))),
    }


def iter_reactions(rdf_file_name: str) -> Iterable[Tuple[int, Any]]:
    with open(rdf_file_name, "r") as rdf_file:
        rdfreader = RDFParser(
            rdf_file,
            except_on_invalid_molecule=False,
            except_on_invalid_reaction=False,
        )
        for index, rxn in enumerate(rdfreader, start=1):
            yield index, rxn


def build_mapper_config(args: argparse.Namespace) -> MapperConfig:
    return MapperConfig(
        selector=args.selector,
        max_copies=args.max_copies,
        min_fragment_atoms=args.min_fragment_atoms,
        max_fragment_atoms=args.max_fragment_atoms,
        max_fragments_per_reactant=args.max_fragments_per_reactant,
        max_matches_per_fragment=args.max_matches_per_fragment,
        max_base_candidates_per_reactant=args.max_base_candidates_per_reactant,
        include_rdkit_mcs_candidates=not args.no_rdkit_mcs,
        compare_bond_order=not args.ignore_bond_order,
        respect_atom_maps=False,
        broken_bond_environment_penalty=args.broken_bond_environment_penalty,
        bond_environment_objective=args.bond_environment_objective,
        bond_environment_rank_tolerance=args.bond_environment_rank_tolerance,
        stable_single_bond_break_penalty=args.stable_single_bond_break_penalty,
        unsaturated_endpoint_break_credit=args.unsaturated_endpoint_break_credit,
        ring_bond_break_penalty=args.ring_bond_break_penalty,
        max_broken_bond_pair_penalty_terms=args.max_broken_bond_pair_penalty_terms,
    )


def run_one(
    task: ReactionTask, config: MapperConfig, keep_agents: bool = False
) -> BenchmarkRecord:
    source_smiles = task.smiles
    unmapped_smiles = clear_atom_maps_from_reaction(
        source_smiles, keep_agents=keep_agents
    )
    expected_smiles = (
        source_smiles if keep_agents else reaction_without_agents(source_smiles)
    )
    expected_reactants, expected_agents, expected_products = (
        normalize_mapped_reaction_parts(expected_smiles)
    )
    expected_normalized = (
        f"{expected_reactants}>{expected_agents}>{expected_products}"
        if expected_agents
        else f"{expected_reactants}>>{expected_products}"
    )

    started = time.perf_counter()
    result = subtractive_map_reaction(unmapped_smiles, config)
    elapsed = time.perf_counter() - started

    predicted_smiles = result.atom_mapped_reaction_smiles()
    predicted_reactants, predicted_agents, predicted_products = (
        normalize_mapped_reaction_parts(predicted_smiles)
    )
    predicted_normalized = (
        f"{predicted_reactants}>{predicted_agents}>{predicted_products}"
        if predicted_agents
        else f"{predicted_reactants}>>{predicted_products}"
    )
    reactants_matched = expected_reactants == predicted_reactants
    products_matched = expected_products == predicted_products
    agents_matched = expected_agents == predicted_agents

    return BenchmarkRecord(
        index=task.index,
        rdf_line=task.rdf_line,
        source_smiles=source_smiles,
        unmapped_smiles=unmapped_smiles,
        expected_smiles=expected_smiles,
        predicted_smiles=predicted_smiles,
        expected_normalized=expected_normalized,
        predicted_normalized=predicted_normalized,
        expected_reactants_normalized=expected_reactants,
        predicted_reactants_normalized=predicted_reactants,
        expected_products_normalized=expected_products,
        predicted_products_normalized=predicted_products,
        reactants_matched=reactants_matched,
        products_matched=products_matched,
        matched=reactants_matched and products_matched and agents_matched,
        mapper_status=result.status,
        elapsed_seconds=elapsed,
        topology_counts=result.diagnostics.get("topology_counts", {}),
    )


def error_record(task: ReactionTask, exc: BaseException) -> BenchmarkRecord:
    return BenchmarkRecord(
        index=task.index,
        rdf_line=task.rdf_line,
        source_smiles=task.smiles,
        unmapped_smiles="",
        expected_smiles=task.smiles,
        predicted_smiles="",
        expected_normalized="",
        predicted_normalized="",
        expected_reactants_normalized="",
        predicted_reactants_normalized="",
        expected_products_normalized="",
        predicted_products_normalized="",
        reactants_matched=False,
        products_matched=False,
        matched=False,
        mapper_status="error",
        elapsed_seconds=0.0,
        topology_counts={},
        error=f"{type(exc).__name__}: {exc}",
    )


def format_record(record: BenchmarkRecord, debug: bool = False) -> str:
    lines = []
    status = "PASS" if record.matched else "FAIL"
    location = f"line {record.rdf_line}" if record.rdf_line is not None else "line ?"
    lines.append(
        f"[{status}] #{record.index} ({location}) {record.elapsed_seconds:.3f}s status={record.mapper_status}"
    )
    if record.error:
        lines.append(f"  error: {record.error}")
    if debug or not record.matched:
        side_status = []
        side_status.append(f"reactants={'ok' if record.reactants_matched else 'diff'}")
        side_status.append(f"products={'ok' if record.products_matched else 'diff'}")
        lines.append(f"  sides:     {', '.join(side_status)}")
        lines.append(f"  unmapped:  {record.unmapped_smiles}")
        if debug or not record.reactants_matched:
            lines.append(
                f"  expected reactants:  {record.expected_reactants_normalized}"
            )
            lines.append(
                f"  predicted reactants: {record.predicted_reactants_normalized}"
            )
        if debug or not record.products_matched:
            lines.append(
                f"  expected products:   {record.expected_products_normalized}"
            )
            lines.append(
                f"  predicted products:  {record.predicted_products_normalized}"
            )
        elif not debug and not record.reactants_matched:
            lines.append(f"  products:   {record.expected_products_normalized}")

    return "\n".join(lines)


def print_record(record: BenchmarkRecord, debug: bool = False) -> None:
    print(format_record(record, debug=debug))


def default_worker_count() -> int:
    if hasattr(os, "process_cpu_count"):
        cpu_count = os.process_cpu_count()
    else:
        cpu_count = os.cpu_count()
    return max(1, cpu_count or 1)


def collect_tasks(args: argparse.Namespace) -> Tuple[List[ReactionTask], int]:
    tasks: List[ReactionTask] = []
    skipped = 0
    valid_seen = 0

    for index, rxn in iter_reactions(args.rdf_file):
        if index < args.start:
            continue
        if rxn is None:
            skipped += 1
            continue

        valid_seen += 1
        if args.limit is not None and valid_seen > args.limit:
            break

        try:
            source_counts = reaction_map_counts(rxn.smiles)
        except Exception:
            skipped += 1
            continue
        if source_counts["reactant_maps"] == 0 or source_counts["product_maps"] == 0:
            skipped += 1
            continue

        tasks.append(
            ReactionTask(
                index=index,
                rdf_line=getattr(rxn, "lineno", None),
                smiles=rxn.smiles,
            )
        )

    return tasks, skipped


def update_progress_postfix(
    progress: tqdm, records: Sequence[BenchmarkRecord], errors: int
) -> None:
    completed = len(records)
    matched = sum(1 for record in records if record.matched)
    mismatched = max(0, completed - matched - errors)
    accuracy = matched / completed if completed else 0.0
    progress.set_postfix(
        {"acc": f"{accuracy:.1%}", "pass": matched, "fail": mismatched, "err": errors},
        refresh=False,
    )


def print_summary(
    records: Sequence[BenchmarkRecord],
    skipped: int,
    errors: int,
    started: float,
    workers: int,
) -> None:
    total_elapsed = time.perf_counter() - started
    completed = len(records)
    matched = sum(1 for record in records if record.matched)
    mismatched = completed - matched
    reactant_matches = sum(1 for record in records if record.reactants_matched)
    product_matches = sum(1 for record in records if record.products_matched)
    accuracy = matched / completed if completed else 0.0
    statuses = Counter(record.mapper_status for record in records)

    print("\nSummary")
    print(f"  completed:  {completed}")
    print(f"  matched:    {matched}")
    print(f"  mismatched: {mismatched}")
    print(f"  accuracy:   {accuracy:.1%}")
    if completed:
        print(
            f"  reactants:  {reactant_matches}/{completed} ({reactant_matches / completed:.1%})"
        )
        print(
            f"  products:   {product_matches}/{completed} ({product_matches / completed:.1%})"
        )
    print(f"  skipped:    {skipped}")
    print(f"  errors:     {errors}")
    print(f"  workers:    {workers}")
    print(f"  elapsed:    {total_elapsed:.3f}s")
    if completed:
        print(
            f"  avg/rxn:    {sum(r.elapsed_seconds for r in records) / completed:.3f}s"
        )
    if statuses:
        print(f"  statuses:   {dict(sorted(statuses.items()))}")


def write_json_report(
    path: str,
    records: Sequence[BenchmarkRecord],
    skipped: int,
    errors: int,
    workers: int,
) -> None:
    payload = {
        "summary": {
            "completed": len(records),
            "matched": sum(1 for record in records if record.matched),
            "mismatched": sum(1 for record in records if not record.matched),
            "reactant_matches": sum(
                1 for record in records if record.reactants_matched
            ),
            "product_matches": sum(1 for record in records if record.products_matched),
            "skipped": skipped,
            "errors": errors,
            "workers": workers,
        },
        "records": [asdict(record) for record in records],
    }
    with open(path, "w") as out:
        json.dump(payload, out, indent=2, sort_keys=True)
        out.write("\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "rdf_file", nargs="?", default="reactions.rdf", help="RDF file to benchmark."
    )
    parser.add_argument(
        "--limit", type=int, help="Stop after this many valid RDF reactions."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="One-based RDF reaction index to start from.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=default_worker_count(),
        help="Worker threads to use.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print expected/predicted mappings for every completed reaction.",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero when any completed reaction mismatches.",
    )
    parser.add_argument(
        "--json-report", help="Write a detailed JSON report to this path."
    )
    parser.add_argument(
        "--keep-agents",
        action="store_true",
        help="Keep the RDF agent field in mapper input.",
    )
    parser.add_argument("--selector", choices=["ilp", "greedy"], default="ilp")
    parser.add_argument("--max-copies", type=int, default=3)
    parser.add_argument("--min-fragment-atoms", type=int, default=1)
    parser.add_argument("--max-fragment-atoms", type=int, default=8)
    parser.add_argument("--max-fragments-per-reactant", type=int, default=2500)
    parser.add_argument("--max-matches-per-fragment", type=int, default=128)
    parser.add_argument("--max-base-candidates-per-reactant", type=int, default=6000)
    parser.add_argument("--broken-bond-environment-penalty", type=float, default=1.0)
    parser.add_argument(
        "--bond-environment-objective",
        choices=["off", "integrated", "rerank"],
        default="off",
    )
    parser.add_argument("--bond-environment-rank-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--stable-single-bond-break-penalty", type=float, default=1.0)
    parser.add_argument("--unsaturated-endpoint-break-credit", type=float, default=0.75)
    parser.add_argument("--ring-bond-break-penalty", type=float, default=2.0)
    parser.add_argument("--max-broken-bond-pair-penalty-terms", type=int, default=25000)
    parser.add_argument("--no-rdkit-mcs", action="store_true")
    parser.add_argument("--ignore-bond-order", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    print(f"Running with {args.workers} threads")

    config = build_mapper_config(args)
    tasks, skipped = collect_tasks(args)
    records: List[BenchmarkRecord] = []
    errors = 0
    started = time.perf_counter()

    with tqdm(total=len(tasks), unit="rxn", desc="Benchmarking") as progress:
        if tasks:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(run_one, task, config, args.keep_agents): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        errors += 1
                        record = error_record(task, exc)

                    records.append(record)
                    if args.debug or not record.matched or record.error:
                        tqdm.write(format_record(record, debug=args.debug))
                    update_progress_postfix(progress, records, errors)
                    progress.update(1)

    records.sort(key=lambda record: record.index)
    print_summary(
        records, skipped=skipped, errors=errors, started=started, workers=args.workers
    )

    if args.json_report:
        write_json_report(
            args.json_report,
            records,
            skipped=skipped,
            errors=errors,
            workers=args.workers,
        )
        print(f"  json:       {args.json_report}")

    if errors:
        return 1
    if args.fail_on_mismatch and any(not record.matched for record in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
