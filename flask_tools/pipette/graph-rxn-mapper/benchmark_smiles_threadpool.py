#!/usr/bin/env python3
"""Compare sequential and threaded RDKit workloads on 1000 preloaded SMILES.

python3 benchmark_smiles_threadpool.py --operation pure_python --limit 100 --repeats 2 --python-iterations 3000

Operation: Pure Python char-arithmetic loop (3000 iterations)
Preloaded SMILES strings outside the timed section.
Loaded 100 items outside the timed section.
Repeats: 2
Sequential mean: 0.675330s
ThreadPoolExecutor(2) mean: 0.681646s
Speedup: 0.991x

RDKit embedding (CPU heavy) shows speedup

python3 benchmark_smiles_threadpool.py --operation embed --limit 50 --repeats 1

Sequential mean: 0.243591s
ThreadPoolExecutor(2) mean: 0.133003s
Speedup: 1.831x

Rdkit MolToSmiles is not that CPU heavy, so no improvement
python3 benchmark_smiles_threadpool.py --operation parse --limit 50 --repeats 1

"""


from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, TypeVar

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")

SMILES_FIELDS = (
    "reactants",
    "products",
    "agents",
    "solvents",
    "catalysts",
    "atmospheres",
)

WorkItem = TypeVar("WorkItem")
WorkResult = TypeVar("WorkResult")


def iter_smiles(jsonl_path: Path) -> Iterable[str]:
    with jsonl_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            for field in SMILES_FIELDS:
                for smiles in record.get(field, []):
                    if smiles:
                        yield smiles


def load_smiles(jsonl_path: Path, limit: int) -> List[str]:
    smiles = []
    for item in iter_smiles(jsonl_path):
        smiles.append(item)
        if len(smiles) == limit:
            break
    if len(smiles) < limit:
        raise ValueError(
            f"Requested {limit} SMILES, found only {len(smiles)} in {jsonl_path}."
        )
    return smiles


def parse_one(smiles: str) -> Chem.Mol | None:
    return Chem.MolFromSmiles(smiles, sanitize=True)


def build_embed_templates(smiles_list: Sequence[str]) -> List[bytes]:
    templates = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            raise ValueError(f"Could not parse SMILES for embedding: {smiles!r}")
        templates.append(Chem.AddHs(mol).ToBinary())
    return templates


def embed_one(mol_binary: bytes) -> int:
    mol = Chem.Mol(mol_binary)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    return AllChem.EmbedMolecule(mol, params)


def pure_python_cpu_one(smiles: str, iterations: int) -> int:
    acc = 0
    for outer in range(iterations):
        for index, char in enumerate(smiles):
            value = ord(char) + index + outer
            acc = ((acc * 131) + (value * value) + outer) % 1_000_000_007
    return acc


def run_sequential(
    items: Sequence[WorkItem], worker_fn: Callable[[WorkItem], WorkResult]
) -> List[WorkResult]:
    return [worker_fn(item) for item in items]


def run_threaded(
    items: Sequence[WorkItem],
    worker_fn: Callable[[WorkItem], WorkResult],
    workers: int,
) -> List[WorkResult]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker_fn, items))


def time_run(fn, *args) -> tuple[float, List[WorkResult]]:
    start = time.perf_counter()
    results = fn(*args)
    elapsed = time.perf_counter() - start
    return elapsed, results


def benchmark(
    items: Sequence[WorkItem],
    worker_fn: Callable[[WorkItem], WorkResult],
    repeats: int,
    workers: int,
    workload_name: str,
    preparation_note: str,
) -> None:
    sequential_times = []
    threaded_times = []

    # Warm up RDKit paths outside the timed section.
    run_sequential(items[:10], worker_fn)
    run_threaded(items[:10], worker_fn, workers)

    for _ in range(repeats):
        seq_elapsed, seq_results = time_run(run_sequential, items, worker_fn)
        thr_elapsed, thr_results = time_run(run_threaded, items, worker_fn, workers)

        if len(seq_results) != len(items) or len(thr_results) != len(items):
            raise RuntimeError(
                "One of the benchmark runs returned the wrong number of results."
            )

        sequential_times.append(seq_elapsed)
        threaded_times.append(thr_elapsed)

    sequential_mean = statistics.mean(sequential_times)
    threaded_mean = statistics.mean(threaded_times)
    speedup = sequential_mean / threaded_mean if threaded_mean else float("inf")

    print(f"Operation: {workload_name}")
    print(preparation_note)
    print(f"Loaded {len(items)} items outside the timed section.")
    print(f"Repeats: {repeats}")
    print(f"Sequential mean: {sequential_mean:.6f}s")
    print(f"ThreadPoolExecutor({workers}) mean: {threaded_mean:.6f}s")
    print(f"Speedup: {speedup:.3f}x")
    print(f"Sequential times: {[round(t, 6) for t in sequential_times]}")
    print(f"Threaded times: {[round(t, 6) for t in threaded_times]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("some-rxns.jsonl"),
        help="JSONL file containing reaction records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of SMILES strings to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of timed runs per strategy.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of ThreadPoolExecutor workers.",
    )
    parser.add_argument(
        "--operation",
        choices=("parse", "embed", "pure_python"),
        default="parse",
        help="RDKit workload to benchmark.",
    )
    parser.add_argument(
        "--python-iterations",
        type=int,
        default=2000,
        help="Outer-loop iterations for the pure Python CPU benchmark.",
    )
    args = parser.parse_args()

    smiles_list = load_smiles(args.input, args.limit)
    if args.operation == "parse":
        benchmark(
            smiles_list,
            worker_fn=parse_one,
            repeats=args.repeats,
            workers=args.workers,
            workload_name="MolFromSmiles",
            preparation_note="Preloaded SMILES strings outside the timed section.",
        )
        return

    if args.operation == "pure_python":
        benchmark(
            smiles_list,
            worker_fn=partial(pure_python_cpu_one, iterations=args.python_iterations),
            repeats=args.repeats,
            workers=args.workers,
            workload_name=f"Pure Python char-arithmetic loop ({args.python_iterations} iterations)",
            preparation_note="Preloaded SMILES strings outside the timed section.",
        )
        return

    embed_templates = build_embed_templates(smiles_list)
    benchmark(
        embed_templates,
        worker_fn=embed_one,
        repeats=args.repeats,
        workers=args.workers,
        workload_name="EmbedMolecule(ETKDGv3)",
        preparation_note="Preloaded hydrogenated molecule templates outside the timed section.",
    )


if __name__ == "__main__":
    main()
