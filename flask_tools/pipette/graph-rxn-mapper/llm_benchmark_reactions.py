#!/usr/bin/env python3
"""Benchmark an LLM atom mapper against mapped RDF reactions."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger
from tqdm import tqdm

from benchmark_reactions import (
    BenchmarkRecord,
    ReactionTask,
    clear_atom_maps_from_reaction,
    collect_tasks,
    default_worker_count,
    format_record,
    mol_from_side,
    normalize_mapped_reaction_parts,
    print_summary,
    reaction_without_agents,
    split_reaction_smiles,
    write_json_report,
)


RDLogger.DisableLog("rdApp.warning")

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_SYSTEM_PROMPT = PROMPT_DIR / "atom_mapping_system.md"
DEFAULT_USER_PROMPT = PROMPT_DIR / "atom_mapping_user.md"
DEFAULT_SKILL_PROMPT = PROMPT_DIR / "atom_mapping_skill.md"


@dataclass
class LLMRecord:
    benchmark: BenchmarkRecord
    model: str
    raw_response: str
    reasoning_summary: str = ""
    confidence: Optional[float] = None


def load_text(path: str) -> str:
    return Path(path).read_text()


def build_system_prompt(args: argparse.Namespace) -> str:
    system_prompt = load_text(args.system_prompt)
    if args.use_skill_prompt and args.skill_prompt:
        skill_prompt = load_text(args.skill_prompt)
        system_prompt = f"{system_prompt}\n\nAdditional atom-mapping skill instructions:\n{skill_prompt}"
    return system_prompt


def side_graph_json(side: str) -> str:
    mol = mol_from_side(side)
    atoms = []
    atom_to_fragment: Dict[int, int] = {}
    for frag_id, atom_ids in enumerate(
        Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=True)
    ):
        for atom_id in atom_ids:
            atom_to_fragment[int(atom_id)] = frag_id

    for atom in mol.GetAtoms():
        atoms.append(
            {
                "id": atom.GetIdx(),
                "fragment": atom_to_fragment.get(atom.GetIdx(), 0),
                "element": atom.GetSymbol(),
                "atomic_num": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "is_aromatic": atom.GetIsAromatic(),
                "isotope": atom.GetIsotope(),
                "neighbors": sorted(n.GetIdx() for n in atom.GetNeighbors()),
            }
        )

    bonds = []
    for bond in mol.GetBonds():
        bonds.append(
            {
                "begin": bond.GetBeginAtomIdx(),
                "end": bond.GetEndAtomIdx(),
                "order": float(bond.GetBondTypeAsDouble()),
                "is_aromatic": bond.GetIsAromatic(),
                "in_ring": bond.IsInRing(),
            }
        )

    return json.dumps(
        {
            "atom_count": mol.GetNumAtoms(),
            "atoms": atoms,
            "bonds": bonds,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "product_to_reactant": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "product_atom": {"type": "integer"},
                        "reactant_atom": {"type": "integer"},
                    },
                    "required": ["product_atom", "reactant_atom"],
                },
            },
            "confidence": {"type": "number"},
            "reasoning_summary": {"type": "string"},
        },
        "required": ["product_to_reactant", "confidence", "reasoning_summary"],
    }


def build_user_prompt(template: str, task: ReactionTask, unmapped_smiles: str) -> str:
    reactants, _agents, products = split_reaction_smiles(unmapped_smiles)
    return template.format(
        reaction_index=task.index,
        unmapped_reaction_smiles=unmapped_smiles,
        reactant_graph_json=side_graph_json(reactants),
        product_graph_json=side_graph_json(products),
    )


def http_json(
    url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers=dict(headers), method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_responses_text(data: Mapping[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return str(data["output_text"])
    chunks: List[str] = []
    for item in data.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content.get("text"), str):
                chunks.append(str(content["text"]))
    return "".join(chunks)


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def call_openai(
    system_prompt: str, user_prompt: str, args: argparse.Namespace
) -> Tuple[Dict[str, Any], str]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in ${args.api_key_env}.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    base_url = args.base_url.rstrip("/")
    schema = response_schema()

    if args.api == "responses":
        payload: Dict[str, Any] = {
            "model": args.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_output_tokens": args.max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "atom_mapping_response",
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        if args.reasoning_effort:
            payload["reasoning"] = {"effort": args.reasoning_effort}
        url = f"{base_url}/responses"
        data = http_json(url, headers, payload, args.timeout)
        text = extract_responses_text(data)
    else:
        payload = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": args.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "atom_mapping_response",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        if args.temperature is not None:
            payload["temperature"] = args.temperature
        url = f"{base_url}/chat/completions"
        data = http_json(url, headers, payload, args.timeout)
        text = data["choices"][0]["message"]["content"]

    return extract_json_object(text), text


def call_openai_with_retries(
    system_prompt: str, user_prompt: str, args: argparse.Namespace
) -> Tuple[Dict[str, Any], str]:
    last_error: Optional[BaseException] = None
    for attempt in range(args.retries + 1):
        try:
            return call_openai(system_prompt, user_prompt, args)
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            RuntimeError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt >= args.retries:
                break
            time.sleep(args.retry_delay * (2**attempt))
    raise RuntimeError(
        f"LLM request failed after {args.retries + 1} attempts: {last_error}"
    )


def parse_product_to_reactant(payload: Mapping[str, Any]) -> List[Tuple[int, int]]:
    raw = payload.get("product_to_reactant")
    if isinstance(raw, dict):
        return [(int(p), int(r)) for p, r in raw.items()]
    if not isinstance(raw, list):
        raise ValueError("Response field product_to_reactant must be a list or object.")

    pairs: List[Tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Each product_to_reactant entry must be an object.")
        pairs.append((int(item["product_atom"]), int(item["reactant_atom"])))
    return pairs


def mapped_reaction_from_pairs(
    unmapped_smiles: str, pairs: Sequence[Tuple[int, int]], keep_agents: bool = False
) -> str:
    reactants, agents, products = split_reaction_smiles(unmapped_smiles)
    reactant_mol = mol_from_side(reactants)
    agent_mol = mol_from_side(agents)
    product_mol = mol_from_side(products)

    for mol in (reactant_mol, agent_mol, product_mol):
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

    product_to_reactant = {int(p): int(r) for p, r in pairs}
    if len(product_to_reactant) != len(pairs):
        raise ValueError("Duplicate product_atom ids in response.")
    if set(product_to_reactant) != set(range(product_mol.GetNumAtoms())):
        missing = sorted(
            set(range(product_mol.GetNumAtoms())) - set(product_to_reactant)
        )
        extra = sorted(set(product_to_reactant) - set(range(product_mol.GetNumAtoms())))
        raise ValueError(
            f"Product atom coverage mismatch; missing={missing}, extra={extra}."
        )
    if len(set(product_to_reactant.values())) != len(product_to_reactant):
        raise ValueError("Duplicate reactant_atom ids in response.")

    for product_atom, reactant_atom in product_to_reactant.items():
        if reactant_atom < 0 or reactant_atom >= reactant_mol.GetNumAtoms():
            raise ValueError(f"Reactant atom id out of range: {reactant_atom}.")
        pa = product_mol.GetAtomWithIdx(product_atom)
        ra = reactant_mol.GetAtomWithIdx(reactant_atom)
        if pa.GetAtomicNum() != ra.GetAtomicNum():
            raise ValueError(
                f"Element mismatch for product atom {product_atom} ({pa.GetSymbol()}) "
                f"and reactant atom {reactant_atom} ({ra.GetSymbol()})."
            )

    next_map = 1
    for product_atom in sorted(product_to_reactant):
        reactant_atom = product_to_reactant[product_atom]
        reactant_mol.GetAtomWithIdx(reactant_atom).SetAtomMapNum(next_map)
        product_mol.GetAtomWithIdx(product_atom).SetAtomMapNum(next_map)
        next_map += 1

    for atom in reactant_mol.GetAtoms():
        if atom.GetAtomMapNum() == 0:
            atom.SetAtomMapNum(next_map)
            next_map += 1
    if keep_agents:
        for atom in agent_mol.GetAtoms():
            if atom.GetAtomMapNum() == 0:
                atom.SetAtomMapNum(next_map)
                next_map += 1

    lhs = Chem.MolToSmiles(reactant_mol, canonical=True, isomericSmiles=True)
    rhs = Chem.MolToSmiles(product_mol, canonical=True, isomericSmiles=True)
    if keep_agents:
        middle = Chem.MolToSmiles(agent_mol, canonical=True, isomericSmiles=True)
        return f"{lhs}>{middle}>{rhs}"
    return f"{lhs}>>{rhs}"


def expected_normalized(
    source_smiles: str, keep_agents: bool
) -> Tuple[str, str, str, str]:
    expected_smiles = (
        source_smiles if keep_agents else reaction_without_agents(source_smiles)
    )
    reactants, agents, products = normalize_mapped_reaction_parts(expected_smiles)
    normalized = (
        f"{reactants}>{agents}>{products}" if agents else f"{reactants}>>{products}"
    )
    return expected_smiles, normalized, reactants, products


def run_one_llm(
    task: ReactionTask,
    system_prompt: str,
    user_template: str,
    args: argparse.Namespace,
) -> LLMRecord:
    started = time.perf_counter()
    source_smiles = task.smiles
    unmapped_smiles = clear_atom_maps_from_reaction(
        source_smiles, keep_agents=args.keep_agents
    )
    expected_smiles, expected_norm, expected_reactants, expected_products = (
        expected_normalized(source_smiles, args.keep_agents)
    )

    user_prompt = build_user_prompt(user_template, task, unmapped_smiles)
    payload, raw_text = call_openai_with_retries(system_prompt, user_prompt, args)
    pairs = parse_product_to_reactant(payload)
    predicted_smiles = mapped_reaction_from_pairs(
        unmapped_smiles, pairs, keep_agents=args.keep_agents
    )
    predicted_reactants, predicted_agents, predicted_products = (
        normalize_mapped_reaction_parts(predicted_smiles)
    )
    predicted_norm = (
        f"{predicted_reactants}>{predicted_agents}>{predicted_products}"
        if predicted_agents
        else f"{predicted_reactants}>>{predicted_products}"
    )
    reactants_matched = expected_reactants == predicted_reactants
    products_matched = expected_products == predicted_products

    record = BenchmarkRecord(
        index=task.index,
        rdf_line=task.rdf_line,
        source_smiles=source_smiles,
        unmapped_smiles=unmapped_smiles,
        expected_smiles=expected_smiles,
        predicted_smiles=predicted_smiles,
        expected_normalized=expected_norm,
        predicted_normalized=predicted_norm,
        expected_reactants_normalized=expected_reactants,
        predicted_reactants_normalized=predicted_reactants,
        expected_products_normalized=expected_products,
        predicted_products_normalized=predicted_products,
        reactants_matched=reactants_matched,
        products_matched=products_matched,
        matched=reactants_matched and products_matched,
        mapper_status="llm",
        elapsed_seconds=time.perf_counter() - started,
        topology_counts={},
    )
    return LLMRecord(
        benchmark=record,
        model=args.model,
        raw_response=raw_text,
        reasoning_summary=str(payload.get("reasoning_summary", "")),
        confidence=float(payload["confidence"]) if "confidence" in payload else None,
    )


def error_llm_record(
    task: ReactionTask, exc: BaseException, args: argparse.Namespace
) -> LLMRecord:
    record = BenchmarkRecord(
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
    return LLMRecord(benchmark=record, model=args.model, raw_response="")


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


def write_llm_json_report(
    path: str, records: Sequence[LLMRecord], skipped: int, errors: int, workers: int
) -> None:
    benchmark_records = [record.benchmark for record in records]
    payload = {
        "summary": {
            "completed": len(records),
            "matched": sum(1 for record in benchmark_records if record.matched),
            "mismatched": sum(1 for record in benchmark_records if not record.matched),
            "reactant_matches": sum(
                1 for record in benchmark_records if record.reactants_matched
            ),
            "product_matches": sum(
                1 for record in benchmark_records if record.products_matched
            ),
            "skipped": skipped,
            "errors": errors,
            "workers": workers,
        },
        "records": [
            {
                **asdict(record.benchmark),
                "model": record.model,
                "confidence": record.confidence,
                "reasoning_summary": record.reasoning_summary,
                "raw_response": record.raw_response,
            }
            for record in records
        ],
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
        help="Worker threads/API calls to use.",
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
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.5"))
    parser.add_argument("--api", choices=["responses", "chat"], default="responses")
    parser.add_argument(
        "--base-url", default=os.environ.get("BASE_URL", "https://livai-api.llnl.gov/")
    )
    parser.add_argument("--api-key-env", default="LIVAI_API_KEY")
    parser.add_argument("--system-prompt", default=str(DEFAULT_SYSTEM_PROMPT))
    parser.add_argument(
        "--skill-prompt",
        default=str(DEFAULT_SKILL_PROMPT),
        help="Additional skill/instruction file appended to the system prompt.",
    )
    parser.add_argument(
        "--no-skill-prompt",
        dest="use_skill_prompt",
        action="store_false",
        help="Do not append the skill prompt.",
    )
    parser.set_defaults(use_skill_prompt=True)
    parser.add_argument("--user-prompt-template", default=str(DEFAULT_USER_PROMPT))
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        help="Responses API reasoning effort for models that support it.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    system_prompt = build_system_prompt(args)
    user_template = load_text(args.user_prompt_template)
    tasks, skipped = collect_tasks(args)
    records: List[LLMRecord] = []
    errors = 0
    started = time.perf_counter()

    with tqdm(total=len(tasks), unit="rxn", desc="LLM mapping") as progress:
        if tasks:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        run_one_llm, task, system_prompt, user_template, args
                    ): task
                    for task in tasks
                }
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        errors += 1
                        record = error_llm_record(task, exc, args)

                    records.append(record)
                    benchmark = record.benchmark
                    if args.debug or not benchmark.matched or benchmark.error:
                        tqdm.write(format_record(benchmark, debug=args.debug))
                        if args.debug and record.reasoning_summary:
                            tqdm.write(f"  llm reasoning: {record.reasoning_summary}")
                    update_progress_postfix(
                        progress, [r.benchmark for r in records], errors
                    )
                    progress.update(1)

    records.sort(key=lambda record: record.benchmark.index)
    benchmark_records = [record.benchmark for record in records]
    print_summary(
        benchmark_records,
        skipped=skipped,
        errors=errors,
        started=started,
        workers=args.workers,
    )
    print(f"  model:      {args.model}")
    print(f"  api:        {args.api}")

    if args.json_report:
        write_llm_json_report(
            args.json_report,
            records,
            skipped=skipped,
            errors=errors,
            workers=args.workers,
        )
        print(f"  json:       {args.json_report}")

    if errors:
        return 1
    if args.fail_on_mismatch and any(
        not record.benchmark.matched for record in records
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
