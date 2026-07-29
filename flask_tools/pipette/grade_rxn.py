###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

"""
A CLI tool and main entry point to pipette. See pipette/README.md for more details.

Usage:
python -m flask_tools.pipette.grade_rxn --rxn-smi 'Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C'

Or:
from flask_tools.pipette.grade_rxn import grade_reaction
from flask_tools.pipette.config import load_config

result = grade_reaction(["Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]) # Optionally, config='llm-judge' or "file.yaml"

# Or
config = load_config('llm-judge')
result = grade_reaction(["Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"], config=config)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from flask_tools.pipette.config import PipetteConfig, load_config, ConfigType
from flask_tools.pipette.constants import ReactionGrade
from flask_tools.pipette.pipeline import build_default_pipeline
from flask_tools.pipette.reaction_fixer import ReactionFixResultDetails

if TYPE_CHECKING:
    from .judge import AsyncLLMJudge
    from flask_tools.pipette.constants import ToolResult

REACTION_SMILES_COLUMNS = (
    "rxn_smiles",
    "reaction_smiles",
    "rxn_smi",
    "smiles",
)


def _get_possible_fixed_rxn_smi(reaction_grade: ReactionGrade) -> str | None:
    """Return a fixed reaction SMILES string if fixer was one of the tool calls."""
    tool_res: ToolResult
    for tool_res in reaction_grade.results:
        if tool_res.name == "llm_reaction_fix":
            d: ReactionFixResultDetails | None = tool_res.data
            if d:
                return d.fixed_reaction_smiles
    return None


def _grade_reactions(
    rxn_smiles_list: list[str],
    config: PipetteConfig | str | None = ConfigType.LLM_JUDGE_NO_DFT,
    judge: AsyncLLMJudge | None = None,
) -> list[ReactionGrade]:
    if isinstance(config, str):
        resolved_config = load_config(config)
    else:
        resolved_config = config or PipetteConfig.from_default_ai_yaml()
    pipeline = build_default_pipeline(config=resolved_config, judge=judge)
    return list(pipeline.grade(rxn_smiles_list))


def _build_output_records(
    rxn_smiles_list: list[str],
    results: list[ReactionGrade],
) -> list[dict]:
    return [
        {
            "rxn_smiles": rxn_smiles,
            "cleaned_rxn_smiles": _get_possible_fixed_rxn_smi(result) or rxn_smiles,
            "grade": result.model_dump(mode="json", exclude_none=True),
        }
        for rxn_smiles, result in zip(rxn_smiles_list, results, strict=True)
    ]


def grade_reaction(
    rxn_smiles_list: str | list[str],
    config: PipetteConfig | str | None = ConfigType.LLM_JUDGE_NO_DFT,
    judge: AsyncLLMJudge | None = None,
    verbose: bool = False,
) -> list[ReactionGrade]:
    """Grade reactions, return list of ReactionGrade
    Args:
        rxn_smiles_list: str or list of strings, of reaction SMILES separated by >>
        config: Optional. PipetteConfig of one of ConfigType strings
        judge: Optional Judge object for final grading
        verbose: If true, prints summary
    """
    if isinstance(rxn_smiles_list, str):
        rxn_smiles_list = [rxn_smiles_list]
    res = _grade_reactions(rxn_smiles_list, config=config, judge=judge)
    assert len(res) == len(rxn_smiles_list)
    if verbose:
        for rxn_smiles, result in zip(rxn_smiles_list, res, strict=True):
            print(f"{rxn_smiles}:\n\n{result}\n\n")
    return res


def grade_reaction_json(
    rxn_smiles: str,
    config: PipetteConfig | str | None = ConfigType.LLM_JUDGE_NO_DFT,
    judge: AsyncLLMJudge | None = None,
) -> dict:
    """Return single-reaction JSON object with full information."""
    return grade_reactions_json([rxn_smiles], config=config, judge=judge)[0]


def grade_reactions_json(
    rxn_smiles_list: list[str],
    config: PipetteConfig | str | None = ConfigType.LLM_JUDGE_NO_DFT,
    judge: AsyncLLMJudge | None = None,
) -> list[dict]:
    """Return the full information JSON payload for one or more reactions."""
    results = _grade_reactions(rxn_smiles_list, config=config, judge=judge)
    return _build_output_records(rxn_smiles_list, results)


def load_reaction_smiles_file(path: str | Path) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    if file_path.suffix.lower() == ".tsv":
        return _load_reaction_smiles_csv(file_path)
    elif file_path.suffix.lower() == ".csv":
        return _load_reaction_smiles_csv(file_path, delimiter=",")
    return _load_reaction_smiles_lines(file_path)


def _load_reaction_smiles_csv(path: Path, delimiter="\t") -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            return []

        column_name = next(
            (name for name in REACTION_SMILES_COLUMNS if name in reader.fieldnames),
            None,
        )
        if column_name is None:
            raise ValueError(
                "CSV file must contain one of these reaction SMILES columns: "
                + ", ".join(REACTION_SMILES_COLUMNS)
            )

        return [
            (row.get(column_name) or "").strip()
            for row in reader
            if (row.get(column_name) or "").strip()
        ]


def _load_reaction_smiles_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> list[dict]:
    parser = argparse.ArgumentParser(
        description="Grade one or more reaction SMILES strings."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--rxn-smi",
        nargs="+",
        dest="rxn_smi",
        help="One or more reaction SMILES strings, for example 'CCO>>CC=O'.",
    )
    input_group.add_argument(
        "-f",
        "--file",
        dest="file",
        help="Reaction SMILES file, one per line or a CSV with a reaction SMILES column.",
    )
    parser.add_argument(
        "--config",
        default=ConfigType.LLM_JUDGE_NO_DFT,
        help=f"Path to a Pipette YAML config file, or the special values '{ConfigType.LLM_JUDGE_WITH_DFT}' (default), "
        f"or '{ConfigType.LLM_JUDGE_NO_DFT}.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Prints out json object",
    )
    parser.add_argument(
        "--no-fix",
        action="store_true",
        help="Disable the LLM reaction fixer for this run.",
    )
    args = parser.parse_args()

    rxn_smiles_list = args.rxn_smi or load_reaction_smiles_file(args.file)
    config = load_config(args.config)
    if args.no_fix:
        config.settings.use_fixing = False
    results = grade_reaction(rxn_smiles_list, config=config, verbose=True)
    output = _build_output_records(rxn_smiles_list, results)
    if args.verbose:
        print("Full output, json formatted:")
        print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    raise SystemExit(0 if bool(main()) else 1)
