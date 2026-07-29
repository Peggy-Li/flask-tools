###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum, StrEnum
import json
import os
from typing import Any

from pydantic import BaseModel, SerializeAsAny

LLM_API_KEY_ENV_VARS = (
    "FLASK_ORCHESTRATOR_API_KEY",
    "PIPETTE_API_KEY",
    "OPENAI_API_KEY",
)

LLM_BASE_URL_ENV_VARS = (
    "FLASK_ORCHESTRATOR_URL",
    "PIPETTE_LLM_BASE_URL",
)


def resolve_llm_api_key(explicit_api_key: str | None = None) -> str | None:
    if explicit_api_key:
        return explicit_api_key
    for env_var in LLM_API_KEY_ENV_VARS:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key
    return None


def resolve_llm_base_url(explicit_base_url: str | None = None) -> str:
    if explicit_base_url:
        return explicit_base_url
    for env_var in LLM_BASE_URL_ENV_VARS:
        base_url = os.environ.get(env_var)
        if base_url:
            return base_url
    else:
        raise ValueError(f"One of {LLM_BASE_URL_ENV_VARS} is required.")


DEFAULT_LLM_BASE_URL = resolve_llm_base_url()


class SmilesContainer(str):
    """A class so most tools can assume rxn_smiles passed to tool.run() is a plain string, but some can play around
    with the presence of the reagents
    """

    def __new__(cls, value, original_smiles=None):
        instance = super().__new__(cls, value)
        return instance

    def __init__(self, value, reagents_smi: str | None = None):
        self.reagents_smi = reagents_smi

    def __repr__(self):
        return f"SmilesContainer({str.__repr__(self)}, reagents_smiles={self.reagents_smi!r})"


class ToolStatus(str, Enum):
    PASS = "pass"  # Reaction passed this tool
    FAIL = "fail"  # Reaction failed to pass this tool
    UNKNOWN = "unknown"  # Result was ambiguous
    NOT_RUN = "not_run"  # Tool was not called (skipped because of prior tool)
    ERROR = "error"  # Tool call errored out


class FinalGrade(str, Enum):
    LIKELY = "likely"  # The reaction is valid and the product(s) are a likely outcome
    POSSIBLE = "possible but unlikely"  # The reaction is plausible but other products exist that would have significantly higher likelihood
    IMPOSSIBLE = "impossible"  # The reaction is physically or chemically impossible
    UNCERTAIN = "uncertain"  #  With the given tools and knowledge, it is not possible to ascertain whether this reaction is possible or not. (Use this, e.g., when certain tools cannot run, or the results are ambiguous)


class ToolResult(BaseModel):
    name: str
    status: ToolStatus
    data: (
        SerializeAsAny[ToolResultDetails] | None
    )  # None if tool had an error or wasn't run. Must use SerializeAsAny or else model_dump only outputs the parent class ToolResultDetails' fields which are no fields.
    comment: str = ""
    skipped_reason: str | None = (
        None  # If a priority checker skipped this tool, like in an exact rule pipeline, or a traceback if there was an error
    )


class ToolResultDetails(BaseModel):
    # Miscellaneous details from a verifier. Each verifier should have their own Details class
    pass


class ReactionGrade(BaseModel):
    final_grade: FinalGrade
    short_comment: str
    results: list[ToolResult]
    comment: str = ""

    def __str__(self) -> str:
        lines = [
            f"ReactionGrade(final_grade={self.final_grade.value}, short_comment={self.short_comment})"
        ]
        if self.comment:
            lines.append(f"comment: {self.comment}")

        if not self.results:
            lines.append("tool_results: none")
            return "\n".join(lines)

        lines.append("tool_results:")
        for result in self.results:
            tool_line = f"  - {result.name}: {result.status.value}"
            if result.comment:
                tool_line += f" - {result.comment}"
            if result.skipped_reason:
                tool_line += f" [skipped: {result.skipped_reason}]"
            lines.append(tool_line)
            if result.data:
                for data_line in result.data.model_dump_json(indent=2).splitlines():
                    lines.append(f"    {data_line}")
        return "\n".join(lines)


Smi = str
CheckerName = str
ToolResultsDict = dict[tuple[Smi, CheckerName], ToolResult]


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RxnSide(StrEnum):
    REACTANTS = "reactants"
    PRODUCTS = "products"


class AllowedUnbalancedMolecule(BaseModel):
    # A molecule that is commonly missing from one side of a rxn. For example, a solvent may be on the left hand side.
    name: str
    smiles: str
    formula: dict[str, int]  # Dict where keys are elements and values are counts
    confidence: Confidence


class ReactionMassImbalanceExplanation(BaseModel):
    # Corresponds to an AllowedUnbalancedMolecule in a reaction
    name: str
    smiles: str
    missing_side: RxnSide  # Which side does is missing this mass
    mass_amu: float
    confidence: Confidence
