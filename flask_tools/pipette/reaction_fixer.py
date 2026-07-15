###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from .config import PipetteConfig
from .constants import ToolResult, resolve_llm_api_key, ToolResultDetails
from .llm_query import query_task, query_task_async
from .smiles import (
    canonicalize_reaction_smiles,
    canonicalize_smiles,
    split_reaction_smiles,
)

# Notes about the prompt:
# The [H+] part is needed for rxns like caffeine, Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C


class ReactionFixResultDetails(ToolResultDetails):
    original_reaction_smiles: str
    fixed_reaction_smiles: str
    removed_agents: list[str]
    added_reactants: list[str]
    removed_products: list[str]
    added_products: list[str]
    reasoning_summary: str


class ReactionFixResponse(BaseModel):
    fixed_reaction_smiles: str
    comment: str = ""


_FixerT = TypeVar("_FixerT", bound="BaseLLMReactionFixer")


class BaseLLMReactionFixer:
    name = "llm_reaction_fix"

    def __init__(
        self,
        *,
        model: str,
        url: str,
        reasoning_effort: Literal["low", "medium", "high"],
        api_key: str,
        prompt_path: Path,
        prompt: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.url = url
        self.reasoning_effort = reasoning_effort
        self.prompt_path = prompt_path
        self._prompt = prompt

    @classmethod
    def from_config(cls: type[_FixerT], config: PipetteConfig) -> _FixerT | None:
        fixer_config = config.llm_reaction_fixer
        if not fixer_config.enabled:
            return None

        api_key = resolve_llm_api_key(fixer_config.api_key)
        if not api_key:
            return None

        return cls(
            model=fixer_config.model,
            url=fixer_config.url,
            reasoning_effort=fixer_config.reasoning_effort,
            api_key=api_key,
            prompt_path=fixer_config.prompt_path,
            prompt=fixer_config.prompt,
        )

    @property
    def system_prompt(self) -> str:
        if self._prompt is not None:
            return self._prompt
        return self.prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def _canonical_component_list(side: str) -> list[str]:
        if not side:
            return []
        return [
            canonicalize_smiles(component) for component in side.split(".") if component
        ]

    @staticmethod
    def _multiset_difference(updated: list[str], original: list[str]) -> list[str]:
        diff = Counter(updated) - Counter(original)
        added: list[str] = []
        for component in sorted(diff):
            added.extend([component] * diff[component])
        return added

    def _parse_reaction_fix(
        self, original_reaction_smiles: str, response_text: str
    ) -> ReactionFixResultDetails:
        try:
            parsed = ReactionFixResponse.model_validate_json(response_text)
        except Exception as exc:
            raise ValueError(
                f"Reaction fixer did not return valid JSON: {response_text}"
            ) from exc

        canonical_fixed = canonicalize_reaction_smiles(
            parsed.fixed_reaction_smiles.strip(), include_agents=True
        )
        original_reactants, original_agents, original_products = split_reaction_smiles(
            canonicalize_reaction_smiles(original_reaction_smiles, include_agents=True)
        )
        fixed_reactants, fixed_agents, fixed_products = split_reaction_smiles(
            canonical_fixed
        )

        return ReactionFixResultDetails(
            original_reaction_smiles=original_reaction_smiles,
            fixed_reaction_smiles=canonical_fixed,
            removed_agents=self._multiset_difference(
                self._canonical_component_list(original_agents),
                self._canonical_component_list(fixed_agents),
            ),
            added_reactants=self._multiset_difference(
                self._canonical_component_list(fixed_reactants),
                self._canonical_component_list(original_reactants),
            ),
            removed_products=self._multiset_difference(
                self._canonical_component_list(original_products),
                self._canonical_component_list(fixed_products),
            ),
            added_products=self._multiset_difference(
                self._canonical_component_list(fixed_products),
                self._canonical_component_list(original_products),
            ),
            reasoning_summary=parsed.comment,
        )

    @staticmethod
    def _build_user_payload(
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> dict[str, Any]:
        serialized_results = [r.model_dump(exclude_none=True) for r in results]
        for s in serialized_results:
            if "skipped_reason" in s:
                del s["skipped_reason"]
        return {
            "reaction_smiles": rxn_smiles,
            "tool_results": serialized_results,
            "instructions": (
                "Return a repaired reaction SMILES that removes agents and adds "
                "missing byproducts or counter-species on either side when needed. "
                "If no credible repair is possible, return the original reaction_smiles."
            ),
        }


class LLMReactionFixer(BaseLLMReactionFixer):
    def fix(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> ReactionFixResultDetails:
        user_prompt = json.dumps(
            self._build_user_payload(rxn_smiles, results),
            indent=2,
            sort_keys=True,
        )
        response_text = query_task(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            api_key=self.api_key,
            url=self.url,
            reasoning_effort=self.reasoning_effort,
            structured_output_schema=ReactionFixResponse,
            agent_name="PipetteFixer",
        )
        try:
            return self._parse_reaction_fix(rxn_smiles, response_text)
        except Exception as exc:
            raise ValueError(
                f"Reaction fixer failed to parse response: {exc} for {rxn_smiles} with response {response_text}"
            ) from exc


class AsyncLLMReactionFixer(BaseLLMReactionFixer):
    async def fix(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> ReactionFixResultDetails:
        user_prompt = json.dumps(
            self._build_user_payload(rxn_smiles, results),
            indent=2,
            sort_keys=True,
        )
        response_text = await query_task_async(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            api_key=self.api_key,
            url=self.url,
            reasoning_effort=self.reasoning_effort,
            structured_output_schema=ReactionFixResponse,
            agent_name="PipetteFixer",
        )
        try:
            return self._parse_reaction_fix(rxn_smiles, response_text)
        except Exception as exc:
            raise ValueError(
                f"Reaction fixer failed to parse response: {exc} for {rxn_smiles} with response {response_text}"
            ) from exc
