###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar, Any

from pydantic import BaseModel

from .config import PipetteConfig, ReasoningEffort
from .constants import (
    FinalGrade,
    LLM_API_KEY_ENV_VARS,
    ReactionGrade,
    ToolResult,
    resolve_llm_api_key,
)
from .llm_query import query_task, query_task_async


class ReactionGradeResponse(BaseModel):
    final_grade: FinalGrade
    short_comment: str = "llm_judge"
    comment: str = ""


_JudgeT = TypeVar("_JudgeT", bound="BaseLLMJudge")


class BaseLLMJudge:
    def __init__(
        self,
        *,
        url: str,
        model: str,
        reasoning_effort: ReasoningEffort,
        api_key: str,
        prompt_path: Path,
        prompt: str | None = None,
        enable_atom_mapping_dict_in_prompt: bool = False,
    ) -> None:
        self.url = url
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key
        self.prompt_path = prompt_path
        self._prompt = prompt
        self.enable_atom_mapping_dict_in_prompt = enable_atom_mapping_dict_in_prompt

    @classmethod
    def from_config(cls: type[_JudgeT], config: PipetteConfig) -> _JudgeT:
        api_key = resolve_llm_api_key(config.llm_judge.api_key)
        if not api_key:
            raise ValueError(
                "LLM Judge requires an API key via llm_judge.api_key, "
                + ", ".join(LLM_API_KEY_ENV_VARS)
                + "."
            )
        return cls(
            url=config.llm_judge.url,
            model=config.llm_judge.model,
            reasoning_effort=config.llm_judge.reasoning_effort,
            api_key=api_key,
            prompt_path=config.llm_judge.prompt_path,
            prompt=config.llm_judge.prompt,
            enable_atom_mapping_dict_in_prompt=(
                config.llm_judge.enable_atom_mapping_dict_in_prompt
            ),
        )

    @property
    def system_prompt(self) -> str:
        if self._prompt is not None:
            return self._prompt
        return self.prompt_path.read_text(encoding="utf-8")

    def _build_user_payload(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> dict[str, Any]:
        serialized_results: list[dict[str, object]] = [
            r.model_dump(exclude_none=True) for r in results
        ]
        for result, serialized_result in zip(results, serialized_results):
            s = serialized_result
            if "skipped_reason" in s:
                del s["skipped_reason"]
            if (
                not self.enable_atom_mapping_dict_in_prompt
                and result.name == "llm_atom_mapping"
                and isinstance(s.get("data"), dict)
            ):
                s["data"].pop("product_to_reactant", None)
        user_payload = {
            "reaction_smiles": rxn_smiles,
            "tool_results": serialized_results,
        }
        return user_payload

    @staticmethod
    def _parse_reaction_grade(
        response_text: str,
        results: list[ToolResult],
    ) -> ReactionGrade:
        try:
            parsed: ReactionGradeResponse = ReactionGradeResponse.model_validate_json(
                response_text
            )
        except Exception as exc:
            raise ValueError(
                f"LLM judge did not return valid JSON: {response_text}"
            ) from exc

        return ReactionGrade(
            final_grade=parsed.final_grade,
            short_comment=parsed.short_comment,
            results=list(results),
            comment=parsed.comment,
        )


class AsyncLLMJudge(BaseLLMJudge):
    async def judge(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> ReactionGrade:
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
            structured_output_schema=ReactionGradeResponse,
            agent_name="PipetteJudge",
        )
        return self._parse_reaction_grade(response_text, results)


class LLMJudge(BaseLLMJudge):
    def judge(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> ReactionGrade:
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
            structured_output_schema=ReactionGradeResponse,
            agent_name="PipetteJudge",
        )
        return self._parse_reaction_grade(response_text, results)
