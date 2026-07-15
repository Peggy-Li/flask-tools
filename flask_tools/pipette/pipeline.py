###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations
from collections.abc import Callable, Iterator
import inspect
import traceback

from .config import PipetteConfig
from .verifiers import (
    BasicSmilesValidationChecker,
    ChargeConservationChecker,
    ExactMatchChecker,
    MassConservationChecker,
    ReactionChecker,
    ReactionEnergyChecker,
)
from .verifiers.base import CacheableReactionChecker
from .judge import AsyncLLMJudge
from .constants import ReactionGrade, ToolResult, ToolStatus, ToolResultsDict
from .reaction_fixer import AsyncLLMReactionFixer, ReactionFixResultDetails
from .smiles import canonicalize_reaction_smiles
from .llm_query import _run_coroutine_sync

CheckerFactory = Callable[[PipetteConfig], ReactionChecker]
LLMJudge = AsyncLLMJudge
LLMReactionFixer = AsyncLLMReactionFixer


def resolve_tool_list(
    tool_list: str | list[str] | None,
    available_tools: list[str],
    use_dft: bool = False,
) -> list[str]:
    if use_dft:
        assert "reaction_energy" in available_tools
    if not use_dft:
        available_tools = [t for t in available_tools if t != "reaction_energy"]
    if tool_list is None:
        return []
    if tool_list == "all":
        return list(available_tools)

    if not isinstance(tool_list, list):
        raise ValueError(
            "PipetteConfig.tool_list must be None, 'all', or a list of tool names."
        )
    if len(set(tool_list)) != len(tool_list):
        raise ValueError(
            "PipetteConfig.tool_list must not contain duplicate tool names."
        )

    unknown = set(tool_list) - set(available_tools)
    if unknown:
        raise ValueError(
            f"Unknown tool names in PipetteConfig.tool_list: {', '.join(sorted(unknown))}"
        )

    return list(tool_list)


class GradingPipeline:
    """Pipeline to call tools. See GradingPipeline.grade(), which takes a list of reaction SMILES and yields
    `ReactionGrade`s. See README for more details.

    Example:
        pipeline = build_default_pipeline()
        reaction_grades = list(pipeline.grade([rxn_smi0, rxn_smi1]))  # grade() yields `ReactionGrade`s

        print(reaction_grades[0].final_grade, reaction_grades[0].comment)
        print(reaction_grades[0].tool_results)
    """

    def __init__(
        self,
        checkers: list[ReactionChecker],
        config: PipetteConfig | None = None,
        judge: AsyncLLMJudge | None = None,
        reaction_fixer: AsyncLLMReactionFixer | None = None,
    ) -> None:
        """judge: If provided, overrides the judge specified by the config"""
        self.checkers = checkers or []
        self.config = config or PipetteConfig()
        self.judge = judge
        self.reaction_fixer = reaction_fixer
        self._validate_configuration()
        if self.judge is None:
            self.judge = AsyncLLMJudge.from_config(self.config)
        if self.reaction_fixer is None:
            self.reaction_fixer = AsyncLLMReactionFixer.from_config(self.config)

    def _validate_configuration(self) -> None:
        checker_names = [checker.name for checker in self.checkers]
        if len(set(checker_names)) != len(checker_names):
            raise ValueError("Checker names must be unique within a grading pipeline.")

        unknown = sorted(set(self.config.llm_judge.allow_fail) - set(checker_names))
        if unknown:
            print(
                "Tool names in PipetteConfig.llm_judge.allow_fail that are not in tool_list: "
                + ", ".join(unknown)
                + ". Ignoring."
            )

    def _should_skip_remaining(
        self, checker: ReactionChecker, result: ToolResult
    ) -> bool:
        if (
            not self.config.settings.stop_on_hard_fail
            or not checker.stops_on_fail
            or (result.status not in {ToolStatus.FAIL, ToolStatus.ERROR})
            or (
                self.config.llm_judge.allow_fail == "all"
                or (checker.name in self.config.llm_judge.allow_fail)
            )
        ):
            return False
        return True

    @staticmethod
    def _should_try_llm_fix(context: dict[str, ToolResult]) -> bool:
        basic = context.get("basic_smiles_validation")
        exact_match = context.get("exact_match")
        charge = context.get("charge_conservation")
        mass = context.get("mass_conservation")
        if basic is None or exact_match is None or charge is None or mass is None:
            return False
        return (
            basic.status is ToolStatus.PASS
            and exact_match.status in (ToolStatus.UNKNOWN, ToolStatus.NOT_RUN)
            and charge.status is ToolStatus.PASS
            # if mass fails, or even if difference is likely a common solvent
            and mass.status == ToolStatus.FAIL
            or mass.data["mass_difference_amu"]
        )

    @staticmethod
    def _with_prefix_results(
        result: ReactionGrade,
        prefix_results: list[tuple[str, str, ToolResult]],
    ) -> ReactionGrade:
        """Combine the tool results from the prefix_results with the results"""
        if not prefix_results:
            return result
        return ReactionGrade(
            final_grade=result.final_grade,
            short_comment=result.short_comment,
            results=[tool_result for _, _, tool_result in prefix_results],
            comment=result.comment,
        )

    @staticmethod
    def _build_fix_result(fix: ReactionFixResultDetails) -> ToolResult:
        return ToolResult(
            name="llm_reaction_fix",
            status=ToolStatus.PASS,
            data=fix,
            comment="LLM proposed a corrected reaction",
        )

    @staticmethod
    async def _attempt_llm_fix_async(
        reaction_fixer: BaseLLMReactionFixer,
        rxn_smiles: str,
        tool_results: ToolResultsDict,
    ) -> tuple[ToolResult, str | None] | None:
        if self.reaction_fixer is None:
            return None

        try:
            fix_result = reaction_fixer.fix(
                rxn_smiles,
                list(tool_results.values()),
            )
            fix = await fix_result if inspect.isawaitable(fix_result) else fix_result
        except Exception as exc:
            return (
                ToolResult(
                    name="llm_reaction_fix",
                    status=ToolStatus.ERROR,
                    data=None,
                    comment=f"LLM reaction fixer failed: {exc}",
                ),
                None,
            )

        if fix.fixed_reaction_smiles == canonicalize_reaction_smiles(
            rxn_smiles, include_agents=True
        ):
            return (
                ToolResult(
                    name="llm_reaction_fix",
                    status=ToolStatus.UNKNOWN,
                    data=fix,
                    comment="LLM reaction fixer did not propose a changed reaction.",
                ),
                None,
            )

        return self._build_fix_result(fix), fix.fixed_reaction_smiles

    async def _finalize_grade_async(
        self,
        rxn_smiles: str,
        all_tool_results: ToolResultsDict,
        before_llm_fix_results: ToolResultsDict,  # Unused, but could be used to exclude the not-fixed results  # noqa
    ) -> ReactionGrade:
        """Exact grading happens without the prefix_results, namely the results before llm-fix of rxn balance.
        AI grading does use it, because it should be smart enough to deal with it.
        """
        if self.judge is None:
            raise ValueError("AI mode requires an LLM judge implementation.")

        judge_result = self.judge.judge(
            rxn_smiles,
            list(all_tool_results.values()),
        )
        res = await judge_result if inspect.isawaitable(judge_result) else judge_result
        assert res is not None
        assert isinstance(res, ReactionGrade)
        return res

    async def grade_one_async(
        self,
        rxn_smiles: str,
        *,
        fix_attempted: bool = False,
        previous_tool_results: ToolResultsDict | None = None,
    ) -> ReactionGrade:
        async def maybe_call_fixer() -> ReactionGrade | None:
            if not (self.config.settings.use_fixing and not fix_attempted):
                # and self._should_try_llm_fix(context)
                return None

            if (rxn_smiles, "llm_reaction_fix") in previous_tool_results:
                raise RuntimeError("llm_reaction_fix already ran")
            fix_attempt = await self._attempt_llm_fix_async(
                self.reaction_fixer,
                rxn_smiles,
                all_tool_results,
            )
            if fix_attempt is not None:
                fix_result, fixed_rxn_smiles = fix_attempt
                all_tool_results[(rxn_smiles, fix_result.name)] = fix_result
                if fixed_rxn_smiles is not None:
                    return await self.grade_one_async(
                        fixed_rxn_smiles,
                        fix_attempted=True,
                        previous_tool_results=all_tool_results,
                    )
            return None

        previous_tool_results = previous_tool_results or {}
        all_tool_results: ToolResultsDict = previous_tool_results.copy()
        should_skip_remaining = False
        skip_reason = ""

        for checker in self.checkers:
            if (rxn_smiles, checker.name) in previous_tool_results:
                continue
            if should_skip_remaining:
                result = checker.skipped(skip_reason)
            else:
                try:
                    is_cacheable = isinstance(checker, CacheableReactionChecker)
                    result = None
                    if is_cacheable:
                        result = checker.check_cache(rxn_smiles)  # noqa
                    if result is None:
                        result = checker.run(rxn_smiles, all_tool_results)
                except Exception as exc:
                    result = checker.errored(
                        f"{checker.name} raised an unexpected error: {exc}",
                        traceback_text=traceback.format_exc(),
                    )

                if self._should_skip_remaining(checker, result):
                    should_skip_remaining = True
                    skip_reason = f"Skipped after hard failure in {checker.name}."

            # prefix.append((rxn_smiles, checker.name, result))
            all_tool_results[(rxn_smiles, checker.name)] = result

            # Reaction fixing / infilling of byproducts
            # If LLM fixing returned a value, call new grade_one with new rxn and
            # it's main tool result list will start from the new rxn
            if checker.name == "exact_match":
                none_or_fixed_and_graded = await maybe_call_fixer()
                if none_or_fixed_and_graded is not None:
                    return none_or_fixed_and_graded
        if (
            not self.checkers
        ):  # We allow using no tools while still attempting to fix the equation
            none_or_fixed_and_graded = await maybe_call_fixer()
            if none_or_fixed_and_graded is not None:
                return none_or_fixed_and_graded

        return await self._finalize_grade_async(
            rxn_smiles, all_tool_results, previous_tool_results
        )

    def grade_one(
        self,
        rxn_smiles: str,
        *,
        fix_attempted: bool = False,
        prefix_results: list[tuple[str, str, ToolResult]] | None = None,
    ) -> ReactionGrade:
        return _run_coroutine_sync(
            self.grade_one_async(
                rxn_smiles,
                fix_attempted=fix_attempted,
                previous_tool_results=prefix_results,
            )
        )

    def grade(self, rxn_smiles_list: list[str]) -> Iterator[ReactionGrade]:
        for rxn_smiles in rxn_smiles_list:
            res = self.grade_one(rxn_smiles)
            yield res


def build_default_pipeline(
    config: PipetteConfig | None = None,
    *,
    judge: AsyncLLMJudge | None = None,
    reaction_fixer: AsyncLLMReactionFixer | None = None,
    checker_factories: dict[str, CheckerFactory] | None = None,
) -> GradingPipeline:
    """Builds a `GradingPipeline`
    Uses the provided or default `PipetteConfig`, and makes a `GradingPipeline` with the
    tools specified in the config's `tool_list`. `tool_list` can be "all" or a list of names (see the `name`
    attribute of `ReactionChecker`).
    """
    # Edit this function when adding new `ReactionChecker`s
    config = config or PipetteConfig()
    possible_checker_factories = checker_factories or {
        "basic_smiles_validation": lambda _: BasicSmilesValidationChecker(),
        "exact_match": lambda _: ExactMatchChecker(),
        "graph_based_balancing": lambda _: todo,
        "atom_mapping": lambda _: todo,
        "charge_conservation": lambda _: ChargeConservationChecker(),
        "mass_conservation": lambda config: MassConservationChecker(config),
        "reaction_energy": lambda config: ReactionEnergyChecker(
            config,
            database=(
                str(config.tools_settings.reaction_energy.database)
                if config.tools_settings.reaction_energy.database is not None
                else None
            ),
        ),
    }
    selected_names = resolve_tool_list(
        config.tool_list,
        list(possible_checker_factories),
        use_dft=config.settings.use_dft,
    )
    checkers = [possible_checker_factories[name](config) for name in selected_names]
    return GradingPipeline(
        checkers=checkers,
        config=config,
        judge=judge,
        reaction_fixer=reaction_fixer,
    )
