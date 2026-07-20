###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

from ..constants import FinalGrade, ToolResult, ToolStatus, ToolResultsDict


class ReactionChecker(ABC):
    name: str
    stops_on_fail: bool = False

    @abstractmethod
    def run(self, rxn_smiles: str, context: ToolResultsDict) -> ToolResult:
        raise NotImplementedError

    async def arun(self, rxn_smiles: str, context: ToolResultsDict) -> ToolResult:
        """A function to be overwritten by tools that benefit from async. Actually what's called in pipeline.
        This calls the sync run method by default.
        Leaving `run()` in ReactionChecker makes writing non async tools simpler, and there are more non async tools.
        For async classes, you can define run like this:
        ```
        def run(self, rxn_smiles: str, context: ToolResultsDict) -> ToolResult:
            return _run_coroutine_sync(self.arun(rxn_smiles, context))
        ```
        See LLMAtomMapper for an example of a class that uses this.
        """
        return self.run(rxn_smiles, context)

    def skipped(self, reason: str) -> ToolResult:
        return ToolResult(
            name=self.name,
            status=ToolStatus.NOT_RUN,
            comment="Tool was not run.",
            data=None,
            skipped_reason=reason,
        )

    def errored(self, reason: str, traceback_text: str | None = None) -> ToolResult:
        return ToolResult(
            name=self.name,
            status=ToolStatus.ERROR,
            data=None,
            comment=reason,
            skipped_reason=traceback_text,  # This field should be deleted before making LLM prompts
        )


class CacheableReactionChecker(ReactionChecker):
    def __init__(self, database: str | None = None) -> None:
        self.database = database

    @abstractmethod
    def check_cache(self, rxn_smiles: str) -> None | ToolResult:
        # Remember to canonicalize first
        raise NotImplementedError
