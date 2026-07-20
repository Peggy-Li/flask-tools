###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from .constants import FinalGrade, ReactionGrade, ToolResult, ToolStatus

__all__ = [
    "FinalGrade",
    "ReactionGrade",
    "ToolResult",
    "ToolStatus",
    "grade_reaction",
]


def __getattr__(name: str):
    if name == "grade_reaction":
        from .grade_rxn import grade_reaction

        return grade_reaction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
