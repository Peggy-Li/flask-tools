###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

"""Integration tests"""

from __future__ import annotations

import dataclasses
import json
import sys
from collections import namedtuple
from copy import copy
from dataclasses import asdict
from typing import Sequence

import pytest
from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs import SequenceType
from rdkit.Chem.ChemUtils import TemplateExpand

from flask_tools.pipette import ToolResult, ToolStatus, FinalGrade
from flask_tools.pipette.constants import FinalGrade, ReactionGrade
from flask_tools.pipette.grade_rxn import grade_reaction, main
from flask_tools.pipette.config import load_config, ConfigType, PipetteConfig
from flask_tools.pipette.graph_rxn_mapper.subtractive_reaction_mapper_new import (
    GraphBasedBalancerResultDetails,
    AtomMappingResultDetails,
)
from flask_tools.pipette.pipeline import GradingPipeline
from flask_tools.pipette.reaction_fixer import ReactionFixResultDetails
from flask_tools.pipette.verifiers import ChargeConservationChecker, ReactionChecker
from conftest import SpyChecker
from flask_tools.pipette.verifiers.charge import ChargeResultDetails
from flask_tools.pipette.verifiers.mass import MassResultDetails

# Caffeine
CAFFEINE_ORIGINAL_REACTION_SMI = (
    "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
)
CAFFEINE_FIXED_REACTION_SMIS = (
    "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.I",
    "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
)


@dataclasses.dataclass()
class RxnToTest:
    should_pass: bool
    orig_smi: str
    possible_fixed_smis: (
        Sequence[str] | None
    )  # None if it's not meant to be balanceable
    accepted_grades: Sequence[FinalGrade]


rxns_to_test = {
    (CAFFEINE := "caffeine"): RxnToTest(
        True,
        orig_smi=CAFFEINE_ORIGINAL_REACTION_SMI,
        possible_fixed_smis=CAFFEINE_FIXED_REACTION_SMIS,
        accepted_grades=[FinalGrade.LIKELY],
    ),
    # mislabeled rxn: common solvent as product pulled from flask-copilot retrosynthesis
    (TOLUENE := "toluene"): RxnToTest(
        False,
        orig_smi="Cl[SiH](C)C.BrCCCCCCO[SiH](C)C>>Cc1ccccc1",
        possible_fixed_smis=None,
        accepted_grades=[FinalGrade.IMPOSSIBLE],
    ),
}


@pytest.mark.llm_query
@pytest.mark.parametrize(
    "rxn_name",
    [
        CAFFEINE,
        TOLUENE,
    ],
)
def test_calls_fixer_caffeine_llm_judge(rxn_name: str) -> None:
    # def test_calls_fixer_caffeine_llm_judge(install_mock_llm_services, config_name) -> None:
    rxn_to_test = rxns_to_test[rxn_name]

    # Short tests that checks fixed smiles and final grade
    config = load_config(ConfigType.LLM_JUDGE_NO_DFT)
    results = grade_reaction([rxn_to_test.orig_smi], config=config)
    assert len(results) == 1
    result: ReactionGrade = results[0]
    tool_results: list[ToolResult] = result.results
    tool_names = [tool_result.name for tool_result in result.results]
    assert "llm_reaction_fix" in tool_names

    fix_result = next(tr for tr in tool_results if tr.name == "llm_reaction_fix")
    assert fix_result.data.original_reaction_smiles == rxn_to_test.orig_smi
    if rxn_to_test.possible_fixed_smis:
        assert (
            fix_result.data.fixed_reaction_smiles in rxn_to_test.possible_fixed_smis
        ), fix_result
        if rxn_name == CAFFEINE:
            assert fix_result.data.added_products in (
                ["I"],
                ["[H+]", "[I-]"],
            ), fix_result

    assert (
        result.final_grade in rxn_to_test.accepted_grades
    ), f"{(result.final_grade, fix_result)}"


expected_tool_call_order = {
    (NO_GRAPH_BALANCER_TOOLS := "no_graph_balancer"): [
        "basic_smiles_validation",
        "exact_match",
        "llm_reaction_fix",
        # This assumes llm_reaction_fix produced a changed rxn. Otherwise, it would not start over
        "basic_smiles_validation",
        "exact_match",
        "charge_conservation",
        "mass_conservation",
        "reaction_energy",
    ],
    (WITH_GRAPH_BALANCER_TOOLS := "graph_balancer"): [
        "basic_smiles_validation",
        "exact_match",
        "graph_based_balancing",
        "basic_smiles_validation",
        "exact_match",
        "llm_reaction_fix",  # T
        "basic_smiles_validation",
        "exact_match",
        "llm_atom_mapping",
        "charge_conservation",
        "mass_conservation",
        "reaction_energy",
    ],
}


@pytest.mark.llm_query
@pytest.mark.parametrize(
    "tool_set_name", [NO_GRAPH_BALANCER_TOOLS, WITH_GRAPH_BALANCER_TOOLS]
)
def test_pipeline_fixed_reaction(
    tool_set_name: str,
    tests_relative_path,
) -> None:
    # In-depth test that checks that ever single expected tool is called
    # Still calls LLM judge so the rxn has to be reasonable
    # A simple rxn without dimerization
    # original = "CCO>>C=C"
    # fixed = "CCO>>C=C.O"
    # aldol condensation of acetaldehyde to crotonaldehyde, which both dimerizes and drops water
    original = "CC=O>>CC=CC=O"
    graph_balanced_hopefully = "CC=O.CC=O>>CC=CC=O"
    fixed = "CC=O.CC=O>>CC=CC=O.O"

    smiles_validation = SpyChecker(
        "basic_smiles_validation",
        lambda rxn_smiles, _: ToolResult(
            name="basic_smiles_validation",
            status=ToolStatus.PASS,
            data=None,
            comment=f"parsed {rxn_smiles}",
        ),
    )
    exact = SpyChecker(
        "exact_match",
        lambda rxn_smiles, _: ToolResult(
            name="exact_match",
            status=ToolStatus.UNKNOWN,
            data=None,
            comment="No exact match.",
        ),
    )
    charge = SpyChecker(
        "charge_conservation",
        lambda rxn_smiles, _: ToolResult(
            name="charge_conservation",
            status=ToolStatus.PASS,
            data=ChargeResultDetails(charge_difference=0),
            comment="Charge is conserved.",
        ),
        stops_on_fail=True,
    )
    mass = SpyChecker(
        "mass_conservation",
        lambda rxn_smiles, _: ToolResult(
            name="mass_conservation",
            status=ToolStatus.FAIL if rxn_smiles == original else ToolStatus.PASS,
            data=None,
            comment="Mass failed." if rxn_smiles == original else "Mass passed.",
        ),
    )

    reaction_energy = SpyChecker(
        "reaction_energy",
        lambda rxn_smiles, _: ToolResult(
            name="reaction_energy",
            status=ToolStatus.PASS,
            comment="Energy passed.",
            data=None,
        ),
    )

    graph_balancer = SpyChecker(
        "graph_based_balancing",
        lambda rxn_smiles, _: ToolResult(
            name="graph_based_balancing",
            status=ToolStatus.PASS,
            comment="Passed",
            data=GraphBasedBalancerResultDetails(
                original_reaction_smiles=original,
                graph_balanced_reaction_smiles=graph_balanced_hopefully,
                graph_mapped_reaction_smiles="",
                final_balanced_reaction_smiles="",
                objective_value=100,
                mapper_status="",
                reasoning_summary="",
            ),
        ),
    )

    llm_atom_mapper = SpyChecker(
        "llm_atom_mapping",
        lambda rxn_smiles, _: ToolResult(
            name="llm_atom_mapping",
            status=ToolStatus.PASS,
            comment="Passed",
            data=AtomMappingResultDetails(
                input_reaction_smiles=rxn_smiles,
                mapped_reaction_smiles="",
                product_to_reactant=[],
                confidence=0.9,
                reasoning_summary="",
            ),
        ),
    )

    class StubReactionFixer:
        # Have a fixed LLM fixer step to better test LLM judge step
        def __init__(self) -> None:
            self.calls: list[str] = []

        def fix(
            self, rxn_smiles: str, results: list[ToolResult]
        ) -> ReactionFixResultDetails:
            self.calls.append(rxn_smiles)
            assert [result.name for result in results] == expected_tool_call_order[
                tool_set_name
            ][: expected_tool_call_order[tool_set_name].index("llm_reaction_fix")]
            return ReactionFixResultDetails(
                original_reaction_smiles=rxn_smiles,
                fixed_reaction_smiles=fixed,
                removed_agents=[],
                added_reactants=[],
                removed_products=[],
                added_products=["O"],
                reasoning_summary="Balanced both sides with water.",
            )

    fixer = StubReactionFixer()
    if tool_set_name == NO_GRAPH_BALANCER_TOOLS:
        tool_list = [smiles_validation, exact, charge, mass, reaction_energy]
    elif tool_set_name == WITH_GRAPH_BALANCER_TOOLS:
        tool_list = [
            smiles_validation,
            exact,
            graph_balancer,
            llm_atom_mapper,
            charge,
            mass,
            reaction_energy,
        ]
    else:
        raise ValueError(f"{tool_set_name=}")
    pipeline = GradingPipeline(
        checkers=tool_list,
        config=PipetteConfig(mode="exact"),
        reaction_fixer=fixer,  # noqa
    )

    result = next(pipeline.grade([original]))

    # Manual sanity check (run test twice): two runs can have all the same tool results, but different final grade
    # dbg_file = tests_relative_path / "tool_res.json"
    # result2 = copy(result)
    # result2.final_grade = None
    # result2.short_comment = None
    # result2.short_comment = None
    # result2.comment = None
    # res_dict = asdict(result2)
    # # Remove the llm judge answer
    # if not dbg_file.exists():
    #     prev_res_dict = None
    #     with open(dbg_file, "w") as f:
    #         json.dump(res_dict, f, indent=2)
    # else:
    #     with open(dbg_file, "r") as f:
    #         prev_res_dict = json.load(f)
    #
    # if prev_res_dict is not None:
    #     # assert res_dict == prev_res_dict
    #     assert json.loads(json.dumps(res_dict)) == prev_res_dict

    assert [tool.name for tool in result.results] == expected_tool_call_order[
        tool_set_name
    ]
    if tool_set_name == NO_GRAPH_BALANCER_TOOLS:
        assert smiles_validation.calls == [original, fixed]
        assert exact.calls == [original, fixed]
    else:
        assert smiles_validation.calls == [original, graph_balanced_hopefully, fixed]
        assert exact.calls == [original, graph_balanced_hopefully, fixed]
    assert fixer.calls == [graph_balanced_hopefully]
    assert charge.calls == [fixed]
    assert mass.calls == [fixed]
    assert reaction_energy.calls == [fixed]

    llm_fix_i = expected_tool_call_order[tool_set_name].index("llm_reaction_fix")
    if tool_set_name == NO_GRAPH_BALANCER_TOOLS:
        assert result.results[llm_fix_i].data.original_reaction_smiles == original
    else:
        assert (
            result.results[llm_fix_i].data.original_reaction_smiles
            == graph_balanced_hopefully
        )
    assert result.results[llm_fix_i].data.fixed_reaction_smiles == fixed
    try:
        assert result.final_grade == FinalGrade.LIKELY, result
    except AssertionError as e:
        # Capturing this assertion to pretty print debug info. The LLM will occasionally fail reaction.
        print(str(result))
        raise e

    # Check that ReactionGrade.__str__ works
    s = str(result)
    assert s
