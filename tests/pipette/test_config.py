###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from textwrap import dedent

from flask_tools.pipette.config import PipetteConfig
from flask_tools.pipette.judge import LLMJudge
from flask_tools.pipette.reaction_fixer import LLMReactionFixer


def test_pipette_config_from_yaml_loads_nested_sections(tmp_path) -> None:
    # Test of the yaml to config conversion is ok, and that paths are relative to the yaml file
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        dedent(
            """
            mode: ai
            tool_list:
              - basic_smiles_validation
              - reaction_energy
            tools_settings:
              reaction_energy:
                database: fake_molecule_energies.tsv
            llm_judge:
              allow_fail:
                - reaction_energy
              enable_atom_mapping_dict_in_prompt: true
              url: https://example.test/v1/
              model: custom-model
              api_key: sk-test
              prompt_path: judge-prompt.txt
              prompt: inline judge prompt
            llm_reaction_fixer:
              enabled: true
              model: gpt-5.5
              api_key: sk-fix
              prompt_path: fixer-prompt.txt
              prompt: inline fixer prompt
            settings:
              stop_on_hard_fail: false
              mass_tolerance_atoms: 2
              reaction_energy_max_ev_mol: 9.5
              use_dft: true
              use_fixing: false
            solvent_catalog_path: solvents.tsv
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    config = PipetteConfig.from_yaml(config_path)

    # Test that paths are relative to the yaml file
    assert (
        config.tools_settings.reaction_energy.database
        == (tmp_path / "fake_molecule_energies.tsv").resolve()
    )
    assert config.llm_judge.prompt_path == (tmp_path / "judge-prompt.txt").resolve()
    assert (
        config.llm_reaction_fixer.prompt_path
        == (tmp_path / "fixer-prompt.txt").resolve()
    )
    assert config.solvent_catalog_path == (tmp_path / "solvents.tsv").resolve()

    # Test other attrs are set correctly
    assert config.tool_list == ["basic_smiles_validation", "reaction_energy"]
    assert config.llm_judge.allow_fail == ["reaction_energy"]
    assert config.llm_judge.enable_atom_mapping_dict_in_prompt is True
    assert config.llm_judge.url == "https://example.test/v1/"
    assert config.llm_judge.model == "custom-model"
    assert config.llm_judge.api_key == "sk-test"
    assert config.llm_judge.prompt == "inline judge prompt"
    assert config.llm_reaction_fixer.enabled is True
    assert config.llm_reaction_fixer.model == "gpt-5.5"
    assert config.llm_reaction_fixer.api_key == "sk-fix"
    assert config.llm_reaction_fixer.prompt == "inline fixer prompt"
    assert config.settings.stop_on_hard_fail is False
    assert config.settings.mass_tolerance_atoms == 2
    assert config.settings.reaction_energy_max_ev_mol == 9.5
    assert config.settings.use_dft is True
    assert config.settings.use_fixing is False
