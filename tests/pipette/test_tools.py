###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from textwrap import dedent
import json
import sys
from pathlib import Path

import pytest


from rdkit.Chem import MolFromSmiles
from rdkit.Chem.inchi import MolToInchiKey, MolToInchi

from flask_tools.pipette.verifiers.reaction_energy import (
    MoleculeEnergyStore,
)


# Should this be removed completely b/c DFT calculation is not implemented rn?
def test_molecule_energy_store_from_csv(tmp_path) -> None:
    smiles_to_inchi = lambda x: MolToInchi(MolFromSmiles(x))
    path = tmp_path / "fake_molecule_energies.tsv"
    ethanol_inchi = smiles_to_inchi(eth_smi := "CCO")
    acetaldehyde_inchi = smiles_to_inchi(acetal_smi := "CC=O")
    path.write_text(
        dedent(
            f"""
            inchi\tenergy_ev_mol
            {ethanol_inchi}\t-7.0
            {acetaldehyde_inchi}\t-10.0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    store = MoleculeEnergyStore.from_csv(path)
    reactants_energy = store.lookup(eth_smi).energy_ev
    products_energy = store.lookup(acetal_smi).energy_ev
    assert reactants_energy == -7.0
    assert products_energy - reactants_energy == -3.0


# Could move this to conftest or some tests/utils.py if another test needs this.
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
DATA_FILE = TESTS_DIR / "data" / "atom_map_rxns.jsonl"
EXPECTED_DIR = TESTS_DIR / "expected_atom_map_res"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask_tools.pipette.graph_rxn_mapper.subtractive_reaction_mapper_v3 import main


def load_cases() -> list[object]:
    cases: list[object] = []
    for line_number, raw_line in enumerate(DATA_FILE.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        entry = json.loads(raw_line)
        rxn_smiles = entry["rxn_smiles"]
        expected_output = entry["expected_output"]
        case_id = entry.get("id", f"line-{line_number}")
        cases.append(pytest.param(rxn_smiles, expected_output, id=case_id))
    return cases


@pytest.mark.parametrize(("rxn_smiles", "expected_output"), load_cases())
def test_atom_mapper_output(
    rxn_smiles: str, expected_output: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # Tests the raw atom mapper CLI version
    expected_path = EXPECTED_DIR / expected_output
    assert expected_path.exists(), f"Missing expected output snapshot: {expected_path}"
    exit_code = main([rxn_smiles])
    captured = capsys.readouterr()
    assert exit_code == 0
    actual_output = captured.out
    assert actual_output == expected_path.read_text()
