###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromInchi

from .base import ReactionChecker
from ..config import PipetteConfig
from ..constants import (
    ToolResult,
    ToolStatus,
    AllowedUnbalancedMolecule,
    Confidence,
    RxnSide,
    ReactionMassImbalanceExplanation,
    ToolResultDetails,
)
from ..smiles import parse_reaction_smi


def atom_bag(
    molecules: list[object], *, explicit_hydrogens: bool = False
) -> dict[str, int]:
    bag: defaultdict[str, int] = defaultdict(int)
    prepared = [Chem.AddHs(mol) if explicit_hydrogens else mol for mol in molecules]
    for molecule in prepared:
        for atom in molecule.GetAtoms():
            bag[atom.GetSymbol()] += 1
    return dict(bag)


def bag_difference(
    products: dict[str, int], reactants: dict[str, int]
) -> dict[str, int]:
    delta: defaultdict[str, int] = defaultdict(int)
    for symbol, count in products.items():
        delta[symbol] += count
    for symbol, count in reactants.items():
        delta[symbol] -= count
    return {symbol: count for symbol, count in delta.items() if count}


def formula_mass(formula: dict[str, int]) -> float:
    table = Chem.GetPeriodicTable()
    return sum(
        table.GetAtomicWeight(symbol) * count for symbol, count in formula.items()
    )


def formula_from_smiles(smiles: str) -> dict[str, int]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid solvent SMILES: {smiles!r}")
    return atom_bag([molecule], explicit_hydrogens=True)


def element_delta(rxn_smiles: str, explicit_hydrogens: bool = False) -> dict[str, int]:
    """Products - Reactants."""
    reactants, _, products = parse_reaction_smi(rxn_smiles)
    reactant_bag = atom_bag(reactants, explicit_hydrogens=explicit_hydrogens)
    product_bag = atom_bag(products, explicit_hydrogens=explicit_hydrogens)
    return bag_difference(product_bag, reactant_bag)


def check_mass_conservation(
    rxn_smiles: str, explicit_hydrogens: bool = False
) -> tuple[bool, str]:
    try:
        delta = element_delta(rxn_smiles, explicit_hydrogens=explicit_hydrogens)
    except Exception as exc:
        return False, f"Error while parsing reaction SMILES: {exc}"

    if delta:
        return False, f"Net difference in atom counts: {delta}"
    return True, "Reaction is mass-conserved."


def find_basic_missing(delta: dict[str, int]) -> list[ReactionMassImbalanceExplanation]:
    """Basic exceptions like missing hydrogens."""
    matches: list[ReactionMassImbalanceExplanation] = []
    if len(delta) == 1 and "H" in delta:
        matches.append(
            ReactionMassImbalanceExplanation(
                name="hydrogen",
                smiles="H",
                missing_side=RxnSide.PRODUCTS if delta["H"] < 0 else RxnSide.REACTANTS,
                mass_amu=round(abs(delta["H"]) * 1.00794, 4),
                confidence=Confidence.HIGH,
            )
        )
    return matches


def find_missing_product_matches(
    delta: dict[str, int],
    rules: list[AllowedUnbalancedMolecule],
) -> list[ReactionMassImbalanceExplanation]:
    matches: list[ReactionMassImbalanceExplanation] = []
    for rule in rules:
        negative_formula = {element: -count for element, count in rule.formula.items()}
        if delta == negative_formula:
            missing_side = RxnSide.PRODUCTS
        elif delta == rule.formula:
            missing_side = RxnSide.REACTANTS
        else:
            continue
        matches.append(
            ReactionMassImbalanceExplanation(
                name=rule.name,
                smiles=rule.smiles,
                confidence=rule.confidence,
                missing_side=missing_side,
                mass_amu=round(formula_mass(rule.formula), 4),
            )
        )
    return matches


def load_solvent_rules(solvents_path: Path) -> list[AllowedUnbalancedMolecule]:
    rules: list[AllowedUnbalancedMolecule] = []

    with solvents_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            inchi = (row.get("inchi") or "").strip()
            smiles = MolToSmiles(MolFromInchi(inchi))
            name = (row.get("common_name") or "").strip()
            try:
                conf = (row.get("commonness") or "").strip().lower() or "low"
                confidence = Confidence(conf)
            except ValueError as e:
                print(f"Confidence was {conf} must be one of {list(Confidence)}")
            if not smiles or not name:
                continue

            rules.append(
                AllowedUnbalancedMolecule(
                    name=name,
                    smiles=smiles,
                    formula=formula_from_smiles(smiles),
                    confidence=confidence,
                )
            )

    return rules


class MassResultDetails(ToolResultDetails):
    mass_difference_amu: float  # Product mass - reactant mass
    element_difference: dict[str, int]  # Product elements - reactant elements
    imbalanced_molecules: list[
        ReactionMassImbalanceExplanation
    ]  # Molecules that cause or could fit the difference
    imbalanced_molecule_confidence: (
        Confidence | None
    )  # None if balanced, otherwise the confidence that a reaction is ok based on the imbalanced molecule. IE, if the imbalance is a common solvent, or a small byproduct, we can likely ignore it, so the confidence is high.


class MassConservationChecker(ReactionChecker):
    """Checks the reactants and products have equal mass. Certain mass differences are allowed if it corresponds
    with common solvents or a missing H
    ToolResult example:
        ToolResult(
            name="mass_conservation",
            status=ToolStatus.PASS,
            data=MassResultDetails(
                mass_difference_amu: 1.00794,
                element_difference={"H": -1},
                imbalanced_molecules=[
                    ReactionMassImbalanceExplanation(
                        name="hydrogen",
                        smiles="H",
                        missing_side="product",
                        mass_amu=1.00794,
                        confidence=Confidence.HIGH,
                    )
                ],
                imbalanced_molecule_confidence=Confidence.HIGH,
            )
            comment="Element counts are conserved.",
        )
    """

    name = "mass_conservation"

    def __init__(self, config: PipetteConfig) -> None:
        self.config = config
        self.missing_product_rules = load_solvent_rules(config.solvent_catalog_path)

    def run(
        self, rxn_smiles: str, context: ToolResultsDict | None = None
    ) -> ToolResult:
        try:
            delta = element_delta(rxn_smiles, explicit_hydrogens=True)
        except Exception as exc:
            return ToolResult(
                name=self.name,
                status=ToolStatus.ERROR,
                data=None,
                comment=f"Mass conservation could not be evaluated: {exc}",
            )

        if not delta:
            return ToolResult(
                name=self.name,
                status=ToolStatus.PASS,
                data=MassResultDetails(
                    mass_difference_amu=0.0,
                    element_difference={},
                    imbalanced_molecules=[],
                    imbalanced_molecule_confidence=None,
                ),
                comment="Element counts are conserved.",
            )

        matches = find_basic_missing(delta)
        if not matches:
            matches = find_missing_product_matches(delta, self.missing_product_rules)
        if matches:
            best_match = matches[0]
            return ToolResult(
                name=self.name,
                status=ToolStatus.PASS,
                data=MassResultDetails(
                    mass_difference_amu=best_match.mass_amu,
                    element_difference=delta,
                    imbalanced_molecules=matches,
                    imbalanced_molecule_confidence=best_match.confidence,
                ),
                comment=(
                    "Element counts are not conserved, but the difference matches a "
                    f"common omitted species or solvent: {best_match.name}."
                ),
            )

        mass_difference = round(
            abs(
                formula_mass({element: abs(count) for element, count in delta.items()})
            ),
            4,
        )
        return ToolResult(
            name=self.name,
            status=ToolStatus.FAIL,
            data=MassResultDetails(
                mass_difference_amu=mass_difference,
                element_difference=delta,
                imbalanced_molecules=[],
                imbalanced_molecule_confidence=None,
            ),
            comment="Element counts are not conserved and do not match a configured common omission.",
        )
