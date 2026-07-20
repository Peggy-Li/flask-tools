###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem.rdchem import Mol


def split_reaction_smiles(rxn_smiles: str) -> tuple[str, str, str]:
    if ">>" in rxn_smiles:
        reactants, products = rxn_smiles.split(">>", maxsplit=1)
        return reactants, "", products

    parts = rxn_smiles.split(">")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]

    raise ValueError(
        "Reaction SMILES must contain either '>>' or exactly two '>' separators."
    )


def parse_mol_smi_w_periods(smiles_block: str, ret_mol=True) -> list[Mol] | list[str]:
    """Parse XXX.YYY.ZZZ into a list of RDKit molecules."""
    if not smiles_block:
        return []

    molecules: list[Mol] = []
    for token in smiles_block.split("."):
        if ret_mol:
            molecule = Chem.MolFromSmiles(token)
        else:
            molecule = token
        if molecule is None:
            raise ValueError(f"Invalid SMILES component: {token!r}")
        molecules.append(molecule)
    return molecules


def parse_reaction_smi(
    rxn_smiles: str, ret_mol=True
) -> tuple[list[Mol], list[Mol], list[Mol]] | tuple[list[str], list[str], list[str]]:
    reactants, agents, products = split_reaction_smiles(rxn_smiles)
    return (
        parse_mol_smi_w_periods(reactants, ret_mol=ret_mol),
        parse_mol_smi_w_periods(agents, ret_mol=ret_mol),
        parse_mol_smi_w_periods(products, ret_mol=ret_mol),
    )


def canonicalize_smiles(smiles: str, allow_fail=False) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        if allow_fail:
            return smiles
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True)


def canonicalize_reaction_smiles(
    rxn_smiles: str, *, include_agents: bool = True
) -> str:
    """Canonicalize a reaction SMILES string, stripping reagents (C in A>C>B) by default.
    Accepts A>B or A>C>B
    Returns string A>B
    """
    reactants, agents, products = split_reaction_smiles(rxn_smiles)

    def canonicalize_side(side: str) -> str:
        if not side:
            return ""
        entries = [canonicalize_smiles(item) for item in side.split(".") if item]
        return ".".join(sorted(entries))

    canonical_reactants = canonicalize_side(reactants)
    canonical_products = canonicalize_side(products)
    if include_agents:
        canonical_agents = canonicalize_side(agents)
        return f"{canonical_reactants}>{canonical_agents}>{canonical_products}"
    return f"{canonical_reactants}>>{canonical_products}"


def smiles_to_inchi(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES component: {smiles!r}")

    inchi = Chem.MolToInchi(molecule)
    if not inchi:
        raise ValueError(
            f"Could not generate an InChIKey for SMILES component: {smiles!r}"
        )
    return inchi


def mol_from_side(side: str) -> Chem.Mol:
    if not side:
        return Chem.Mol()
    mol = Chem.MolFromSmiles(side, sanitize=True)
    if mol is None:
        raise ValueError(f"Could not parse reaction side: {side!r}")
    return mol


def clear_atom_maps_from_side(side: str) -> str:
    mol = mol_from_side(side)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return canonical_side_smiles(mol)


def clear_atom_maps_from_reaction(
    reaction_smiles: str, keep_agents: bool = False
) -> str:
    reactants, agents, products = split_reaction_smiles(reaction_smiles)
    cleared_reactants = clear_atom_maps_from_side(reactants)
    cleared_products = clear_atom_maps_from_side(products)
    if keep_agents:
        cleared_agents = clear_atom_maps_from_side(agents)
        return f"{cleared_reactants}>{cleared_agents}>{cleared_products}"
    return f"{cleared_reactants}>>{cleared_products}"


def canonical_side_smiles(mol: Chem.Mol) -> str:
    """Canonicalize a reaction side as a sorted multiset of mapped fragments."""
    if mol.GetNumAtoms() == 0:
        return ""
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    smiles = [
        Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
        for fragment in fragments
    ]
    return ".".join(sorted(smiles))
