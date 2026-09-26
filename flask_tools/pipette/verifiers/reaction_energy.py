###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol, Iterable
from typing_extensions import TypedDict

from .base import CacheableReactionChecker
from ..config import PipetteConfig
from ..constants import (
    FinalGrade,
    ToolResult,
    ToolStatus,
    ToolResultsDict,
    ToolResultDetails,
)
from ..smiles import (
    split_reaction_smiles,
    smiles_to_inchi,
    parse_reaction_smi,
    canonicalize_smiles,
    canonicalize_reaction_smiles,
)


# Not exposing to users, keeping as a dataclass?
@dataclass
class MoleculeEnergyRecord:
    smi: str
    energy_ev: float
    source: str


class ReactionEnergyMetaDict(TypedDict):
    reactants: dict[str, float]
    products: dict[str, float]


@dataclass
class ReactionEnergyRecord:
    energy_difference_ev_mol: float
    source: dict[str, str]
    metadata: ReactionEnergyMetaDict


class DFTExecutor(Protocol):
    def run(self, rxn_smiles: str) -> MoleculeEnergyRecord: ...


@dataclass
class CSVCache:
    path: Path

    @staticmethod
    def get_rows_from_csv(path) -> dict[str, dict[str, object]]:
        current_rows: dict[str, dict[str, object]] = {}
        if not path.exists():
            return {}
        delim = "\t" if str(path).endswith("tsv") else ","
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delim)
            for row in reader:
                inchi = (row.get("inchi") or "").strip()
                raw_energy = (row.get("energy_ev_mol") or "").strip()
                if not inchi or not raw_energy:
                    continue
                current_rows[inchi] = {
                    "energy_ev_mol": float(raw_energy),
                }
            if not len(current_rows):
                print(
                    f"Warning: {path} provided but could not parse rows. Used delimiter '{delim}'"
                )
        return current_rows

    def update_csv(self, rows_by_inchi: dict[str, dict[str, object]]) -> None:
        current_rows: dict[str, dict[str, object]] = self.get_rows_from_csv(self.path)

        current_rows.update(rows_by_inchi)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["inchi", "energy_ev_mol"],
                delimiter="\t",
            )
            writer.writeheader()
            for inchi in sorted(current_rows):
                writer.writerow(
                    {
                        "inchi": inchi,
                        "energy_ev_mol": current_rows[inchi]["energy_ev_mol"],
                    }
                )


@dataclass
class MoleculeEnergyStore:
    path: Path
    energies_by_inchi: dict[str, float]
    csv_cache: CSVCache | None = None

    @classmethod
    def from_csv(cls, path: str | Path) -> MoleculeEnergyStore:
        resolved_path = Path(path).expanduser().resolve()
        rows = CSVCache.get_rows_from_csv(resolved_path)
        energies_by_inchi: dict[str, float] = {
            k: v["energy_ev_mol"] for k, v in rows.items()
        }

        return cls(
            path=resolved_path,
            energies_by_inchi=energies_by_inchi,
            csv_cache=CSVCache(resolved_path),
        )

    def save(
        self, records: MoleculeEnergyRecord | Iterable[MoleculeEnergyRecord]
    ) -> None:
        if not records:
            return
        if isinstance(records, MoleculeEnergyRecord):
            records = [records]
        update: dict[str, dict[str, object]] = {}
        for record in records:  # noqa
            inchi = smiles_to_inchi(record.smi)
            update[inchi] = {
                "inchi": inchi,
                "energy_ev_mol": record.energy_ev,
            }
            self.energies_by_inchi[inchi] = record.energy_ev
        if self.csv_cache is not None:
            self.csv_cache.update_csv(update)

    def lookup(self, mol_smi: str) -> MoleculeEnergyRecord | None:
        energy = self.energies_by_inchi.get(smiles_to_inchi(mol_smi), None)
        if energy is None:
            return None
        else:
            return MoleculeEnergyRecord(
                smi=mol_smi, energy_ev=energy, source="molecule_energy_csv"
            )


class ReactionEnergyResultDetails(ToolResultDetails):
    energy_difference_ev_mol: float  # products energy - reactants energy in eV
    source: str  # Name of source, i.e. database or method
    metadata: ReactionEnergyMetaDict  # A dictionary of energy per molecule for both reactants and products. E.g.: {"reactants": {"smi1": 0.123, ...} "products": ...}


class ReactionEnergyChecker(CacheableReactionChecker):
    """Checks if reaction energy is favorable, looking at difference between the product molecules' energy and the
    reactants'. Threshold is set by the config value `reaction_energy_max_ev_mol` with a default of 0.2eV to include
    endothermic reactions.

    Expects a DFTExecutor that should return a ReactionEnergyRecord. This is translated to a ToolResult in _result_from_record

    ToolResult example:
        ToolResult(
            name=self.name,
            status=ToolStatus.PASS if passed else ToolStatus.FAIL,
            data=ReactionEnergyResultDetails(
                energy_difference_ev_mol = 0.123
                source = record.source,  # dft or database name
                metadata = {"reactants": {"smi1": 0.123, ...} "products": ...}
            )
            comment=(
                "Reaction energy is within the allowed threshold."
                if passed
                else "Reaction energy exceeds the allowed threshold."
            ),
        )
    """

    name = "reaction_energy"

    def __init__(
        self,
        config: PipetteConfig,
        database: str | MoleculeEnergyStore | None = None,
        dft_executor: DFTExecutor | None = None,
    ) -> None:
        database_str = database if isinstance(database, str) else None
        super().__init__(database=database_str)
        self.config = config
        if not isinstance(database, str):
            self.store = database
        else:
            if database_str.endswith((".csv", ".tsv")):
                self.store = MoleculeEnergyStore.from_csv(database_str)
            else:
                raise ValueError(
                    f"Unsupported reaction energy database format, only csv & tsv supported: {database_str}"
                )
        if dft_executor is not None:
            self.dft_executor = dft_executor
        else:
            # In the future, some DFT method will be the default, instantiate it here?
            raise ValueError("DFT executor is required for reaction energy checks.")

    def result_from_energies(self, record: ReactionEnergyRecord | None) -> ToolResult:
        """Create a ToolResult from a ReactionEnergyRecord."""
        if record is None:
            return ToolResult(
                name=self.name,
                status=ToolStatus.UNKNOWN,
                data=None,
                comment="no cached reaction energy was found and no dft fallback was run.",
            )

        passed = (
            record.energy_difference_ev_mol
            <= self.config.settings.reaction_energy_max_ev_mol
        )
        return ToolResult(
            name=self.name,
            status=ToolStatus.PASS if passed else ToolStatus.FAIL,
            data=ReactionEnergyResultDetails(
                energy_difference_ev_mol=record.energy_difference_ev_mol,
                source=record.source,
                metadata=record.metadata,
            ),
            comment=(
                "Reaction energy is within the allowed threshold."
                if passed
                else "Reaction energy exceeds the allowed threshold."
            ),
        )

    def check_cache(self, rxn_smi: str) -> None | ToolResult:
        if self.store is None:
            return None
        reactants: list[str]
        reactants, _, products = parse_reaction_smi(
            canonicalize_reaction_smiles(rxn_smi), ret_mol=False
        )

        energies = {}
        role_total_energies = {"reactants": 0, "products": 0}
        for role, smis in (("reactants", reactants), ("products", products)):
            for smi in smis:
                mol_energy = self.store.lookup(smi)
                if mol_energy is None:
                    return None
                energies[smi] = mol_energy.energy_ev
                role_total_energies[role] += mol_energy.energy_ev

        record = ReactionEnergyRecord(
            energy_difference_ev_mol=role_total_energies["products"]
            - role_total_energies["reactants"],
            source={s: "cache" for s in (reactants + products)},
            metadata={
                "reactants": {smi: energies[smi] for smi in reactants},
                "products": {smi: energies[smi] for smi in products},
            },
        )
        return self.result_from_energies(record)

    def run(self, rxn_smiles: str, context: ToolResultsDict) -> ToolResult:
        reactants, _, products = parse_reaction_smi(
            rxn_smiles, ret_mol=False
        )  # strings

        energies = {}
        role_total_energies = {"reactants": 0, "products": 0}
        sources = {}
        for role, smis in (("reactants", reactants), ("products", products)):
            for smi in smis:
                mol_energy = None
                if self.store:
                    mol_energy = self.store.lookup(smi)
                if not mol_energy:
                    mol_energy = self.dft_executor.run(smi)
                sources[smi] = mol_energy.source
                energies[smi] = mol_energy
                role_total_energies[role] += mol_energy.energy_ev
                if isinstance(self.store, MoleculeEnergyStore):
                    self.store.save(mol_energy)

        record = ReactionEnergyRecord(
            energy_difference_ev_mol=role_total_energies["products"]
            - role_total_energies["reactants"],
            source=sources,
            metadata={
                "reactants": {smi: energies[smi] for smi in reactants},
                "products": {smi: energies[smi] for smi in products},
            },
        )
        return self.result_from_energies(record)
