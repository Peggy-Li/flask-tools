###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

#!/usr/bin/env python3
"""
Subtractive common-subgraph atom mapper for reaction topology analysis.

This module implements the approach discussed in the conversation:

    product graph - common subgraph occurrences of reactant copies = residual

The subtraction unit is NOT necessarily a full reactant.  It is a connected
common subgraph occurrence between a reactant component and the product side.
The selector can be an ILP (via scipy.optimize.milp) or a greedy heuristic.

Typical use:

    python subtractive_reaction_mapper.py 'CC.CNC>>CCNCC'
    python subtractive_reaction_mapper.py '[C:1]C[C:2]>>[C:1]CC[C:2]'
    python subtractive_reaction_mapper.py '[C:1][C:2].[C:3][N:4][C:5]>>[C:1][C:3][N:4][C:5][C:2]'

Python API:

    from subtractive_reaction_mapper import subtractive_map_reaction
    result = subtractive_map_reaction('CC.CNC>>CCNCC')
    print(result.to_jsonable())

Dependencies:
    rdkit, networkx
Optional dependency:
    scipy, for ILP selection.  If scipy MILP is unavailable, selector='ilp'
    falls back to greedy selection unless fallback=False is passed.

Important modeling notes:
    * Reactant components are copied virtually up to max_copies.
    * Candidates are connected common subgraph occurrences.
    * Multiple candidates may be chosen from the same reactant copy, which is
      how true split lineages are represented and penalized.
    * Atom maps, when present on both sides, are treated as hard anchors by
      default.  This allows examples such as [C:1]C[C:2]>>[C:1]CC[C:2] to
      report a stretched/split lineage rather than remapping to a contiguous
      product subgraph.
    * After subtraction, connected residual fragments are reported on both the
      product and reactant sides.  Whole uncovered product components are
      flagged as byproduct/missing-source candidates.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdFMCS

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    import scipy.sparse as sp
    import numpy as np

    SCIPY_MILP_AVAILABLE = True
except Exception:  # pragma: no cover - import availability depends on env
    Bounds = None  # type: ignore
    LinearConstraint = None  # type: ignore
    milp = None  # type: ignore
    sp = None  # type: ignore
    np = None  # type: ignore
    SCIPY_MILP_AVAILABLE = False


INF = 10**12


@dataclass(frozen=True)
class ReactionAtomMapperConfig:
    """Configuration for candidate generation, selection, and diagnostics."""

    max_copies: int = 3
    min_fragment_atoms: int = 1
    max_fragment_atoms: int = 8
    max_fragments_per_reactant: int = 2500
    max_matches_per_fragment: int = 128
    max_base_candidates_per_reactant: int = 6000
    include_rdkit_mcs_candidates: bool = True
    max_mcs_matches: int = 256
    respect_atom_maps: Optional[bool] = None  # None means auto-detect.
    require_atom_map_match_when_present: bool = True
    mapped_reactants_single_copy: bool = True
    compare_formal_charge: bool = True
    compare_aromaticity: bool = False
    compare_isotope: bool = False
    compare_bond_order: bool = True
    allow_extra_product_edges_in_candidate: bool = True
    selector: str = "ilp"  # ilp or greedy
    fallback_to_greedy: bool = True

    # Linear objective terms.  These are deliberately simple; detailed topology
    # diagnostics are computed after selection.
    atom_reward: float = 10.0
    preserved_bond_reward: float = 5.0
    atom_map_anchor_bonus: float = 25.0
    candidate_piece_penalty: float = 2.0
    active_copy_penalty: float = 1.0
    unused_reactant_atom_penalty_active_copy: float = 6.0
    extra_product_edge_penalty: float = 2.0
    single_atom_piece_penalty: float = 4.0
    broken_bond_environment_penalty: float = 1.0
    bond_environment_objective: str = "off"  # off, integrated, or rerank
    bond_environment_rank_tolerance: float = 1.0e-6
    stable_single_bond_break_penalty: float = 1.0
    unsaturated_endpoint_break_credit: float = 0.75
    ring_bond_break_penalty: float = 2.0
    max_broken_bond_pair_penalty_terms: int = 25000

    # Diagnostic distances.
    max_segment_distance: int = 8

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        base_dir: Path,
    ) -> ReactionEnergyConfig:
        mapping = _validate_mapping_format(
            data, name="tools_settings.reaction_atom_mapper"
        )
        return cls(todo)  # todo


@dataclass(frozen=True)
class ReactantComponent:
    rid: int
    mol: Chem.Mol
    graph: nx.Graph
    smiles: str


@dataclass(frozen=True)
class Candidate:
    """A connected common-subgraph subtraction candidate before copy expansion."""

    cid: int
    reactant_id: int
    reactant_atoms: Tuple[int, ...]
    product_atoms: Tuple[int, ...]
    r_to_p: Tuple[Tuple[int, int], ...]
    preserved_bonds: int
    extra_product_edges: int
    atom_map_matches: int
    score: float
    source: str

    def mapping_dict(self) -> Dict[int, int]:
        return dict(self.r_to_p)

    def product_atom_set(self) -> FrozenSet[int]:
        return frozenset(self.product_atoms)

    def reactant_atom_set(self) -> FrozenSet[int]:
        return frozenset(self.reactant_atoms)


@dataclass(frozen=True)
class ExpandedCandidate:
    xid: int
    base: Candidate
    copy_id: int

    @property
    def reactant_id(self) -> int:
        return self.base.reactant_id

    @property
    def score(self) -> float:
        return self.base.score

    @property
    def product_atoms(self) -> Tuple[int, ...]:
        return self.base.product_atoms

    @property
    def reactant_atoms(self) -> Tuple[int, ...]:
        return self.base.reactant_atoms

    @property
    def r_to_p(self) -> Tuple[Tuple[int, int], ...]:
        return self.base.r_to_p


@dataclass
class SelectedPiece:
    # todo: add doc
    reactant_id: int
    copy_id: int
    candidate_id: int
    source: str
    reactant_atoms: Tuple[int, ...]
    product_atoms: Tuple[int, ...]
    r_to_p: Dict[int, int]
    preserved_bonds: int
    extra_product_edges: int
    score: float


@dataclass
class SubtractiveMappingResult:
    reaction_smiles: str
    selector: str  # todo what is this
    objective_value: float
    status: str
    reactant_components: List[ReactantComponent]
    product_mol: Chem.Mol
    product_graph: nx.Graph
    selected_pieces: List[SelectedPiece]
    diagnostics: Dict[str, Any]
    config: ReactionAtomMapperConfig

    def atom_mapped_reaction_smiles(self) -> str:
        return build_atom_mapped_reaction_smiles(self)

    def to_jsonable(self) -> Dict[str, Any]:
        reactants = [
            {
                "reactant_id": rc.rid,
                "smiles": rc.smiles,
                "atom_count": rc.mol.GetNumAtoms(),
                "bond_count": rc.mol.GetNumBonds(),
            }
            for rc in self.reactant_components
        ]
        pieces = [
            {
                "reactant_id": p.reactant_id,
                "copy_id": p.copy_id,
                "candidate_id": p.candidate_id,
                "source": p.source,
                "reactant_atoms": list(p.reactant_atoms),
                "product_atoms": list(p.product_atoms),
                "r_to_p": {str(k): v for k, v in sorted(p.r_to_p.items())},
                "preserved_bonds": p.preserved_bonds,
                "extra_product_edges": p.extra_product_edges,
                "score": p.score,
            }
            for p in self.selected_pieces
        ]
        return {
            "reaction_smiles": self.reaction_smiles,
            "atom_mapped_reaction_smiles": self.atom_mapped_reaction_smiles(),
            "selector": self.selector,
            "status": self.status,
            "objective_value": self.objective_value,
            "reactants": reactants,
            "product_smiles": (
                Chem.MolToSmiles(self.product_mol, canonical=True)
                if self.product_mol is not None
                else ""
            ),
            "selected_pieces": pieces,
            "diagnostics": self.diagnostics,
            "config": dataclasses.asdict(self.config),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_jsonable(), indent=indent, sort_keys=True)


def _copy_mol_with_fresh_atom_maps(
    mol: Chem.Mol, atom_to_map_num: Mapping[int, int]
) -> Chem.Mol:
    """Return a copy of mol with only atom_to_map_num atom maps set.

    Existing atom-map numbers are cleared first so copied reactants get unique
    map numbers, which is important when a reactant is reused via virtual copies.
    """
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(0)
    for atom_idx, map_num in atom_to_map_num.items():
        if 0 <= int(atom_idx) < out.GetNumAtoms():
            out.GetAtomWithIdx(int(atom_idx)).SetAtomMapNum(int(map_num))
    return out


def _next_available_atom_map(used: Set[int]) -> int:
    value = 1
    while value in used:
        value += 1
    return value


def build_atom_mapped_reaction_smiles(result: "SubtractiveMappingResult") -> str:
    """Construct a partially atom-mapped reaction SMILES from selected pieces.

    The left side contains one molecule for each active virtual reactant copy;
    unused original reactant components are included once without atom maps.
    Product atoms not covered by selected subtraction pieces remain unmapped.

    Existing atom-map numbers are preserved when they are unambiguous. This
    keeps anchored inputs such as [C:1]C[C:2]>>[C:1]CC[C:2] readable, while
    still assigning fresh unique map numbers for unmapped or copied atoms.
    """
    pairs: List[Tuple[int, int, int, int]] = []
    for piece in result.selected_pieces:
        for reactant_atom, product_atom in piece.r_to_p.items():
            pairs.append(
                (piece.reactant_id, piece.copy_id, reactant_atom, product_atom)
            )
    pairs = sorted(set(pairs), key=lambda x: (x[0], x[1], x[2], x[3]))

    # Prefer existing map numbers only when that preference is unique across
    # all selected pairs. This avoids duplicate atom-map labels when a mapped
    # reactant is virtually copied.
    preferred_by_pair: Dict[Tuple[int, int, int, int], int] = {}
    preferred_counts: Counter[int] = Counter()
    for rid, copy_id, reactant_atom, product_atom in pairs:
        rc = result.reactant_components[rid]
        rm = int(rc.mol.GetAtomWithIdx(reactant_atom).GetAtomMapNum())
        pm = int(result.product_mol.GetAtomWithIdx(product_atom).GetAtomMapNum())
        preferred = 0
        if rm > 0 and pm > 0 and rm == pm:
            preferred = rm
        elif rm > 0 and pm == 0:
            preferred = rm
        elif pm > 0 and rm == 0:
            preferred = pm
        if preferred > 0:
            preferred_by_pair[(rid, copy_id, reactant_atom, product_atom)] = preferred
            preferred_counts[preferred] += 1

    rcopy_atom_to_map: Dict[Tuple[int, int, int], int] = {}
    product_atom_to_map: Dict[int, int] = {}
    used_maps: Set[int] = set()

    def assign_pair(
        rid: int, copy_id: int, reactant_atom: int, product_atom: int, map_num: int
    ) -> None:
        rcopy_atom_to_map[(rid, copy_id, reactant_atom)] = map_num
        product_atom_to_map[product_atom] = map_num
        used_maps.add(map_num)

    # First assign unambiguous existing atom maps.
    for rid, copy_id, reactant_atom, product_atom in pairs:
        pair = (rid, copy_id, reactant_atom, product_atom)
        preferred = preferred_by_pair.get(pair, 0)
        if preferred > 0 and preferred_counts[preferred] == 1:
            assign_pair(rid, copy_id, reactant_atom, product_atom, preferred)

    # Then assign fresh map numbers for everything else. Sort by product atom so
    # the generated labels are stable and easy to inspect on the product side.
    for rid, copy_id, reactant_atom, product_atom in sorted(
        pairs, key=lambda x: (x[3], x[0], x[1], x[2])
    ):
        rkey = (rid, copy_id, reactant_atom)
        if rkey in rcopy_atom_to_map and product_atom in product_atom_to_map:
            continue
        if rkey in rcopy_atom_to_map:
            map_num = rcopy_atom_to_map[rkey]
        elif product_atom in product_atom_to_map:
            map_num = product_atom_to_map[product_atom]
        else:
            map_num = _next_available_atom_map(used_maps)
        assign_pair(rid, copy_id, reactant_atom, product_atom, map_num)

    active_copies_by_reactant: Dict[int, Set[int]] = defaultdict(set)
    for rid, copy_id, _atom in rcopy_atom_to_map:
        active_copies_by_reactant[rid].add(copy_id)

    reactant_smiles_parts: List[str] = []
    for rc in sorted(result.reactant_components, key=lambda x: x.rid):
        copy_ids = sorted(active_copies_by_reactant.get(rc.rid, set()))
        if not copy_ids:
            copy_ids = [0]
        for copy_id in copy_ids:
            atom_maps = {
                atom_idx: map_num
                for (rid, k, atom_idx), map_num in rcopy_atom_to_map.items()
                if rid == rc.rid and k == copy_id
            }
            reactant_mol = _copy_mol_with_fresh_atom_maps(rc.mol, atom_maps)
            reactant_smiles_parts.append(Chem.MolToSmiles(reactant_mol, canonical=True))

    product_mol = _copy_mol_with_fresh_atom_maps(
        result.product_mol, product_atom_to_map
    )
    product_smiles = Chem.MolToSmiles(product_mol, canonical=True)
    return f"{'.'.join(reactant_smiles_parts)}>>{product_smiles}"


# ---------------------------------------------------------------------------
# RDKit / graph helpers
# ---------------------------------------------------------------------------


def parse_reaction_smiles_to_halves(reaction_smiles: str) -> Tuple[str, str]:
    """Return reactant_side, product_side for SMILES or reaction SMILES."""
    if ">>" in reaction_smiles:
        left, right = reaction_smiles.split(">>", 1)
        return left.strip(), right.strip()
    parts = reaction_smiles.split(">")
    if len(parts) == 3:
        return parts[0].strip(), parts[2].strip()
    raise ValueError(
        "Expected reaction SMILES containing '>>' or 'reactants>agents>products'."
    )


def mol_from_side(side: str) -> Chem.Mol:
    if side == "":
        return Chem.Mol()
    mol = Chem.MolFromSmiles(side, sanitize=True)
    if mol is None:
        raise ValueError(f"Could not parse SMILES side: {side!r}")
    return mol


def split_reactant_components(reactant_side: str) -> List[ReactantComponent]:
    mol = mol_from_side(reactant_side)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    comps: List[ReactantComponent] = []
    for rid, frag in enumerate(frags):
        # Remove atom-map numbers from canonical component SMILES?  Keep them,
        # since they are useful when inspecting anchored examples.
        smi = Chem.MolToSmiles(frag, canonical=True)
        comps.append(
            ReactantComponent(rid=rid, mol=frag, graph=mol_to_nx(frag), smiles=smi)
        )
    return comps


def atom_attrs(atom: Chem.Atom) -> Dict[str, Any]:
    return {
        "atomic_num": atom.GetAtomicNum(),
        "symbol": atom.GetSymbol(),
        "formal_charge": atom.GetFormalCharge(),
        "isotope": atom.GetIsotope(),
        "is_aromatic": atom.GetIsAromatic(),
        "atom_map": atom.GetAtomMapNum(),
    }


def bond_order_value(bond: Chem.Bond) -> float:
    # RDKit BondTypeAsDouble handles aromatic as 1.5.
    try:
        return float(bond.GetBondTypeAsDouble())
    except Exception:
        return float(bond.GetBondType())


def bond_attrs(bond: Chem.Bond) -> Dict[str, Any]:
    return {
        "bond_order": bond_order_value(bond),
        "bond_type": str(bond.GetBondType()),
        "is_aromatic": bond.GetIsAromatic(),
    }


def mol_to_nx(mol: Chem.Mol) -> nx.Graph:
    g = nx.Graph()
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx(), **atom_attrs(atom))
    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **bond_attrs(bond))
    return g


def side_has_any_atom_maps(mol: Chem.Mol) -> bool:
    return any(atom.GetAtomMapNum() > 0 for atom in mol.GetAtoms())


def auto_respect_atom_maps(
    reactants: Sequence[ReactantComponent],
    product_mol: Chem.Mol,
    cfg: ReactionAtomMapperConfig,
) -> bool:
    if cfg.respect_atom_maps is not None:
        return bool(cfg.respect_atom_maps)
    left = any(side_has_any_atom_maps(rc.mol) for rc in reactants)
    right = side_has_any_atom_maps(product_mol)
    return left and right


def atom_compatible_attrs(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    cfg: ReactionAtomMapperConfig,
    respect_maps: bool,
) -> bool:
    if a["atomic_num"] != b["atomic_num"]:
        return False
    if cfg.compare_formal_charge and a.get("formal_charge") != b.get("formal_charge"):
        return False
    if cfg.compare_aromaticity and a.get("is_aromatic") != b.get("is_aromatic"):
        return False
    if cfg.compare_isotope and a.get("isotope") != b.get("isotope"):
        return False
    if respect_maps and cfg.require_atom_map_match_when_present:
        ma = int(a.get("atom_map") or 0)
        mb = int(b.get("atom_map") or 0)
        # Hard-anchor semantics: if either endpoint is mapped, both must have the
        # same nonzero map number.  Unmapped atoms can only match unmapped atoms.
        if ma != mb:
            return False
    return True


def bond_compatible_attrs(
    a: Mapping[str, Any], b: Mapping[str, Any], cfg: ReactionAtomMapperConfig
) -> bool:
    if (
        cfg.compare_bond_order
        and abs(float(a.get("bond_order", 0.0)) - float(b.get("bond_order", 0.0)))
        > 1.0e-6
    ):
        return False
    return True


def atom_compatible_mol(
    ra: Chem.Atom, pa: Chem.Atom, cfg: ReactionAtomMapperConfig, respect_maps: bool
) -> bool:
    return atom_compatible_attrs(atom_attrs(ra), atom_attrs(pa), cfg, respect_maps)


def bond_compatible_mol(
    rb: Chem.Bond, pb: Chem.Bond, cfg: ReactionAtomMapperConfig
) -> bool:
    return bond_compatible_attrs(bond_attrs(rb), bond_attrs(pb), cfg)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def connected_subsets_limited(
    g: nx.Graph, min_size: int, max_size: int, max_subsets: int
) -> List[Tuple[int, ...]]:
    """Generate connected node subsets up to max_size, largest first.

    This is intentionally bounded.  Exhaustive connected-subgraph enumeration is
    exponential, so this helper produces a diverse bounded set suitable for
    common-subgraph candidate generation.
    """
    if g.number_of_nodes() == 0:
        return []
    max_size = min(max_size, g.number_of_nodes())
    min_size = max(1, min_size)

    seen: Set[FrozenSet[int]] = set()
    q: deque[FrozenSet[int]] = deque()
    for n in sorted(g.nodes):
        fs = frozenset([n])
        seen.add(fs)
        q.append(fs)

    out: List[Tuple[int, ...]] = []
    while q and len(seen) <= max_subsets * 8:
        cur = q.popleft()
        if len(cur) >= min_size:
            out.append(tuple(sorted(cur)))
            if len(out) >= max_subsets:
                break
        if len(cur) >= max_size:
            continue
        boundary: Set[int] = set()
        for u in cur:
            boundary.update(g.neighbors(u))
        for v in sorted(boundary - set(cur)):
            nxt = frozenset(set(cur) | {v})
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    out.sort(key=lambda xs: (-len(xs), xs))
    return out[:max_subsets]


def count_internal_edges(g: nx.Graph, nodes: Iterable[int]) -> int:
    s = set(nodes)
    return sum(1 for u, v in g.edges if u in s and v in s)


def validate_mapping_edges(
    r_graph: nx.Graph,
    p_graph: nx.Graph,
    r_to_p: Mapping[int, int],
    cfg: ReactionAtomMapperConfig,
) -> Tuple[bool, int, int]:
    """Check reactant edges are present/compatible in product.

    Returns (valid, preserved_bonds, extra_product_edges_among_selected_atoms).
    If allow_extra_product_edges_in_candidate is False, extra product edges make
    the candidate invalid.  Otherwise they are allowed and penalized.
    """
    preserved = 0
    for ru, rv, rdata in r_graph.edges(data=True):
        if ru not in r_to_p or rv not in r_to_p:
            continue
        pu, pv = r_to_p[ru], r_to_p[rv]
        if not p_graph.has_edge(pu, pv):
            return False, 0, 0
        if not bond_compatible_attrs(rdata, p_graph.edges[pu, pv], cfg):
            return False, 0, 0
        preserved += 1

    inv = {p: r for r, p in r_to_p.items()}
    extra = 0
    selected_p = set(inv)
    for pu, pv in p_graph.subgraph(selected_p).edges:
        ru, rv = inv[pu], inv[pv]
        if not r_graph.has_edge(ru, rv):
            extra += 1
    if extra and not cfg.allow_extra_product_edges_in_candidate:
        return False, 0, 0
    return True, preserved, extra


def candidate_score(
    r_graph: nx.Graph,
    p_graph: nx.Graph,
    r_to_p: Mapping[int, int],
    preserved_bonds: int,
    extra_product_edges: int,
    cfg: ReactionAtomMapperConfig,
) -> Tuple[float, int]:
    atom_count = len(r_to_p)
    anchor_matches = 0
    for r, p in r_to_p.items():
        rm = int(r_graph.nodes[r].get("atom_map") or 0)
        pm = int(p_graph.nodes[p].get("atom_map") or 0)
        if rm > 0 and rm == pm:
            anchor_matches += 1
    score = (
        cfg.atom_reward * atom_count
        + cfg.preserved_bond_reward * preserved_bonds
        + cfg.atom_map_anchor_bonus * anchor_matches
        - cfg.extra_product_edge_penalty * extra_product_edges
    )
    if atom_count == 1:
        score -= cfg.single_atom_piece_penalty
    return score, anchor_matches


def mapping_key(reactant_id: int, r_to_p: Mapping[int, int]) -> Tuple[Any, ...]:
    return (reactant_id, tuple(sorted(r_to_p.items())))


def add_candidate_if_valid(
    candidates: Dict[Tuple[Any, ...], Candidate],
    reactant_id: int,
    r_graph: nx.Graph,
    p_graph: nx.Graph,
    r_to_p: Mapping[int, int],
    cfg: ReactionAtomMapperConfig,
    source: str,
    next_id: List[int],
) -> None:
    if not r_to_p:
        return
    if len(set(r_to_p.values())) != len(r_to_p):
        return
    # Candidate must be connected on the reactant side and on the product side.
    # This prevents one candidate from hiding a connectivity break.  A split
    # lineage must be represented as multiple selected pieces.
    r_nodes = tuple(sorted(r_to_p))
    p_nodes = tuple(sorted(r_to_p.values()))
    if len(r_nodes) > 1:
        if not nx.is_connected(r_graph.subgraph(r_nodes)):
            return
        if not nx.is_connected(p_graph.subgraph(p_nodes)):
            return
    valid, preserved, extra = validate_mapping_edges(r_graph, p_graph, r_to_p, cfg)
    if not valid:
        return
    score, anchors = candidate_score(r_graph, p_graph, r_to_p, preserved, extra, cfg)
    key = mapping_key(reactant_id, r_to_p)
    existing = candidates.get(key)
    if existing is not None and existing.score >= score:
        return
    cid = next_id[0]
    next_id[0] += 1
    candidates[key] = Candidate(
        cid=cid,
        reactant_id=reactant_id,
        reactant_atoms=tuple(sorted(r_to_p)),
        product_atoms=tuple(sorted(r_to_p.values())),
        r_to_p=tuple(sorted(r_to_p.items())),
        preserved_bonds=preserved,
        extra_product_edges=extra,
        atom_map_matches=anchors,
        score=score,
        source=source,
    )


def generate_fragment_candidates_for_reactant(
    rc: ReactantComponent,
    product_graph: nx.Graph,
    cfg: ReactionAtomMapperConfig,
    respect_maps: bool,
    next_id: List[int],
) -> List[Candidate]:
    """Generate connected reactant-fragment candidates via NetworkX matching."""
    r_graph = rc.graph
    candidates: Dict[Tuple[Any, ...], Candidate] = {}
    fragments = connected_subsets_limited(
        r_graph,
        min_size=cfg.min_fragment_atoms,
        max_size=cfg.max_fragment_atoms,
        max_subsets=cfg.max_fragments_per_reactant,
    )

    def nm(p_attrs: Mapping[str, Any], r_attrs: Mapping[str, Any]) -> bool:
        return atom_compatible_attrs(r_attrs, p_attrs, cfg, respect_maps)

    def em(p_attrs: Mapping[str, Any], r_attrs: Mapping[str, Any]) -> bool:
        return bond_compatible_attrs(r_attrs, p_attrs, cfg)

    for frag_nodes in fragments:
        r_sub = r_graph.subgraph(frag_nodes).copy()
        gm = nx.algorithms.isomorphism.GraphMatcher(
            product_graph, r_sub, node_match=nm, edge_match=em
        )
        if cfg.allow_extra_product_edges_in_candidate and hasattr(
            gm, "subgraph_monomorphisms_iter"
        ):
            iterator = gm.subgraph_monomorphisms_iter()
        else:
            iterator = gm.subgraph_isomorphisms_iter()

        n_matches = 0
        for p_to_r in iterator:
            # p_to_r maps product node -> reactant node.  Invert.
            r_to_p = {r: p for p, r in p_to_r.items()}
            # GraphMatcher can return mappings larger than the query for some
            # monomorphism variants; filter to the fragment atom set.
            r_to_p = {r: p for r, p in r_to_p.items() if r in r_sub.nodes}
            if set(r_to_p) != set(r_sub.nodes):
                continue
            add_candidate_if_valid(
                candidates,
                rc.rid,
                r_graph,
                product_graph,
                r_to_p,
                cfg,
                "nx_fragment",
                next_id,
            )
            n_matches += 1
            if n_matches >= cfg.max_matches_per_fragment:
                break
        if len(candidates) >= cfg.max_base_candidates_per_reactant:
            break

    vals = list(candidates.values())
    vals.sort(
        key=lambda c: (
            -c.score,
            -len(c.reactant_atoms),
            c.reactant_atoms,
            c.product_atoms,
        )
    )
    return vals[: cfg.max_base_candidates_per_reactant]


def generate_rdkit_mcs_candidates_for_reactant(
    rc: ReactantComponent,
    product_mol: Chem.Mol,
    product_graph: nx.Graph,
    cfg: ReactionAtomMapperConfig,
    respect_maps: bool,
    next_id: List[int],
) -> List[Candidate]:
    """Generate large candidates from RDKit FindMCS."""
    if rc.mol.GetNumAtoms() == 0 or product_mol.GetNumAtoms() == 0:
        return []
    candidates: Dict[Tuple[Any, ...], Candidate] = {}
    try:
        params = rdFMCS.MCSParameters()
        params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        params.BondTyper = (
            rdFMCS.BondCompare.CompareOrder
            if cfg.compare_bond_order
            else rdFMCS.BondCompare.CompareAny
        )
        params.RingMatchesRingOnly = True
        params.CompleteRingsOnly = False
        params.Timeout = 5
        mcs = rdFMCS.FindMCS([rc.mol, product_mol], params)
    except Exception:
        return []
    if mcs.canceled or not mcs.smartsString:
        return []
    query = Chem.MolFromSmarts(mcs.smartsString)
    if query is None or query.GetNumAtoms() == 0:
        return []
    try:
        r_matches = list(
            rc.mol.GetSubstructMatches(
                query, uniquify=True, maxMatches=cfg.max_mcs_matches
            )
        )
        p_matches = list(
            product_mol.GetSubstructMatches(
                query, uniquify=True, maxMatches=cfg.max_mcs_matches
            )
        )
    except TypeError:
        # Older RDKit versions may not accept maxMatches as keyword.
        r_matches = list(rc.mol.GetSubstructMatches(query, True))[: cfg.max_mcs_matches]
        p_matches = list(product_mol.GetSubstructMatches(query, True))[
            : cfg.max_mcs_matches
        ]

    for r_match in r_matches[: cfg.max_mcs_matches]:
        for p_match in p_matches[: cfg.max_mcs_matches]:
            r_to_p = {int(r): int(p) for r, p in zip(r_match, p_match)}
            ok = True
            for r, p in r_to_p.items():
                if not atom_compatible_mol(
                    rc.mol.GetAtomWithIdx(r),
                    product_mol.GetAtomWithIdx(p),
                    cfg,
                    respect_maps,
                ):
                    ok = False
                    break
            if not ok:
                continue
            add_candidate_if_valid(
                candidates,
                rc.rid,
                rc.graph,
                product_graph,
                r_to_p,
                cfg,
                "rdkit_mcs",
                next_id,
            )
            if len(candidates) >= cfg.max_base_candidates_per_reactant:
                break
        if len(candidates) >= cfg.max_base_candidates_per_reactant:
            break
    vals = list(candidates.values())
    vals.sort(
        key=lambda c: (
            -c.score,
            -len(c.reactant_atoms),
            c.reactant_atoms,
            c.product_atoms,
        )
    )
    return vals[: cfg.max_base_candidates_per_reactant]


def generate_base_candidates(
    reactants: Sequence[ReactantComponent],
    product_mol: Chem.Mol,
    product_graph: nx.Graph,
    cfg: ReactionAtomMapperConfig,
    respect_maps: bool,
) -> List[Candidate]:
    """todo: doc. what does base candidates mean?
    * Candidates are connected common subgraph occurrences.
    """
    next_id = [0]
    all_candidates: Dict[Tuple[Any, ...], Candidate] = {}
    for rc in reactants:
        per_reactant: List[Candidate] = []
        if cfg.include_rdkit_mcs_candidates:
            per_reactant.extend(
                generate_rdkit_mcs_candidates_for_reactant(
                    rc, product_mol, product_graph, cfg, respect_maps, next_id
                )
            )
        per_reactant.extend(
            generate_fragment_candidates_for_reactant(
                rc, product_graph, cfg, respect_maps, next_id
            )
        )
        # Deduplicate across MCS and fragment generation.
        local: Dict[Tuple[Any, ...], Candidate] = {}
        for c in per_reactant:
            key = mapping_key(c.reactant_id, c.mapping_dict())
            if key not in local or c.score > local[key].score:
                local[key] = c
        vals = list(local.values())
        vals.sort(
            key=lambda c: (
                -c.score,
                -len(c.reactant_atoms),
                c.reactant_atoms,
                c.product_atoms,
            )
        )
        vals = vals[: cfg.max_base_candidates_per_reactant]
        for c in vals:
            key = mapping_key(c.reactant_id, c.mapping_dict())
            all_candidates[key] = c
    # Reassign candidate IDs densely for readability.
    vals = list(all_candidates.values())
    vals.sort(
        key=lambda c: (
            c.reactant_id,
            -c.score,
            -len(c.reactant_atoms),
            c.reactant_atoms,
            c.product_atoms,
        )
    )
    dense: List[Candidate] = []
    for cid, c in enumerate(vals):
        dense.append(dataclasses.replace(c, cid=cid))
    return dense


# ---------------------------------------------------------------------------
# Candidate selection: ILP and greedy
# ---------------------------------------------------------------------------


def reactant_has_mapped_atoms(rc: ReactantComponent) -> bool:
    return any(int(a.GetAtomMapNum()) > 0 for a in rc.mol.GetAtoms())


def copy_count_for_reactant(
    rc: ReactantComponent, cfg: ReactionAtomMapperConfig
) -> int:
    if cfg.mapped_reactants_single_copy and reactant_has_mapped_atoms(rc):
        return 1
    return cfg.max_copies


def expand_candidates(
    base_candidates: Sequence[Candidate],
    cfg: ReactionAtomMapperConfig,
    reactants: Optional[Sequence[ReactantComponent]] = None,
) -> List[ExpandedCandidate]:
    expanded: List[ExpandedCandidate] = []
    xid = 0
    copy_counts: Dict[int, int] = defaultdict(lambda: cfg.max_copies)
    if reactants is not None:
        copy_counts = defaultdict(
            lambda: cfg.max_copies,
            {rc.rid: copy_count_for_reactant(rc, cfg) for rc in reactants},
        )
    for c in base_candidates:
        for k in range(copy_counts[c.reactant_id]):
            expanded.append(ExpandedCandidate(xid=xid, base=c, copy_id=k))
            xid += 1
    return expanded


def select_candidates_greedy(
    base_candidates: Sequence[Candidate],
    reactants: Sequence[ReactantComponent],
    cfg: ReactionAtomMapperConfig,
) -> Tuple[List[ExpandedCandidate], float, str]:
    expanded = expand_candidates(base_candidates, cfg, reactants)
    expanded.sort(
        key=lambda x: (
            -(
                x.score
                - cfg.candidate_piece_penalty
                + cfg.unused_reactant_atom_penalty_active_copy * len(x.reactant_atoms)
            ),
            -len(x.product_atoms),
            x.reactant_id,
            x.copy_id,
        )
    )
    used_product: Set[int] = set()
    used_reactant_by_copy: Set[Tuple[int, int, int]] = set()
    active_copies: Set[Tuple[int, int]] = set()
    chosen: List[ExpandedCandidate] = []
    objective = 0.0
    for x in expanded:
        marginal = (
            x.score
            - cfg.candidate_piece_penalty
            + cfg.unused_reactant_atom_penalty_active_copy * len(x.reactant_atoms)
        )
        if (x.reactant_id, x.copy_id) not in active_copies:
            rc_atoms = reactants[x.reactant_id].graph.number_of_nodes()
            marginal -= (
                cfg.active_copy_penalty
                + cfg.unused_reactant_atom_penalty_active_copy * rc_atoms
            )
        if marginal <= 0:
            continue
        if any(p in used_product for p in x.product_atoms):
            continue
        if any(
            (x.reactant_id, x.copy_id, r) in used_reactant_by_copy
            for r in x.reactant_atoms
        ):
            continue
        chosen.append(x)
        objective += marginal
        used_product.update(x.product_atoms)
        for r in x.reactant_atoms:
            used_reactant_by_copy.add((x.reactant_id, x.copy_id, r))
        active_copies.add((x.reactant_id, x.copy_id))
    return chosen, objective, "greedy"


def atom_has_multiple_bond_to_hetero(atom: Chem.Atom) -> bool:
    """Generic local electronic environment test used for bond-break scoring."""
    for bond in atom.GetBonds():
        if bond_order_value(bond) < 1.5:
            continue
        other = bond.GetOtherAtom(atom)
        if other.GetAtomicNum() not in {1, 6}:
            return True
    return False


def atom_is_saturated_carbon(atom: Chem.Atom) -> bool:
    if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
        return False
    return all(bond_order_value(bond) <= 1.1 for bond in atom.GetBonds())


def bond_environment_break_penalty(
    mol: Chem.Mol, begin_atom: int, end_atom: int, cfg: ReactionAtomMapperConfig
) -> float:
    """Return a local-environment penalty for breaking a reactant bond.

    This intentionally uses generic atom/bond features rather than named
    functional groups.  Single bonds from saturated carbon to hetero atoms are
    treated as harder to break, while bonds attached to an atom with a multiple
    bond to a hetero atom are treated as more plausible reaction-center breaks.
    """
    scale = max(0.0, float(cfg.broken_bond_environment_penalty))
    if scale == 0.0:
        return 0.0

    bond = mol.GetBondBetweenAtoms(int(begin_atom), int(end_atom))
    if bond is None:
        return 0.0

    a1 = mol.GetAtomWithIdx(int(begin_atom))
    a2 = mol.GetAtomWithIdx(int(end_atom))
    order = bond_order_value(bond)
    penalty = scale

    if order > 1.1:
        penalty += scale * (order - 1.0)
    if bond.IsInRing():
        penalty += cfg.ring_bond_break_penalty
    if a1.GetIsAromatic() or a2.GetIsAromatic():
        penalty += 0.5 * scale

    has_unsaturated_endpoint = atom_has_multiple_bond_to_hetero(
        a1
    ) or atom_has_multiple_bond_to_hetero(a2)
    if has_unsaturated_endpoint:
        penalty -= cfg.unsaturated_endpoint_break_credit

    atomic_nums = {a1.GetAtomicNum(), a2.GetAtomicNum()}
    has_hetero = any(z not in {1, 6} for z in atomic_nums)
    has_saturated_carbon = atom_is_saturated_carbon(a1) or atom_is_saturated_carbon(a2)
    if (
        order <= 1.1
        and has_hetero
        and has_saturated_carbon
        and not has_unsaturated_endpoint
    ):
        penalty += cfg.stable_single_bond_break_penalty

    return max(0.05 * scale, penalty)


def broken_reactant_bond_pair_penalties(
    expanded: Sequence[ExpandedCandidate],
    reactants: Sequence[ReactantComponent],
    product_graph: nx.Graph,
    cfg: ReactionAtomMapperConfig,
) -> List[Tuple[int, int, float]]:
    """Build pairwise penalties for selected pieces that break reactant bonds."""
    if cfg.broken_bond_environment_penalty <= 0.0:
        return []

    by_reactant_copy_atom: Dict[Tuple[int, int, int], List[Tuple[int, int]]] = (
        defaultdict(list)
    )
    for i, x in enumerate(expanded):
        r_to_p = dict(x.r_to_p)
        for r in x.reactant_atoms:
            by_reactant_copy_atom[(x.reactant_id, x.copy_id, r)].append((i, r_to_p[r]))

    penalties: Dict[Tuple[int, int], float] = defaultdict(float)
    reactant_sets = [set(x.reactant_atoms) for x in expanded]
    product_sets = [set(x.product_atoms) for x in expanded]
    for rc in reactants:
        n_copies = copy_count_for_reactant(rc, cfg)
        for ru, rv, rdata in rc.graph.edges(data=True):
            bond_penalty = bond_environment_break_penalty(rc.mol, ru, rv, cfg)
            if bond_penalty <= 0.0:
                continue
            for copy_id in range(n_copies):
                left = by_reactant_copy_atom.get((rc.rid, copy_id, ru), [])
                right = by_reactant_copy_atom.get((rc.rid, copy_id, rv), [])
                for i, pu in left:
                    for j, pv in right:
                        if i == j:
                            continue
                        # These pairs are already mutually exclusive by atom
                        # coverage constraints, so a pairwise break variable
                        # would only enlarge the ILP without changing feasible
                        # solutions or the objective.
                        if reactant_sets[i] & reactant_sets[j]:
                            continue
                        if product_sets[i] & product_sets[j]:
                            continue
                        if product_graph.has_edge(pu, pv) and bond_compatible_attrs(
                            rdata, product_graph.edges[pu, pv], cfg
                        ):
                            continue
                        penalties[tuple(sorted((i, j)))] += bond_penalty

    items = list(penalties.items())
    if (
        cfg.max_broken_bond_pair_penalty_terms > 0
        and len(items) > cfg.max_broken_bond_pair_penalty_terms
    ):
        items = sorted(items, key=lambda kv: (-kv[1], kv[0]))[
            : cfg.max_broken_bond_pair_penalty_terms
        ]
    return [(i, j, penalty) for (i, j), penalty in sorted(items)]


def select_candidates_ilp(
    base_candidates: Sequence[Candidate],
    reactants: Sequence[ReactantComponent],
    product_graph: nx.Graph,
    cfg: ReactionAtomMapperConfig,
) -> Tuple[List[ExpandedCandidate], float, str]:
    if not SCIPY_MILP_AVAILABLE:
        if cfg.fallback_to_greedy:
            return select_candidates_greedy(base_candidates, reactants, cfg)
        raise RuntimeError("scipy.optimize.milp is not available.")
    if (
        np is None
        or sp is None
        or milp is None
        or LinearConstraint is None
        or Bounds is None
    ):
        raise RuntimeError("scipy.optimize.milp is not available.")

    mode = cfg.bond_environment_objective.lower().strip()
    if mode not in {"off", "integrated", "rerank"}:
        raise ValueError(
            "bond_environment_objective must be 'off', 'integrated', or 'rerank'."
        )

    expanded = expand_candidates(base_candidates, cfg, reactants)
    n_x = len(expanded)
    copy_keys: List[Tuple[int, int]] = []
    for rc in reactants:
        for k in range(cfg.max_copies):
            copy_keys.append((rc.rid, k))
    copy_index = {ck: i for i, ck in enumerate(copy_keys)}
    n_y = len(copy_keys)
    if n_x == 0:
        return [], 0.0, "ilp_no_candidates"

    primary_c_base = np.zeros(n_x + n_y, dtype=float)
    for i, x in enumerate(expanded):
        # scipy minimizes, so negate the maximization coefficient.
        coeff = (
            x.score
            - cfg.candidate_piece_penalty
            + cfg.unused_reactant_atom_penalty_active_copy * len(x.reactant_atoms)
        )
        primary_c_base[i] = -coeff
    for ck, yi in copy_index.items():
        rid, _copy_id = ck
        primary_c_base[n_x + yi] = (
            cfg.active_copy_penalty
            + cfg.unused_reactant_atom_penalty_active_copy
            * reactants[rid].graph.number_of_nodes()
        )

    def solve(
        include_bond_environment: bool,
        primary_floor: Optional[float] = None,
        secondary_only: bool = False,
    ) -> Any:
        broken_pair_penalties = (
            broken_reactant_bond_pair_penalties(expanded, reactants, product_graph, cfg)
            if include_bond_environment
            else []
        )
        n_z = len(broken_pair_penalties)
        z_start = n_x + n_y
        n_vars = n_x + n_y + n_z

        primary_c = np.zeros(n_vars, dtype=float)
        primary_c[: n_x + n_y] = primary_c_base
        c = np.zeros(n_vars, dtype=float) if secondary_only else primary_c.copy()
        if include_bond_environment:
            for zi, (_i, _j, penalty) in enumerate(broken_pair_penalties):
                c[z_start + zi] = penalty

        constraint_rows: List[int] = []
        constraint_cols: List[int] = []
        constraint_data: List[float] = []
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []

        def add_sparse_constraint(
            coeffs: Mapping[int, float], lower: float, upper: float
        ) -> None:
            row_idx = len(lower_bounds)
            for col_idx, value in coeffs.items():
                if value != 0.0:
                    constraint_rows.append(row_idx)
                    constraint_cols.append(int(col_idx))
                    constraint_data.append(float(value))
            lower_bounds.append(float(lower))
            upper_bounds.append(float(upper))

        # Product atom covered at most once.
        for p in product_graph.nodes:
            coeffs: Dict[int, float] = {}
            for i, x in enumerate(expanded):
                if p in x.product_atoms:
                    coeffs[i] = 1.0
            if coeffs:
                add_sparse_constraint(coeffs, -np.inf, 1.0)

        # Reactant atom per copy used at most once.
        for rc in reactants:
            for k in range(cfg.max_copies):
                for r in rc.graph.nodes:
                    coeffs = {}
                    for i, x in enumerate(expanded):
                        if (
                            x.reactant_id == rc.rid
                            and x.copy_id == k
                            and r in x.reactant_atoms
                        ):
                            coeffs[i] = 1.0
                    if coeffs:
                        add_sparse_constraint(coeffs, -np.inf, 1.0)

        # x_j <= y_{reactant,copy}
        for i, x in enumerate(expanded):
            add_sparse_constraint(
                {i: 1.0, n_x + copy_index[(x.reactant_id, x.copy_id)]: -1.0},
                -np.inf,
                0.0,
            )

        # y_{reactant,copy} <= sum selected pieces using that copy.
        for ck, yi in copy_index.items():
            coeffs = {n_x + yi: 1.0}
            any_piece = False
            for i, x in enumerate(expanded):
                if (x.reactant_id, x.copy_id) == ck:
                    coeffs[i] = coeffs.get(i, 0.0) - 1.0
                    any_piece = True
            if any_piece:
                add_sparse_constraint(coeffs, -np.inf, 0.0)
            else:
                add_sparse_constraint(coeffs, 0.0, 0.0)

        if include_bond_environment:
            # z_ij is forced on when both selected pieces are present and their
            # mapped endpoints imply a broken reactant bond.
            for zi, (i, j, _penalty) in enumerate(broken_pair_penalties):
                add_sparse_constraint(
                    {i: 1.0, j: 1.0, z_start + zi: -1.0}, -np.inf, 1.0
                )

        # Symmetry breaking: y_{i,k+1} <= y_{i,k}
        for rc in reactants:
            for k in range(cfg.max_copies - 1):
                add_sparse_constraint(
                    {
                        n_x + copy_index[(rc.rid, k + 1)]: 1.0,
                        n_x + copy_index[(rc.rid, k)]: -1.0,
                    },
                    -np.inf,
                    0.0,
                )

        if primary_floor is not None:
            coeffs = {i: float(v) for i, v in enumerate(primary_c) if v != 0.0}
            add_sparse_constraint(coeffs, -np.inf, -float(primary_floor))

        constraints: List[LinearConstraint] = []
        if lower_bounds:
            a = sp.coo_matrix(
                (constraint_data, (constraint_rows, constraint_cols)),
                shape=(len(lower_bounds), n_vars),
            ).tocsr()
            constraints.append(
                LinearConstraint(a, np.array(lower_bounds), np.array(upper_bounds))
            )

        return milp(
            c=c,
            constraints=constraints,
            bounds=Bounds(0.0, 1.0),
            integrality=np.ones(n_vars, dtype=int),
            options={"time_limit": 30.0},
        )

    def finalize(
        res: Any, objective: float, status: str
    ) -> Tuple[List[ExpandedCandidate], float, str]:
        xval = res.x[:n_x]
        return [expanded[i] for i, v in enumerate(xval) if v > 0.5], objective, status

    try:
        if mode == "integrated":
            res = solve(include_bond_environment=True)
            if getattr(res, "success", False) and getattr(res, "x", None) is not None:
                return finalize(res, -float(res.fun), "ilp")
        else:
            res = solve(include_bond_environment=False)
            if getattr(res, "success", False) and getattr(res, "x", None) is not None:
                primary_objective = -float(res.fun)
                if mode == "rerank" and cfg.broken_bond_environment_penalty > 0.0:
                    floor = primary_objective - max(
                        0.0, cfg.bond_environment_rank_tolerance
                    )
                    rerank = solve(
                        include_bond_environment=True,
                        primary_floor=floor,
                        secondary_only=True,
                    )
                    if (
                        getattr(rerank, "success", False)
                        and getattr(rerank, "x", None) is not None
                    ):
                        return finalize(rerank, primary_objective, "ilp_rerank")
                    return finalize(
                        res,
                        primary_objective,
                        f"ilp_rerank_primary_only_after_status:{getattr(rerank, 'message', 'unknown')}",
                    )
                return finalize(res, primary_objective, "ilp")
    except Exception as e:
        if cfg.fallback_to_greedy:
            chosen, obj, status = select_candidates_greedy(
                base_candidates, reactants, cfg
            )
            return chosen, obj, f"greedy_fallback_after_ilp_error:{e}"
        raise

    if cfg.fallback_to_greedy:
        chosen, obj, status = select_candidates_greedy(base_candidates, reactants, cfg)
        return (
            chosen,
            obj,
            f"greedy_fallback_after_ilp_status:{getattr(res, 'message', 'unknown')}",
        )
    raise RuntimeError(f"MILP failed: {getattr(res, 'message', 'unknown')}")


def selected_pieces_from_expanded(
    chosen: Sequence[ExpandedCandidate],
) -> List[SelectedPiece]:
    pieces: List[SelectedPiece] = []
    for x in chosen:
        pieces.append(
            SelectedPiece(
                reactant_id=x.reactant_id,
                copy_id=x.copy_id,
                candidate_id=x.base.cid,
                source=x.base.source,
                reactant_atoms=x.reactant_atoms,
                product_atoms=x.product_atoms,
                r_to_p=dict(x.r_to_p),
                preserved_bonds=x.base.preserved_bonds,
                extra_product_edges=x.base.extra_product_edges,
                score=x.score,
            )
        )
    pieces.sort(
        key=lambda p: (p.reactant_id, p.copy_id, -len(p.product_atoms), p.product_atoms)
    )
    return pieces


# ---------------------------------------------------------------------------
# Topology diagnostics
# ---------------------------------------------------------------------------


def lineage_label(lineage: Tuple[int, int]) -> str:
    rid, copy = lineage
    return f"R{rid}/copy{copy}"


def source_to_jsonable(src: Any) -> str:
    if (
        isinstance(src, tuple)
        and len(src) == 2
        and all(isinstance(x, int) for x in src)
    ):
        return lineage_label(src)
    return str(src)


def collapse_consecutive(xs: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out


def safe_shortest_path(
    g: nx.Graph, source: int, target: int
) -> Tuple[float, List[int]]:
    try:
        path = nx.shortest_path(g, source, target)
        return float(len(path) - 1), list(path)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return math.inf, []


def build_mapping_indexes(
    pieces: Sequence[SelectedPiece], product_graph: nx.Graph
) -> Tuple[
    Dict[int, Tuple[int, int]],
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int, int], int],
    Dict[Tuple[int, int], Set[int]],
]:
    """Return product source, reactant-copy->product mapping, inverse, lineage atoms."""
    product_source: Dict[int, Tuple[int, int]] = {}
    rcopy_to_product: Dict[Tuple[int, int, int], int] = {}
    product_to_rcopy_atom: Dict[Tuple[int, int, int], int] = {}
    atoms_by_lineage: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
    for piece in pieces:
        lin = (piece.reactant_id, piece.copy_id)
        for r, p in piece.r_to_p.items():
            product_source[p] = lin
            rcopy_to_product[(piece.reactant_id, piece.copy_id, r)] = p
            product_to_rcopy_atom[(piece.reactant_id, piece.copy_id, p)] = r
            atoms_by_lineage[lin].add(p)
    return product_source, rcopy_to_product, product_to_rcopy_atom, atoms_by_lineage


def mapped_anchor_segments(
    r_graph: nx.Graph,
    mapped_atoms: Set[int],
    max_distance: int,
) -> List[Dict[str, Any]]:
    """Return compressed segments between mapped reactant anchor atoms.

    A segment is a pair of mapped atoms whose shortest path in the reactant has
    no mapped interior atom.  This catches partial mappings such as
    [C:1]C[C:2] where only the endpoints are anchors.
    """
    mapped = sorted(mapped_atoms)
    segments: List[Dict[str, Any]] = []
    seen: Set[Tuple[int, int]] = set()
    for i, u in enumerate(mapped):
        for v in mapped[i + 1 :]:
            try:
                path = nx.shortest_path(r_graph, u, v)
            except nx.NetworkXNoPath:
                continue
            d = len(path) - 1
            if d > max_distance:
                continue
            interior = path[1:-1]
            if any(x in mapped_atoms for x in interior):
                continue
            key = (u, v)
            if key in seen:
                continue
            seen.add(key)
            segments.append(
                {
                    "reactant_atoms": [u, v],
                    "reactant_distance": d,
                    "reactant_path": path,
                    "interior_unmapped_reactant_atoms": interior,
                }
            )
    segments.sort(key=lambda e: (e["reactant_distance"], e["reactant_atoms"]))
    return segments


def product_lineage_blocks(
    product_graph: nx.Graph,
    product_source: Mapping[int, Tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Contract maximal connected blocks with the same product source."""
    # Blocks are computed over all product atoms.  Uncovered atoms get source
    # ('uncovered', -1) but will be printed as 'uncovered'.
    node_source: Dict[int, Any] = {
        p: product_source.get(p, "uncovered") for p in product_graph.nodes
    }
    visited: Set[int] = set()
    blocks: List[Dict[str, Any]] = []
    block_id_of_node: Dict[int, int] = {}
    for start in product_graph.nodes:
        if start in visited:
            continue
        src = node_source[start]
        stack = [start]
        visited.add(start)
        atoms: List[int] = []
        while stack:
            u = stack.pop()
            atoms.append(u)
            for v in product_graph.neighbors(u):
                if v not in visited and node_source[v] == src:
                    visited.add(v)
                    stack.append(v)
        bid = len(blocks)
        for a in atoms:
            block_id_of_node[a] = bid
        blocks.append(
            {
                "block_id": bid,
                "source": source_to_jsonable(src),
                "atoms": sorted(atoms),
                "size": len(atoms),
            }
        )

    q_edges_set: Set[Tuple[int, int]] = set()
    for u, v in product_graph.edges:
        bu, bv = block_id_of_node[u], block_id_of_node[v]
        if bu != bv:
            q_edges_set.add(tuple(sorted((bu, bv))))
    q_edges = [{"block_1": a, "block_2": b} for a, b in sorted(q_edges_set)]
    return blocks, q_edges


def _copy_mol_clearing_atom_maps(mol: Chem.Mol) -> Chem.Mol:
    """Return a shallow molecule copy with atom-map numbers removed."""
    out = Chem.Mol(mol)
    for atom in out.GetAtoms():
        atom.SetAtomMapNum(0)
    return out


def fragment_smiles_for_atoms(
    mol: Chem.Mol, atoms: Iterable[int], clear_atom_maps: bool = False
) -> str:
    """Return canonical SMILES for a fragment induced by atoms.

    The atom set is expected to be connected, but RDKit can also render a
    disconnected set.  For residual reporting we call this on connected
    components so summaries prioritize fragments over individual atoms.
    """
    atom_list = sorted(int(a) for a in atoms)
    if not atom_list:
        return ""
    use_mol = _copy_mol_clearing_atom_maps(mol) if clear_atom_maps else mol
    return Chem.MolFragmentToSmiles(use_mol, atomsToUse=atom_list, canonical=True)


def connected_atom_components(graph: nx.Graph, atoms: Iterable[int]) -> List[List[int]]:
    """Connected components of graph induced by atoms, largest first."""
    atom_set = set(int(a) for a in atoms)
    if not atom_set:
        return []
    sub = graph.subgraph(atom_set)
    comps = (
        [sorted(c) for c in nx.connected_components(sub)]
        if sub.number_of_nodes()
        else []
    )
    comps.sort(key=lambda xs: (-len(xs), xs))
    return comps


def edge_count_in_atom_set(graph: nx.Graph, atoms: Iterable[int]) -> int:
    atom_set = set(int(a) for a in atoms)
    return sum(1 for u, v in graph.edges if u in atom_set and v in atom_set)


def boundary_bonds_for_atom_set(
    graph: nx.Graph, atoms: Iterable[int]
) -> List[List[int]]:
    atom_set = set(int(a) for a in atoms)
    bonds: Set[Tuple[int, int]] = set()
    for u in atom_set:
        for v in graph.neighbors(u):
            if v not in atom_set:
                bonds.add(tuple(sorted((u, v))))
    return [list(b) for b in sorted(bonds)]


def product_component_lookup(
    product_graph: nx.Graph,
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Return atom->component and component->atoms for product connected components."""
    atom_to_component: Dict[int, int] = {}
    component_atoms: Dict[int, List[int]] = {}
    comps = (
        [sorted(c) for c in nx.connected_components(product_graph)]
        if product_graph.number_of_nodes()
        else []
    )
    comps.sort(key=lambda xs: (xs[0] if xs else INF, xs))
    for cid, atoms in enumerate(comps):
        component_atoms[cid] = atoms
        for atom in atoms:
            atom_to_component[atom] = cid
    return atom_to_component, component_atoms


def compute_residual_fragments(
    reactants: Sequence[ReactantComponent],
    product_mol: Chem.Mol,
    product_graph: nx.Graph,
    pieces: Sequence[SelectedPiece],
) -> Dict[str, Any]:
    """Summarize product and reactant remainders after selected subtractions.

    Product residuals are connected components of product atoms not covered by
    any selected common-subgraph piece.  A residual that is an entire disconnected
    product molecule/component is flagged as a likely byproduct or missing-source
    product; a residual that touches selected mapped material is flagged as a
    partial unmapped product fragment.

    Reactant residuals are unused connected fragments from active virtual
    reactant copies, plus whole original reactant components that were never
    selected at all.  We do not list every inactive virtual copy, because those
    are just optional copies that were not needed.
    """
    product_source, rcopy_to_product, _product_to_rcopy_atom, atoms_by_lineage = (
        build_mapping_indexes(pieces, product_graph)
    )
    covered_product_atoms = set(product_source)
    uncovered_product_atoms = set(product_graph.nodes) - covered_product_atoms

    product_atom_to_component, product_components = product_component_lookup(
        product_graph
    )
    product_residuals: List[Dict[str, Any]] = []
    for ridx, atoms in enumerate(
        connected_atom_components(product_graph, uncovered_product_atoms)
    ):
        atom_set = set(atoms)
        parent_ids = sorted(
            {
                product_atom_to_component[a]
                for a in atoms
                if a in product_atom_to_component
            }
        )
        is_whole_component = False
        if len(parent_ids) == 1:
            parent_atoms = set(product_components[parent_ids[0]])
            is_whole_component = atom_set == parent_atoms
        boundary = boundary_bonds_for_atom_set(product_graph, atoms)
        adjacent_sources = sorted(
            {
                source_to_jsonable(product_source[v])
                for u, v in (tuple(b) for b in boundary)
                if v in product_source and u in atom_set
            }
            | {
                source_to_jsonable(product_source[u])
                for u, v in (tuple(b) for b in boundary)
                if u in product_source and v in atom_set
            }
        )
        classification = (
            "whole_uncovered_product_component"
            if is_whole_component
            else "partial_uncovered_product_fragment"
        )
        product_residuals.append(
            {
                "residual_id": ridx,
                "classification": classification,
                "atoms": atoms,
                "size": len(atoms),
                "bond_count": edge_count_in_atom_set(product_graph, atoms),
                "smiles": fragment_smiles_for_atoms(
                    product_mol, atoms, clear_atom_maps=False
                ),
                "unmapped_smiles": fragment_smiles_for_atoms(
                    product_mol, atoms, clear_atom_maps=True
                ),
                "parent_product_components": parent_ids,
                "is_whole_product_component": is_whole_component,
                "touches_selected_mapping": bool(adjacent_sources),
                "boundary_bonds_to_nonresidual_atoms": boundary,
                "adjacent_selected_lineages": adjacent_sources,
            }
        )

    product_residuals.sort(key=lambda e: (-e["size"], e["classification"], e["atoms"]))
    for i, entry in enumerate(product_residuals):
        entry["residual_id"] = i

    byproduct_candidates = [
        e
        for e in product_residuals
        if e["classification"] == "whole_uncovered_product_component"
    ]
    partial_product_residuals = [
        e
        for e in product_residuals
        if e["classification"] == "partial_uncovered_product_fragment"
    ]

    active_lineages = sorted(atoms_by_lineage)
    active_reactants = {rid for rid, _k in active_lineages}
    reactant_residuals: List[Dict[str, Any]] = []

    for rid, k in active_lineages:
        rc = reactants[rid]
        used_r = {
            r for (rr, kk, r), _p in rcopy_to_product.items() if rr == rid and kk == k
        }
        unused_r = set(rc.graph.nodes) - used_r
        for atoms in connected_atom_components(rc.graph, unused_r):
            boundary = boundary_bonds_for_atom_set(rc.graph, atoms)
            reactant_residuals.append(
                {
                    "classification": "unused_fragment_in_active_reactant_copy",
                    "reactant_id": rid,
                    "copy_id": k,
                    "lineage": lineage_label((rid, k)),
                    "atoms": atoms,
                    "size": len(atoms),
                    "bond_count": edge_count_in_atom_set(rc.graph, atoms),
                    "smiles": fragment_smiles_for_atoms(
                        rc.mol, atoms, clear_atom_maps=False
                    ),
                    "unmapped_smiles": fragment_smiles_for_atoms(
                        rc.mol, atoms, clear_atom_maps=True
                    ),
                    "boundary_bonds_to_selected_reactant_atoms": boundary,
                }
            )

    for rc in reactants:
        if rc.rid in active_reactants:
            continue
        atoms = sorted(rc.graph.nodes)
        if not atoms:
            continue
        reactant_residuals.append(
            {
                "classification": "unused_reactant_component",
                "reactant_id": rc.rid,
                "copy_id": None,
                "lineage": f"R{rc.rid}/unused_component",
                "atoms": atoms,
                "size": len(atoms),
                "bond_count": edge_count_in_atom_set(rc.graph, atoms),
                "smiles": Chem.MolToSmiles(rc.mol, canonical=True),
                "unmapped_smiles": Chem.MolToSmiles(
                    _copy_mol_clearing_atom_maps(rc.mol), canonical=True
                ),
                "boundary_bonds_to_selected_reactant_atoms": [],
            }
        )

    reactant_residuals.sort(
        key=lambda e: (
            -e["size"],
            e["classification"],
            e["reactant_id"],
            -1 if e["copy_id"] is None else e["copy_id"],
            e["atoms"],
        )
    )
    for i, entry in enumerate(reactant_residuals):
        entry["residual_id"] = i

    product_residual_smiles = ".".join(
        e["unmapped_smiles"] for e in product_residuals if e["unmapped_smiles"]
    )
    product_byproduct_candidate_smiles = ".".join(
        e["unmapped_smiles"] for e in byproduct_candidates if e["unmapped_smiles"]
    )
    reactant_residual_smiles = ".".join(
        e["unmapped_smiles"] for e in reactant_residuals if e["unmapped_smiles"]
    )

    return {
        "product_residual_fragments": product_residuals,
        "product_byproduct_candidates": byproduct_candidates,
        "product_partial_unmapped_fragments": partial_product_residuals,
        "reactant_residual_fragments": reactant_residuals,
        "product_residual_smiles": product_residual_smiles,
        "product_byproduct_candidate_smiles": product_byproduct_candidate_smiles,
        "reactant_residual_smiles": reactant_residual_smiles,
        "counts": {
            "product_residual_fragment_count": len(product_residuals),
            "product_residual_atom_count": len(uncovered_product_atoms),
            "product_byproduct_candidate_count": len(byproduct_candidates),
            "product_partial_unmapped_fragment_count": len(partial_product_residuals),
            "reactant_residual_fragment_count": len(reactant_residuals),
            "reactant_residual_atom_count": sum(
                int(e["size"]) for e in reactant_residuals
            ),
            "active_reactant_residual_fragment_count": sum(
                1
                for e in reactant_residuals
                if e["classification"] == "unused_fragment_in_active_reactant_copy"
            ),
            "unused_reactant_component_count": sum(
                1
                for e in reactant_residuals
                if e["classification"] == "unused_reactant_component"
            ),
        },
    }


def compute_diagnostics(
    reactants: Sequence[ReactantComponent],
    product_mol: Chem.Mol,
    product_graph: nx.Graph,
    pieces: Sequence[SelectedPiece],
    cfg: ReactionAtomMapperConfig,
) -> Dict[str, Any]:
    product_source, rcopy_to_product, product_to_rcopy_atom, atoms_by_lineage = (
        build_mapping_indexes(pieces, product_graph)
    )

    covered_product_atoms = set(product_source)
    uncovered_product_atoms = sorted(set(product_graph.nodes) - covered_product_atoms)

    active_copies_counter = Counter((p.reactant_id, p.copy_id) for p in pieces)
    active_copies_by_reactant: Dict[str, int] = defaultdict(int)
    pieces_by_lineage: Dict[Tuple[int, int], List[SelectedPiece]] = defaultdict(list)
    for p in pieces:
        active_copies_by_reactant[str(p.reactant_id)] = max(
            active_copies_by_reactant[str(p.reactant_id)], p.copy_id + 1
        )
        pieces_by_lineage[(p.reactant_id, p.copy_id)].append(p)

    # Unused reactant atoms by active copy.  For inactive copies, every atom is
    # unused by definition, but we usually care about active lineages.
    unused_reactant_atoms_active: Dict[str, List[int]] = {}
    for lin in sorted(atoms_by_lineage):
        rid, k = lin
        used_r = {
            r for (rr, kk, r), p in rcopy_to_product.items() if rr == rid and kk == k
        }
        all_r = set(reactants[rid].graph.nodes)
        unused_reactant_atoms_active[lineage_label(lin)] = sorted(all_r - used_r)

    # Lineage split: same lineage product atoms induce multiple connected blocks.
    lineage_split_events: List[Dict[str, Any]] = []
    for lin, p_atoms in sorted(atoms_by_lineage.items()):
        if not p_atoms:
            continue
        sub = product_graph.subgraph(p_atoms)
        comps = (
            [sorted(c) for c in nx.connected_components(sub)]
            if sub.number_of_nodes()
            else []
        )
        if len(comps) > 1:
            lineage_split_events.append(
                {
                    "lineage": lineage_label(lin),
                    "num_product_blocks": len(comps),
                    "extra_blocks": len(comps) - 1,
                    "blocks": comps,
                    "piece_count_for_lineage": len(pieces_by_lineage.get(lin, [])),
                }
            )

    # Reactant bond preservation/breakage/deletion.
    reactant_bond_events: List[Dict[str, Any]] = []
    for lin in sorted(atoms_by_lineage):
        rid, k = lin
        rc = reactants[rid]
        for ru, rv, rdata in rc.graph.edges(data=True):
            key_u = (rid, k, ru)
            key_v = (rid, k, rv)
            mu = rcopy_to_product.get(key_u)
            mv = rcopy_to_product.get(key_v)
            if mu is None or mv is None:
                reactant_bond_events.append(
                    {
                        "event": "reactant_bond_deleted_or_unmapped",
                        "lineage": lineage_label(lin),
                        "reactant_bond": [ru, rv],
                        "mapped_product_atoms": [mu, mv],
                    }
                )
            elif product_graph.has_edge(mu, mv):
                compatible = bond_compatible_attrs(
                    rdata, product_graph.edges[mu, mv], cfg
                )
                reactant_bond_events.append(
                    {
                        "event": (
                            "reactant_bond_preserved"
                            if compatible
                            else "reactant_bond_order_changed"
                        ),
                        "lineage": lineage_label(lin),
                        "reactant_bond": [ru, rv],
                        "product_bond": [mu, mv],
                    }
                )
            else:
                reactant_bond_events.append(
                    {
                        "event": "reactant_bond_broken",
                        "lineage": lineage_label(lin),
                        "reactant_bond": [ru, rv],
                        "mapped_product_atoms": [mu, mv],
                    }
                )

    # Product bond provenance.
    product_bond_events: List[Dict[str, Any]] = []
    for pu, pv, pdata in product_graph.edges(data=True):
        su = product_source.get(pu)
        sv = product_source.get(pv)
        if su is None or sv is None:
            product_bond_events.append(
                {
                    "event": "product_bond_touches_uncovered_atom",
                    "product_bond": [pu, pv],
                    "source_1": source_to_jsonable(
                        su if su is not None else "uncovered"
                    ),
                    "source_2": source_to_jsonable(
                        sv if sv is not None else "uncovered"
                    ),
                }
            )
        elif su != sv:
            product_bond_events.append(
                {
                    "event": "interlineage_product_bond_formed",
                    "product_bond": [pu, pv],
                    "source_1": source_to_jsonable(su),
                    "source_2": source_to_jsonable(sv),
                }
            )
        else:
            rid, k = su
            ru = product_to_rcopy_atom.get((rid, k, pu))
            rv = product_to_rcopy_atom.get((rid, k, pv))
            if ru is None or rv is None:
                continue
            if reactants[rid].graph.has_edge(ru, rv):
                compatible = bond_compatible_attrs(
                    reactants[rid].graph.edges[ru, rv], pdata, cfg
                )
                product_bond_events.append(
                    {
                        "event": (
                            "product_bond_explained_by_reactant_bond"
                            if compatible
                            else "product_bond_order_changed_from_reactant"
                        ),
                        "product_bond": [pu, pv],
                        "lineage": lineage_label(su),
                        "reactant_bond": [ru, rv],
                    }
                )
            else:
                product_bond_events.append(
                    {
                        "event": "intralineage_product_bond_formed",
                        "product_bond": [pu, pv],
                        "lineage": lineage_label(su),
                        "reactant_atoms": [ru, rv],
                    }
                )

    # Segment / path diagnostics.
    segment_events: Dict[str, List[Dict[str, Any]]] = {
        "segments": [],
        "lineage_restricted_breaks": [],
        "foreign_or_unknown_bridged_breaks": [],
        "stretches": [],
        "contractions": [],
    }
    for lin in sorted(atoms_by_lineage):
        rid, k = lin
        rc = reactants[rid]
        mapped_r_atoms = {
            r for (rr, kk, r), p in rcopy_to_product.items() if rr == rid and kk == k
        }
        if len(mapped_r_atoms) < 2:
            continue
        segments = mapped_anchor_segments(
            rc.graph, mapped_r_atoms, cfg.max_segment_distance
        )
        same_lineage_product_atoms = atoms_by_lineage[lin]
        same_subgraph = product_graph.subgraph(same_lineage_product_atoms).copy()
        for seg in segments:
            ru, rv = seg["reactant_atoms"]
            pu = rcopy_to_product[(rid, k, ru)]
            pv = rcopy_to_product[(rid, k, rv)]
            d_full, path_full = safe_shortest_path(product_graph, pu, pv)
            d_same, path_same = safe_shortest_path(same_subgraph, pu, pv)
            event = {
                "lineage": lineage_label(lin),
                "reactant_atoms": [ru, rv],
                "product_atoms": [pu, pv],
                "reactant_distance": seg["reactant_distance"],
                "reactant_path": seg["reactant_path"],
                "product_distance_full": None if math.isinf(d_full) else int(d_full),
                "product_path_full": path_full,
                "product_distance_same_lineage": (
                    None if math.isinf(d_same) else int(d_same)
                ),
                "product_path_same_lineage": path_same,
            }
            segment_events["segments"].append(event)
            if math.isinf(d_same):
                segment_events["lineage_restricted_breaks"].append(event)
                if not math.isinf(d_full):
                    source_sequence = [
                        product_source.get(a, "uncovered") for a in path_full
                    ]
                    bridge_sources = [s for s in source_sequence[1:-1] if s != lin]
                    bridged_event = dict(event)
                    bridged_event.update(
                        {
                            "source_sequence": [
                                source_to_jsonable(s) for s in source_sequence
                            ],
                            "collapsed_source_sequence": [
                                source_to_jsonable(s)
                                for s in collapse_consecutive(source_sequence)
                            ],
                            "bridge_sources": sorted(
                                {source_to_jsonable(s) for s in bridge_sources}
                            ),
                            "foreign_or_unknown_bridge_atom_count": len(bridge_sources),
                        }
                    )
                    segment_events["foreign_or_unknown_bridged_breaks"].append(
                        bridged_event
                    )
            d_r = int(seg["reactant_distance"])
            if not math.isinf(d_full):
                if d_full > d_r:
                    stretch_event = dict(event)
                    stretch_event["stretch"] = int(d_full - d_r)
                    segment_events["stretches"].append(stretch_event)
                elif d_full < d_r:
                    contract_event = dict(event)
                    contract_event["contraction"] = int(d_r - d_full)
                    segment_events["contractions"].append(contract_event)

    blocks, q_edges = product_lineage_blocks(product_graph, product_source)
    residuals = compute_residual_fragments(
        reactants, product_mol, product_graph, pieces
    )

    # Counts for easy logging/filtering.
    rb_counts = Counter(e["event"] for e in reactant_bond_events)
    pb_counts = Counter(e["event"] for e in product_bond_events)
    topology_counts = {
        "selected_piece_count": len(pieces),
        "active_lineage_count": len(atoms_by_lineage),
        "covered_product_atom_count": len(covered_product_atoms),
        "uncovered_product_atom_count": len(uncovered_product_atoms),
        "lineage_split_event_count": len(lineage_split_events),
        "lineage_extra_block_count": sum(
            e["extra_blocks"] for e in lineage_split_events
        ),
        "reactant_bond_preserved_count": rb_counts.get("reactant_bond_preserved", 0),
        "reactant_bond_broken_count": rb_counts.get("reactant_bond_broken", 0),
        "reactant_bond_deleted_or_unmapped_count": rb_counts.get(
            "reactant_bond_deleted_or_unmapped", 0
        ),
        "interlineage_product_bond_formed_count": pb_counts.get(
            "interlineage_product_bond_formed", 0
        ),
        "intralineage_product_bond_formed_count": pb_counts.get(
            "intralineage_product_bond_formed", 0
        ),
        "product_bond_touches_uncovered_atom_count": pb_counts.get(
            "product_bond_touches_uncovered_atom", 0
        ),
        "lineage_restricted_break_count": len(
            segment_events["lineage_restricted_breaks"]
        ),
        "foreign_or_unknown_bridged_break_count": len(
            segment_events["foreign_or_unknown_bridged_breaks"]
        ),
        "foreign_or_unknown_bridge_atom_count": sum(
            e.get("foreign_or_unknown_bridge_atom_count", 0)
            for e in segment_events["foreign_or_unknown_bridged_breaks"]
        ),
        "segment_stretch_count": len(segment_events["stretches"]),
        "segment_stretch_total": sum(
            e.get("stretch", 0) for e in segment_events["stretches"]
        ),
        "segment_contraction_count": len(segment_events["contractions"]),
        "segment_contraction_total": sum(
            e.get("contraction", 0) for e in segment_events["contractions"]
        ),
        "product_residual_fragment_count": residuals["counts"][
            "product_residual_fragment_count"
        ],
        "product_residual_atom_count": residuals["counts"][
            "product_residual_atom_count"
        ],
        "product_byproduct_candidate_count": residuals["counts"][
            "product_byproduct_candidate_count"
        ],
        "product_partial_unmapped_fragment_count": residuals["counts"][
            "product_partial_unmapped_fragment_count"
        ],
        "reactant_residual_fragment_count": residuals["counts"][
            "reactant_residual_fragment_count"
        ],
        "reactant_residual_atom_count": residuals["counts"][
            "reactant_residual_atom_count"
        ],
    }

    return {
        "active_copies_by_reactant": dict(active_copies_by_reactant),
        "selected_piece_count_by_lineage": {
            lineage_label(k): len(v) for k, v in sorted(pieces_by_lineage.items())
        },
        "uncovered_product_atoms": uncovered_product_atoms,
        "unused_reactant_atoms_by_active_lineage": unused_reactant_atoms_active,
        "lineage_split_events": lineage_split_events,
        "reactant_bond_events": reactant_bond_events,
        "product_bond_events": product_bond_events,
        "segment_events": segment_events,
        "product_lineage_quotient": {"blocks": blocks, "edges": q_edges},
        "residuals": residuals,
        "topology_counts": topology_counts,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def subtractive_map_reaction(
    reaction_smiles: str,
    config: Optional[ReactionAtomMapperConfig] = None,
    **config_overrides: Any,
) -> SubtractiveMappingResult:
    """Run subtractive common-subgraph mapping and topology diagnostics.

    Parameters
    ----------
    reaction_smiles:
        Reaction SMILES, either reactants>>products or reactants>agents>products.
    config:
        Optional MapperConfig.  Keyword overrides can also be supplied.

    Returns
    -------
    SubtractiveMappingResult
    """
    cfg = config or ReactionAtomMapperConfig()
    if config_overrides:
        cfg = dataclasses.replace(cfg, **config_overrides)
    if cfg.max_copies < 1:
        raise ValueError("max_copies must be at least 1.")
    left, right = parse_reaction_smiles_to_halves(reaction_smiles)
    reactants = split_reactant_components(left)
    product_mol = mol_from_side(right)
    product_graph = mol_to_nx(product_mol)
    respect_maps = auto_respect_atom_maps(reactants, product_mol, cfg)

    base_candidates = generate_base_candidates(
        reactants, product_mol, product_graph, cfg, respect_maps
    )

    if cfg.selector == "ilp":
        chosen, objective, status = select_candidates_ilp(
            base_candidates, reactants, product_graph, cfg
        )
    elif cfg.selector == "greedy":
        chosen, objective, status = select_candidates_greedy(
            base_candidates, reactants, cfg
        )
    else:
        raise ValueError("selector must be 'ilp' or 'greedy'.")

    pieces = selected_pieces_from_expanded(chosen)
    diagnostics = compute_diagnostics(
        reactants, product_mol, product_graph, pieces, cfg
    )
    diagnostics["candidate_generation"] = {
        "base_candidate_count": len(base_candidates),
        "expanded_candidate_count": len(
            expand_candidates(base_candidates, cfg, reactants)
        ),
        "respect_atom_maps": respect_maps,
    }
    return SubtractiveMappingResult(
        reaction_smiles=reaction_smiles,
        selector=cfg.selector,
        objective_value=objective,
        status=status,
        reactant_components=list(reactants),
        product_mol=product_mol,
        product_graph=product_graph,
        selected_pieces=pieces,
        diagnostics=diagnostics,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_bool_auto(value: str) -> Optional[bool]:
    v = value.lower().strip()
    if v in {"auto", "none"}:
        return None
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected auto, true, or false.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Subtractive common-subgraph reaction mapper."
    )
    p.add_argument("reaction", help="Reaction SMILES, e.g. 'CC.CNC>>CCNCC'.")
    p.add_argument("--selector", choices=["ilp", "greedy"], default="ilp")
    p.add_argument("--max-copies", type=int, default=3)
    p.add_argument("--min-fragment-atoms", type=int, default=1)
    p.add_argument("--max-fragment-atoms", type=int, default=8)
    p.add_argument("--max-fragments-per-reactant", type=int, default=2500)
    p.add_argument("--max-matches-per-fragment", type=int, default=128)
    p.add_argument("--max-base-candidates-per-reactant", type=int, default=6000)
    p.add_argument(
        "--respect-atom-maps",
        type=_parse_bool_auto,
        default=None,
        help="auto, true, or false; default auto",
    )
    p.add_argument(
        "--no-rdkit-mcs", action="store_true", help="Disable RDKit MCS seed candidates."
    )
    p.add_argument(
        "--allow-mapped-reactant-copies",
        action="store_true",
        help="Allow reactant components containing atom-map anchors to use multiple virtual copies.",
    )
    p.add_argument(
        "--unused-reactant-atom-penalty",
        type=float,
        default=6.0,
        help="Penalty per unused atom in each active reactant copy.",
    )
    p.add_argument(
        "--broken-bond-environment-penalty",
        type=float,
        default=1.0,
        help="Scale for local-environment penalties on broken reactant bonds.",
    )
    p.add_argument(
        "--bond-environment-objective",
        choices=["off", "integrated", "rerank"],
        default="off",
        help="How to use local bond-environment penalties in ILP selection.",
    )
    p.add_argument(
        "--bond-environment-rank-tolerance",
        type=float,
        default=1.0e-6,
        help="Primary-objective tolerance for reranking near-tied ILP solutions.",
    )
    p.add_argument(
        "--stable-single-bond-break-penalty",
        type=float,
        default=1.0,
        help="Extra penalty for breaking saturated-carbon/hetero single bonds.",
    )
    p.add_argument(
        "--unsaturated-endpoint-break-credit",
        type=float,
        default=0.75,
        help="Credit for breaking bonds attached to atoms with multiple bonds to hetero atoms.",
    )
    p.add_argument(
        "--ring-bond-break-penalty",
        type=float,
        default=2.0,
        help="Extra penalty for breaking ring bonds.",
    )
    p.add_argument(
        "--max-broken-bond-pair-penalty-terms",
        type=int,
        default=25000,
        help="Maximum pairwise broken-bond ILP terms to keep; 0 means no cap.",
    )
    p.add_argument("--ignore-bond-order", action="store_true")
    p.add_argument("--json-indent", type=int, default=2)
    p.add_argument(
        "--summary",
        action="store_true",
        help="Print a concise summary instead of full JSON.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = ReactionAtomMapperConfig(
        selector=args.selector,
        max_copies=args.max_copies,
        min_fragment_atoms=args.min_fragment_atoms,
        max_fragment_atoms=args.max_fragment_atoms,
        max_fragments_per_reactant=args.max_fragments_per_reactant,
        max_matches_per_fragment=args.max_matches_per_fragment,
        max_base_candidates_per_reactant=args.max_base_candidates_per_reactant,
        respect_atom_maps=args.respect_atom_maps,
        include_rdkit_mcs_candidates=not args.no_rdkit_mcs,
        compare_bond_order=not args.ignore_bond_order,
        mapped_reactants_single_copy=not args.allow_mapped_reactant_copies,
        unused_reactant_atom_penalty_active_copy=args.unused_reactant_atom_penalty,
        broken_bond_environment_penalty=args.broken_bond_environment_penalty,
        bond_environment_objective=args.bond_environment_objective,
        bond_environment_rank_tolerance=args.bond_environment_rank_tolerance,
        stable_single_bond_break_penalty=args.stable_single_bond_break_penalty,
        unsaturated_endpoint_break_credit=args.unsaturated_endpoint_break_credit,
        ring_bond_break_penalty=args.ring_bond_break_penalty,
        max_broken_bond_pair_penalty_terms=args.max_broken_bond_pair_penalty_terms,
    )
    result = subtractive_map_reaction(args.reaction, cfg)
    if args.summary:
        print(
            json.dumps(
                {
                    "reaction_smiles": result.reaction_smiles,
                    "atom_mapped_reaction_smiles": result.atom_mapped_reaction_smiles(),
                    "status": result.status,
                    "objective_value": result.objective_value,
                    "selected_pieces": [
                        {
                            "lineage": lineage_label((p.reactant_id, p.copy_id)),
                            "reactant_atoms": list(p.reactant_atoms),
                            "product_atoms": list(p.product_atoms),
                            "source": p.source,
                        }
                        for p in result.selected_pieces
                    ],
                    "topology_counts": result.diagnostics["topology_counts"],
                    "residual_summary": {
                        "product_residual_smiles": result.diagnostics["residuals"][
                            "product_residual_smiles"
                        ],
                        "product_byproduct_candidate_smiles": result.diagnostics[
                            "residuals"
                        ]["product_byproduct_candidate_smiles"],
                        "reactant_residual_smiles": result.diagnostics["residuals"][
                            "reactant_residual_smiles"
                        ],
                        "counts": result.diagnostics["residuals"]["counts"],
                    },
                    "product_residual_fragments": [
                        {
                            "classification": f["classification"],
                            "smiles": f["smiles"],
                            "unmapped_smiles": f["unmapped_smiles"],
                            "atoms": f["atoms"],
                            "size": f["size"],
                            "touches_selected_mapping": f["touches_selected_mapping"],
                        }
                        for f in result.diagnostics["residuals"][
                            "product_residual_fragments"
                        ]
                    ],
                    "reactant_residual_fragments": [
                        {
                            "classification": f["classification"],
                            "lineage": f["lineage"],
                            "smiles": f["smiles"],
                            "unmapped_smiles": f["unmapped_smiles"],
                            "atoms": f["atoms"],
                            "size": f["size"],
                        }
                        for f in result.diagnostics["residuals"][
                            "reactant_residual_fragments"
                        ]
                    ],
                    "candidate_generation": result.diagnostics["candidate_generation"],
                },
                indent=args.json_indent,
                sort_keys=True,
            )
        )
    else:
        print(result.to_json(indent=args.json_indent))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
