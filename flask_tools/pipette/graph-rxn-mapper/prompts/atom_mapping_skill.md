# Atom Mapping Skill

Use this workflow to infer reaction atom provenance.

1. Treat the input as two explicit molecular graphs: reactant side and product
   side. Use atom ids from those graphs, not SMILES character positions.
2. First identify unchanged scaffolds by preserving element identity, bond
   order, ring membership, aromatic systems, and local neighborhoods.
3. Then identify reaction centers by locating bonds broken in reactants and
   bonds formed in products.
4. Prefer mappings that explain all product atoms with the fewest chemically
   implausible lineage changes.
5. Use local bond environment, not named reaction memorization: carbonyl-like
   polarized centers, saturated carbon-hetero bonds, ring bonds, aromatic
   frameworks, pi systems, and formal charges should influence provenance.
6. For heteroatom provenance, decide which oxygen/nitrogen/sulfur atom becomes
   each product heteroatom by considering the bond that was broken and the bond
   that was formed.
7. For automorphic atoms, choose the assignment that best preserves adjacent
   atom environments and minimizes unnecessary bond-change distance.
8. Return only product-to-reactant atom id pairs. Do not output mapped SMILES
   directly unless explicitly requested.
