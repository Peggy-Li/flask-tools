Map this reaction.

Reaction index: {reaction_index}
Unmapped reaction SMILES:
{unmapped_reaction_smiles}

Reactant-side atom graph uses global reactant atom ids. Product-side atom graph
uses global product atom ids. Return product_to_reactant pairs using these ids.

Reactant graph JSON:
{reactant_graph_json}

Product graph JSON:
{product_graph_json}

Output JSON schema:
{{
  "product_to_reactant": [
    {{"product_atom": 0, "reactant_atom": 0}}
  ],
  "confidence": 0.0,
  "reasoning_summary": "brief chemistry rationale"
}}

The product_to_reactant list must contain exactly one entry for every product
atom id.
