# Pipette

Pipette grades predicted chemical reactions by running a sequence of validation
tools and combining their outputs into a final reaction grade.

The current pipeline includes:

- reaction SMILES parsing
- basic SMILES validation
- exact-match checker interfaces for reaction databases
- graph based balancing - Attempt to balance reaction by adding copies of reactants. Good for dimerization reactions
  - If this is enabled, the reaction fixing LLM is called here instead of later
- atom mapping
- reaction fixing LLM call
  - Run if no exact match is found
  - If new reaction is returned, goes back to start. Only allowed to run once in a pipeline
- charge conservation
- mass conservation with simple missing-product heuristics and solvent catalogs
- (Planned) reaction-energy checker interfaces for cached DFT results or external runs
- Grading:
  - an exact rule-based grader
  - an AI-judge pipeline interface that can run selected tools and hand the results
    to an external LLM judge

You must have an api key set with an environment variable, either `FLASK_ORCHESTRATOR_API_KEY, OPENAI_API_KEY, or PIPETTE_API_KEY`.
Optional: Setting the LLM model with `FLASK_ORCHESTRATOR_MODEL`, and setting the LLM url with `FLASK_ORCHESTRATOR_URL or PIPETTE_LLM_BASE_URL`


## Install

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e .[dev]
```

## Quick start

```shell
python -m flask_tools.pipette.grade_rxn --rxn-smi 'Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
```
Note, the default config is `llm-judge` (no DFT) and may be left out in the CLI and in the python code.

```python
from flask_tools.pipette.grade_rxn import grade_reaction
from flask_tools.pipette.config import load_config

result = grade_reaction(["Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]) # Optionally, config='llm-judge' or "file.yaml"

# Or
config = load_config('llm-judge')
result = grade_reaction(["Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C"], config=config)
```
This loads the config from `pipette/assets/llm-judge.yaml`.

Configs can also be a custom yaml file.


Example human-readable output
```aiignore
Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C:
ReactionGrade(final_grade=likely, short_comment=plausible_n_methylation)
comment: The reaction is chemically plausible: methyl iodide can N-methylate the xanthine NH to give the trimethylated product, with HI represented as [H+] and [I-]. The SMILES parses correctly, and both charge and mass are conserved.
tool_results:
  - basic_smiles_validation: pass - Reaction SMILES parsed successfully.
    {
      "reactant_count": 2,
      "product_count": 1
    }
  - exact_match: unknown - No reaction database backend is configured.
  - llm_reaction_fix: pass - N-methylation of the xanthine NH with methyl iodide requires HI as the byproduct, represented as [H+] and [I-]. No agents were present to remove.
    {
      "original_reaction_smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
      "fixed_reaction_smiles": "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
      "removed_agents": [],
      "added_reactants": [],
      "added_products": [
        "[H+]",
        "[I-]"
      ]
    }
  - basic_smiles_validation: pass - Reaction SMILES parsed successfully.
    {
      "reactant_count": 2,
      "product_count": 3
    }
  - exact_match: unknown - No reaction database backend is configured.
  - charge_conservation: pass - Charge is conserved.
    {
      "charge_difference": 0
    }
  - mass_conservation: pass - Element counts are conserved.
    {
      "mass_difference_amu": 0.0,
      "element_difference": {},
      "imbalanced_molecules": [],
      "imbalanced_molecule_confidence": null,
      "closest_stoich": null
    }
```

#### With DFT Reaction Energy Set up
Not implemented

## Reaction Fixing
Unbalanced reactions can be fixed by adding missing byproducts or reactants. This is a default part of the pipeline
and uses an LLM. An example of an unbalanced reaction is:

```smiles
Not balanced
Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Balanced
Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C.[OH3+].[I-]
```

## Disabling tools or reaction fixing

The llm reaction fixer can be disabled with `python grade_rxn.py --no-fix ...` or setting `use_fix: False` in the config
`settings` section.

The tools can also be disabled by setting `tool_list: null` in the the config.

## Full JSON Output
`results` from `grade_reactions()` is a list of `ReactionGrade` objects, which is a series of `ToolResults` and a final grade:
```
[
  {
    "rxn_smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "cleaned_rxn_smiles": "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
    "grade": {
      "final_grade": "likely",
      "short_comment": "plausible_n_methylation",
      "results": [
        {
          "name": "basic_smiles_validation",
          "status": "pass",
          "data": {
            "reactant_count": 2,
            "product_count": 1
          },
          "comment": "Reaction SMILES parsed successfully.",
          "skipped_reason": null
        },
        {
          "name": "exact_match",
          "status": "unknown",
          "data": {},
          "comment": "No reaction database backend is configured.",
          "skipped_reason": null
        },
        {
          "name": "llm_reaction_fix",
          "status": "pass",
          "data": {
            "original_reaction_smiles": "Cn1cnc2c1c(=O)[nH]c(=O)n2C.CI>>CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "fixed_reaction_smiles": "CI.Cn1cnc2c1c(=O)[nH]c(=O)n2C>>Cn1c(=O)c2c(ncn2C)n(C)c1=O.[H+].[I-]",
            "removed_agents": [],
            "added_reactants": [],
            "added_products": [
              "[H+]",
              "[I-]"
            ]
          },
          "comment": "N-methylation of the xanthine NH with methyl iodide requires HI as the byproduct, represented as [H+] and [I-]. No agents were present to remove.",
          "skipped_reason": null
        },
        {
          "name": "basic_smiles_validation",
          "status": "pass",
          "data": {
            "reactant_count": 2,
            "product_count": 3
          },
          "comment": "Reaction SMILES parsed successfully.",
          "skipped_reason": null
        },
        {
          "name": "exact_match",
          "status": "unknown",
          "data": {},
          "comment": "No reaction database backend is configured.",
          "skipped_reason": null
        },
        {
          "name": "charge_conservation",
          "status": "pass",
          "data": {
            "charge_difference": 0
          },
          "comment": "Charge is conserved.",
          "skipped_reason": null
        },
        {
          "name": "mass_conservation",
          "status": "pass",
          "data": {
            "mass_difference_amu": 0.0,
            "element_difference": {},
            "imbalanced_molecules": [],
            "imbalanced_molecule_confidence": null,
            "closest_stoich": null
          },
          "comment": "Element counts are conserved.",
          "skipped_reason": null
        }
      ],
      "comment": "The reaction is chemically plausible: methyl iodide can N-methylate the xanthine NH to give the trimethylated product, with HI represented as [H+] and [I-]. The SMILES parses correctly, and both charge and mass are conserved."
    }
  }
]

```

By default, `grade_reaction(...)` loads the packaged "llm-config-no-dft" rule based config YAML by from
`pipette/assets/ai_judge_no_dft.yaml`.

To restrict the pipeline to specific tools, set `PipetteConfig.tool_list` to
either `"all"` or an explicit list of tool names such as
`["basic_smiles_validation", "charge_conservation"]`.

To load your own config YAML config,

```shell
python -m flask_tools.pipette.grade_rxn --rxn-smi 'CCO>>CC=O' --config my-config.yaml
```
Or
```python
from flask_tools.pipette.config import PipetteConfig

config = PipetteConfig.from_yaml("my-config.yaml")
```

# Tests

`pytest`
Or
`pytest -m llm_query` to run the tests that use LLM

# ReactionDecoder / RDT

`pipette` now includes a Python wrapper around the Java-based ReactionDecoder Tool.
The Java helper entrypoint lives in this repo, so you do not need to maintain a fork of `ReactionDecoder`.

Build the fat jar:

```bash
./scripts/install_rdt.sh
export PIPETTE_RDT_JAR=/absolute/path/to/ReactionDecoder/target/rdt-4.0.0-jar-with-dependencies.jar
export PIPETTE_RDT_HELPER_BUILD_DIR=/absolute/path/to/flask-tools/flask_tools/pipette/_java_build
```

Use it from Python:

```python
from flask_tools.pipette.rdt import (
    map_reaction_smiles_with_rdt,
    map_reaction_smiles_list_with_rdt,
)

mapped = map_reaction_smiles_with_rdt("CC(=O)O.OCC>>CC(=O)OCC.O")
mapped_many = map_reaction_smiles_list_with_rdt(
    [
        "CC(=O)O.OCC>>CC(=O)OCC.O",
        "CCO>>CC=O",
    ]
)
```

Or from the CLI:

```bash
pipette-rdt --rxn-smi 'CC(=O)O.OCC>>CC(=O)OCC.O'
pipette-rdt --file reactions.txt --json
```

If the input uses `reactants>agents>products`, the wrapper strips agents for RDT, maps the core reaction, and then reinserts the original agents into the returned reaction SMILES.
