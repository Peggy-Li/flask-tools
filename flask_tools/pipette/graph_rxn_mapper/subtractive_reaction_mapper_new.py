import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from flask_tools.pipette import ToolResult
from flask_tools.pipette.config import PipetteConfig, ReasoningEffort
from flask_tools.pipette.smiles import (
    remove_atom_mapping_from_smiles,
    split_reaction_smiles,
)
from flask_tools.pipette.verifiers import ReactionChecker
from flask_tools.pipette.reaction_fixer import (
    AsyncLLMReactionFixer,
    BaseLLMReactionFixer,
)
from .foo import llm_benchmark_reactions
from .foo.llm_benchmark_reactions import mapped_reaction_from_pairs
from .subtractive_reaction_mapper_v3 import (
    subtractive_map_reaction,
    SubtractiveMappingResult,
)
from ..constants import (
    SmilesContainer,
    ToolResultDetails,
    ToolResultsDict,
    ToolStatus,
    resolve_llm_api_key,
)
from ..llm_query import _run_coroutine_sync, query_task
from ..pipeline import GradingPipeline

"""
Overall flow
    ```
        O.A > B > AA
           | rm reagent (New GraphBasedBalancer tool)
           V
        O.A >> AA
           | Algorithmic balancing + atom mapping  (New GraphBasedBalancer tool)
           V
        O.A[:1].A[:2] >> A[:1]A[:2] (New GraphBasedBalancer tool)
           | rm atom map
           V
        O.A.A >> AA
           | LLM balance (Existing LLMFixer tool)
           V
        O.A.A >> AA.O
           | Add back reagents (New LLMAtomMapper tool) (in case reagent label was incorrect)
           V
        O.A.A > B > AA.O
           | LLM atom mapping (New LLMAtomMapper tool)
           V
        O[:3].A[:1].A[:2] > B > A[:1]A[:2].O[:3]
    ```
"""


class GraphBasedBalancer(ReactionChecker):
    name = "graph_based_balancing"

    def __init__(self, config: PipetteConfig) -> None:
        self.config = config
        self.atom_map_config = config.tools_settings.reaction_mapper

    def run(
        self, rxn_smiles: str, context: ToolResultsDict | None = None
    ) -> ToolResult:
        """
        The subtractive_mapping is used for balancing dimerization.
        then LLM atom mapping
        Args:
            rxn_smiles:
            context:

        Returns:

        """
        context = context if context is not None else {}
        # Rm agents
        reactants_smi, agents_smi, products_smi = split_reaction_smiles(rxn_smiles)
        reactants_products_smi = reactants_smi + ">>" + products_smi
        # Balance
        res: SubtractiveMappingResult = subtractive_map_reaction(
            reactants_products_smi, config=self.atom_map_config
        )
        # Rm atom mapping
        graph_mapped_smi = res.atom_mapped_reaction_smiles()
        initial_balanced_smi = remove_atom_mapping_from_smiles(graph_mapped_smi)
        if initial_balanced_smi is None:
            return ToolResult(
                name=self.name,
                status=ToolStatus.ERROR,
                data=None,
                comment="Graph-based mapper produced an invalid atom-mapped reaction SMILES.",
            )

        # Call the llm balancing (llm fixing) already in pipette
        reaction_fixer = AsyncLLMReactionFixer.from_config(self.config)
        llm_balanced_tool_res = None
        llm_balanced_smi = None
        if reaction_fixer is not None:
            llm_balanced_smi: str | None
            llm_balanced_tool_res, llm_balanced_smi = (
                GradingPipeline.attempt_llm_fix_async(
                    reaction_fixer, initial_balanced_smi, context
                )
            )
            context[(initial_balanced_smi, BaseLLMReactionFixer.name)] = (
                llm_balanced_tool_res
            )

        if llm_balanced_smi is None:  # Error, no change, or balancer not enabled
            llm_balanced_smi = initial_balanced_smi

        new_reactants_smi, new_agents_smi, new_products_smi = split_reaction_smiles(
            llm_balanced_smi
        )
        if new_agents_smi:
            raise ValueError(
                f"LLM balancing unexpectedly produced an agent. Input {initial_balanced_smi}, output {llm_balanced_smi} "
            )

        balanced_smi = f"{new_reactants_smi}>{agents_smi}>{new_products_smi}"
        return ToolResult(
            name=self.name,
            status=ToolStatus.PASS,
            data=GraphBasedBalancerResultDetails(
                original_reaction_smiles=rxn_smiles,
                graph_mapped_reaction_smiles=graph_mapped_smi,
                graph_balanced_reaction_smiles=initial_balanced_smi,
                final_balanced_reaction_smiles=balanced_smi,
                objective_value=res.objective_value,
                mapper_status=res.status,
                reasoning_summary=(
                    "Subtractive graph mapping balanced the reactant/product sides, "
                    "then the LLM reaction fixer was used to optionally add missing species."
                ),
            ),
            comment="Graph-based balancing completed.",
        )


class GraphBasedBalancerResultDetails(ToolResultDetails):
    original_reaction_smiles: str
    graph_mapped_reaction_smiles: str
    graph_balanced_reaction_smiles: str
    final_balanced_reaction_smiles: str
    objective_value: float
    mapper_status: str
    reasoning_summary: str


class AtomMapping(BaseModel):
    product_atom: int
    reactant_atom: int


class ReactionMapping(BaseModel):
    product_to_reactant: list[AtomMapping]
    confidence: float
    reasoning_summary: str


class AtomMappingResultDetails(ToolResultDetails):
    input_reaction_smiles: str
    mapped_reaction_smiles: str
    product_to_reactant: list[AtomMapping]
    confidence: float
    reasoning_summary: str


class LLMAtomMapper(ReactionChecker):
    name = "llm_atom_mapping"

    def __init__(
        self,
        config: PipetteConfig,
        url: str,
        model: str,
        reasoning_effort: ReasoningEffort,
        api_key: str,
        user_prompt_path: Path,
        system_prompt_path: Path,
        skill_prompt_path: Path,
    ) -> None:
        self.config = config
        self.atom_map_config = config.tools_settings.reaction_mapper
        self.url = url
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.skill_prompt_path = skill_prompt_path

    @classmethod
    def from_config(cls, config: PipetteConfig) -> "LLMAtomMapper":
        atom_mapping_config = config.tools_settings.llm_atom_mapping
        api_key = resolve_llm_api_key(atom_mapping_config.api_key)
        if not api_key:
            raise ValueError(
                "LLM atom mapper requires an API key via "
                "tools_settings.llm_atom_mapping.api_key or the standard LLM env vars."
            )
        return cls(
            config=config,
            url=atom_mapping_config.url,
            model=atom_mapping_config.model,
            reasoning_effort=atom_mapping_config.reasoning_effort,
            api_key=api_key,
            user_prompt_path=atom_mapping_config.user_prompt_path,
            system_prompt_path=atom_mapping_config.system_prompt_path,
            skill_prompt_path=atom_mapping_config.skill_prompt_path,
        )

    def _build_user_payload(
        self,
        rxn_smiles: str,
        results: list[ToolResult],
    ) -> dict[str, Any]:
        serialized_results = [r.model_dump(exclude_none=True) for r in results]
        for s in serialized_results:
            if "skipped_reason" in s:
                del s["skipped_reason"]

        user_template = self.user_prompt_path.read_text(encoding="utf-8")
        reactants_smi, _agents_smi, products_smi = split_reaction_smiles(rxn_smiles)
        user_prompt = user_template.format(
            unmapped_reaction_smiles=rxn_smiles,
            reactant_graph_json=llm_benchmark_reactions.side_graph_json(reactants_smi),
            product_graph_json=llm_benchmark_reactions.side_graph_json(products_smi),
        )

        return {
            "reaction_smiles": rxn_smiles,
            "tool_results": serialized_results,
            "instructions": user_prompt,
        }

    def run(
        self, rxn_smiles: str | SmilesContainer, context: ToolResultsDict | None = None
    ) -> ToolResult:
        """
        LLM atom mapping
        Args:
            rxn_smiles:
            context:

        Returns:

        """
        context = context if context is not None else {}
        # Add reagents back in
        if isinstance(rxn_smiles, SmilesContainer):
            reactants, agents, products = split_reaction_smiles(rxn_smiles)
            if agents and agents != rxn_smiles.reagents_smi:
                raise ValueError(
                    f"How did these diverge?"
                )  # Just in case. Nothing in current code would do this
            agents = rxn_smiles.reagents_smi
            rxn_smiles = f"{reactants}>{agents}>{products}"

        # Call LLM atom mapper
        user_prompt = json.dumps(
            self._build_user_payload(rxn_smiles, list(context.values())),
            indent=2,
            sort_keys=True,
        )
        system_prompt = self.system_prompt_path.read_text(encoding="utf-8")
        skill_prompt = self.skill_prompt_path.read_text(encoding="utf-8")
        system_prompt = f"{system_prompt}\n\nAdditional atom-mapping skill instructions:\n{skill_prompt}"
        response_text = query_task(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=self.model,
            api_key=self.api_key,
            url=self.url,
            reasoning_effort=self.reasoning_effort,
            structured_output_schema=ReactionMapping,
            agent_name="PipetteAtomMapper",
        )
        return self._parse_output(rxn_smiles, response_text)

    def _parse_output(self, rxn_smiles: str, response_text: str) -> ToolResult:
        try:
            parsed = ReactionMapping.model_validate_json(response_text)
        except Exception as exc:
            raise ValueError(
                f"LLM atom mapper did not return valid JSON: {response_text}"
            ) from exc

        mapped_reaction_smiles = mapped_reaction_from_pairs(
            rxn_smiles,
            [
                (mapping.product_atom, mapping.reactant_atom)
                for mapping in parsed.product_to_reactant
            ],
            keep_agents=True,
        )
        return ToolResult(
            name=self.name,
            status=ToolStatus.PASS,
            data=AtomMappingResultDetails(
                input_reaction_smiles=rxn_smiles,
                mapped_reaction_smiles=mapped_reaction_smiles,
                product_to_reactant=parsed.product_to_reactant,
                confidence=parsed.confidence,
                reasoning_summary=parsed.reasoning_summary,
            ),
            comment="LLM atom mapping completed.",
        )
