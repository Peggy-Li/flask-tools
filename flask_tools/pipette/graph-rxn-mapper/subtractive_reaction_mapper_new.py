from flask_tools.pipette import ToolResult
from flask_tools.pipette.smiles import split_reaction_smiles
from flask_tools.pipette.verifiers import ReactionChecker


class MassConservationChecker(ReactionChecker):
    def __init__(self, config: PipetteConfig) -> None:
        self.config = config

    def run(
        self, rxn_smiles: str, context: dict[str, ToolResult] | None = None
    ) -> ToolResult:
        """
        ```
            O.A > B > AA
               | rm reagent
               V
            O.A >> AA
               | Algorithmic balancing + atom mapping
               V
            O.A[:1].A[:2] >> A[:1]A[:2]
               | rm atom map
               V
            O.A.A >> AA
               | LLM balance
               V
            O.A.A >> AA.O
               | Add back reagents (in case reagent label was incorrect)
               V
            O.A.A > B > AA.O
               | LLM atom mapping
               V
            O[:3].A[:1].A[:2] > B > A[:1]A[:2].O[:3]
        ```
        Args:
            rxn_smiles:
            context:

        Returns:

        """
        # Rm agents
        reactants_smi, agents_smi, products_smi = split_reaction_smiles(rxn_smiles)
        reactants_products_smi = reactants_smi + ">>" + products_smi
        # Balance
