###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml

from .constants import DEFAULT_LLM_BASE_URL, resolve_llm_base_url
from .graph_rxn_mapper.subtractive_reaction_mapper_v3 import ReactionAtomMapperConfig

ReasoningEffort = Literal["low", "medium", "high"]


def package_data_path(filename: str) -> Path:
    return Path(__file__).with_name("data") / filename


def package_config_path(filename: str) -> Path:
    return Path(__file__).with_name("assets") / filename


def graph_rxn_mapper_prompt_path(filename: str) -> Path:
    # Relative to top level of pipette module / graph_rxn_mapper / prmopts
    return Path(__file__).with_name("graph_rxn_mapper") / "prompts" / filename


def _validate_mapping_format(data: object, *, name: str) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a mapping.")
    return data


def _resolve_optional_path(path_value: object, *, base_dir: Path) -> Path | None:
    if path_value is None:
        return None
    if not isinstance(path_value, str):
        raise ValueError("Path values in PipetteConfig YAML must be strings.")

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _resolve_defaultable_cwd_path(
    path_value: object,
    *,
    default_path: Path,
    name: str,
) -> Path:
    if path_value is None or path_value == "default":
        return default_path
    if not isinstance(path_value, str):
        raise ValueError(f"{name} must be a string, 'default', or null.")

    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


@dataclass
class PipelineConfig:
    stop_on_hard_fail: bool = True
    mass_tolerance_atoms: int = 0
    reaction_energy_max_ev_mol: float = (
        0.5  # Permissive. Most rxns are below 0, but some that need heating up can be positive
    )
    use_dft: bool = False
    use_fixing: bool = True

    # All the from_mapping() is annoying, could this be better?
    @classmethod
    def from_mapping(cls, data: object) -> PipelineConfig:
        mapping = _validate_mapping_format(data, name="settings")
        use_fixing = mapping.get("use_fixing", cls.use_fixing)
        if not isinstance(use_fixing, bool):
            raise ValueError("settings.use_fixing must be a boolean.")
        return cls(
            stop_on_hard_fail=mapping.get("stop_on_hard_fail", cls.stop_on_hard_fail),
            mass_tolerance_atoms=mapping.get(
                "mass_tolerance_atoms", cls.mass_tolerance_atoms
            ),
            reaction_energy_max_ev_mol=mapping.get(
                "reaction_energy_max_ev_mol", cls.reaction_energy_max_ev_mol
            ),
            use_dft=mapping.get("use_dft", cls.use_dft),
            use_fixing=use_fixing,
        )


@dataclass
class LLMConfig:
    url: str = DEFAULT_LLM_BASE_URL
    model: str = "gpt-5.4"
    reasoning_effort: ReasoningEffort = "medium"
    api_key: str | None = None
    prompt_path: Path = field(
        default_factory=lambda: package_config_path("judge-prompt.txt")
    )
    prompt: str | None = None

    @classmethod
    def _llm_kwargs_from_mapping(
        cls,
        mapping: dict[str, Any],
        *,
        name: str,
        base_dir: Path,
        default_prompt_filename: str,
    ) -> dict[str, Any]:
        url = mapping.get("url")
        if url is not None and not isinstance(url, str):
            raise ValueError(f"{name}.url must be a string when provided.")

        model = mapping.get("model", cls.model)
        if not isinstance(model, str):
            raise ValueError(f"{name}.model must be a string.")

        reasoning_effort = mapping.get("reasoning_effort", cls.reasoning_effort)
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                f"{name}.reasoning_effort must be 'low', 'medium', or 'high'."
            )

        api_key = mapping.get("api_key")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError(f"{name}.api_key must be a string when provided.")

        prompt = mapping.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"{name}.prompt must be a string when provided.")

        prompt_path = _resolve_optional_path(
            mapping.get("prompt_path"), base_dir=base_dir
        )

        return {
            "url": resolve_llm_base_url(url),
            "model": model,
            "reasoning_effort": reasoning_effort,
            "api_key": api_key,
            "prompt_path": prompt_path or package_config_path(default_prompt_filename),
            "prompt": prompt,
        }


@dataclass
class LLMJudgeConfig(LLMConfig):
    allow_fail: Literal["all"] | list[str] = field(default_factory=list)
    enable_atom_mapping_dict_in_prompt: bool = False
    prompt_path: Path = field(
        default_factory=lambda: package_config_path("judge-prompt.txt")
    )

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        base_dir: Path,
    ) -> LLMJudgeConfig:
        mapping = _validate_mapping_format(data, name="llm_judge")
        allow_fail = mapping.get("allow_fail", [])
        if allow_fail != "all":
            if not isinstance(allow_fail, list) or not all(
                isinstance(name, str) for name in allow_fail
            ):
                raise ValueError(
                    "llm_judge.allow_fail must be 'all' or a list of tool names."
                )
        enable_atom_mapping_dict_in_prompt = mapping.get(
            "enable_atom_mapping_dict_in_prompt",
            cls.enable_atom_mapping_dict_in_prompt,
        )
        if not isinstance(enable_atom_mapping_dict_in_prompt, bool):
            raise ValueError(
                "llm_judge.enable_atom_mapping_dict_in_prompt must be a boolean."
            )
        return cls(
            allow_fail=allow_fail if allow_fail == "all" else list(allow_fail),
            enable_atom_mapping_dict_in_prompt=enable_atom_mapping_dict_in_prompt,
            **cls._llm_kwargs_from_mapping(
                mapping,
                name="llm_judge",
                base_dir=base_dir,
                default_prompt_filename="judge-prompt.txt",
            ),
        )


@dataclass
class LLMReactionFixerConfig(LLMConfig):
    enabled: bool = True
    prompt_path: Path = field(
        default_factory=lambda: package_config_path("fixer-prompt.txt")
    )

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        base_dir: Path,
    ) -> LLMReactionFixerConfig:
        mapping = _validate_mapping_format(data, name="llm_reaction_fixer")
        enabled = mapping.get("enabled", cls.enabled)
        if not isinstance(enabled, bool):
            raise ValueError("llm_reaction_fixer.enabled must be a boolean.")

        return cls(
            enabled=enabled,
            **cls._llm_kwargs_from_mapping(
                mapping,
                name="llm_reaction_fixer",
                base_dir=base_dir,
                default_prompt_filename="fixer-prompt.txt",
            ),
        )


@dataclass
class ReactionEnergyConfig:
    database: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        base_dir: Path,
    ) -> ReactionEnergyConfig:
        mapping = _validate_mapping_format(data, name="tools_settings.reaction_energy")
        return cls(
            database=_resolve_optional_path(mapping.get("database"), base_dir=base_dir),
        )


@dataclass
class LLMAtomMappingConfig:
    url: str = DEFAULT_LLM_BASE_URL
    model: str = "gpt-5.4"
    reasoning_effort: ReasoningEffort = "medium"
    api_key: str | None = None
    system_prompt_path: Path = field(
        default_factory=lambda: graph_rxn_mapper_prompt_path("atom_mapping_system.md")
    )
    user_prompt_path: Path = field(
        default_factory=lambda: graph_rxn_mapper_prompt_path("atom_mapping_user.md")
    )
    skill_prompt_path: Path = field(
        default_factory=lambda: graph_rxn_mapper_prompt_path("atom_mapping_skill.md")
    )

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        base_dir: Path,
    ) -> LLMAtomMappingConfig:
        mapping = _validate_mapping_format(data, name="tools_settings.llm_atom_mapping")
        del base_dir

        url = mapping.get("url")
        if url is not None and not isinstance(url, str):
            raise ValueError(
                "tools_settings.llm_atom_mapping.url must be a string when provided."
            )

        model = mapping.get("model", cls.model)
        if not isinstance(model, str):
            raise ValueError("tools_settings.llm_atom_mapping.model must be a string.")

        reasoning_effort = mapping.get("reasoning_effort", cls.reasoning_effort)
        if reasoning_effort not in {"low", "medium", "high"}:
            raise ValueError(
                "tools_settings.llm_atom_mapping.reasoning_effort must be 'low', 'medium', or 'high'."
            )

        api_key = mapping.get("api_key")
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError(
                "tools_settings.llm_atom_mapping.api_key must be a string when provided."
            )

        return cls(
            url=resolve_llm_base_url(url),
            model=model,
            reasoning_effort=reasoning_effort,
            api_key=api_key,
            system_prompt_path=_resolve_defaultable_cwd_path(
                mapping.get("system_prompt_path"),
                default_path=graph_rxn_mapper_prompt_path("atom_mapping_system.md"),
                name="tools_settings.llm_atom_mapping.system_prompt_path",
            ),
            user_prompt_path=_resolve_defaultable_cwd_path(
                mapping.get("user_prompt_path"),
                default_path=graph_rxn_mapper_prompt_path("atom_mapping_user.md"),
                name="tools_settings.llm_atom_mapping.user_prompt_path",
            ),
            skill_prompt_path=_resolve_defaultable_cwd_path(
                mapping.get("skill_prompt_path"),
                default_path=graph_rxn_mapper_prompt_path("atom_mapping_skill.md"),
                name="tools_settings.llm_atom_mapping.skill_prompt_path",
            ),
        )


@dataclass
class ToolsConfig:
    reaction_energy: ReactionEnergyConfig = field(default_factory=ReactionEnergyConfig)
    reaction_mapper: ReactionAtomMapperConfig = field(
        default_factory=ReactionAtomMapperConfig
    )
    llm_atom_mapping: LLMAtomMappingConfig = field(default_factory=LLMAtomMappingConfig)

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        base_dir: Path,
    ) -> ToolsConfig:
        mapping = _validate_mapping_format(data, name="tools_settings")
        return cls(
            reaction_energy=ReactionEnergyConfig.from_mapping(
                mapping.get("reaction_energy"),
                base_dir=base_dir,
            ),
            llm_atom_mapping=LLMAtomMappingConfig.from_mapping(
                mapping.get("llm_atom_mapping"),
                base_dir=base_dir,
            ),
        )


@dataclass
class PipetteConfig:
    mode: str = "exact"
    tool_list: str | list[str] | None = "all"
    llm_judge: LLMJudgeConfig = field(default_factory=LLMJudgeConfig)
    llm_reaction_fixer: LLMReactionFixerConfig = field(
        default_factory=LLMReactionFixerConfig
    )
    settings: PipelineConfig = field(default_factory=PipelineConfig)
    tools_settings: ToolsConfig = field(default_factory=ToolsConfig)
    solvent_catalog_path: Path = field(
        default_factory=lambda: package_data_path("solvents.tsv")
    )

    @classmethod
    def from_mapping(
        cls,
        data: object,
        *,
        base_dir: Path | None = None,
    ) -> PipetteConfig:
        mapping = _validate_mapping_format(data, name="PipetteConfig")
        resolved_base_dir = base_dir or Path.cwd()

        tool_list = mapping.get("tool_list", "all")
        if tool_list is None:
            tool_list = []
        elif tool_list != "all":
            if not isinstance(tool_list, list) or not all(
                isinstance(name, str) for name in tool_list
            ):
                raise ValueError(
                    "tool_list must be None, 'all', or a list of tool names."
                )
            tool_list = list(tool_list)

        solvent_catalog_path = _resolve_optional_path(
            mapping.get("solvent_catalog_path"),
            base_dir=resolved_base_dir,
        )
        pipeline_settings = mapping.get("settings", mapping.get("rules"))

        return cls(
            mode=mapping.get("mode", "exact"),
            tool_list=tool_list,
            llm_judge=LLMJudgeConfig.from_mapping(
                mapping.get("llm_judge"),
                base_dir=resolved_base_dir,
            ),
            llm_reaction_fixer=LLMReactionFixerConfig.from_mapping(
                mapping.get("llm_reaction_fixer"),
                base_dir=resolved_base_dir,
            ),
            settings=PipelineConfig.from_mapping(pipeline_settings),
            tools_settings=ToolsConfig.from_mapping(
                mapping.get("tools_settings"),
                base_dir=resolved_base_dir,
            ),
            solvent_catalog_path=solvent_catalog_path
            or package_data_path("solvents.tsv"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipetteConfig:
        config_path = Path(path).expanduser().resolve()
        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls.from_mapping(data, base_dir=config_path.parent)

    @classmethod
    def ai_default_path(cls) -> Path:
        return package_config_path("ai_judge_with_dft.yaml")

    @classmethod
    def from_default_ai_yaml(cls) -> PipetteConfig:
        return cls.from_yaml(cls.ai_default_path())


def load_config(config_arg: str) -> PipetteConfig:
    if config_arg == ConfigType.LLM_JUDGE_WITH_DFT:
        return PipetteConfig.from_default_ai_yaml()
    elif config_arg == ConfigType.LLM_JUDGE_NO_DFT:
        return PipetteConfig.from_yaml(package_config_path("ai_judge_no_dft.yaml"))
    return PipetteConfig.from_yaml(config_arg)


class ConfigType(StrEnum):
    LLM_JUDGE_WITH_DFT = "llm-judge-with-dft"
    LLM_JUDGE_NO_DFT = "llm-judge"
