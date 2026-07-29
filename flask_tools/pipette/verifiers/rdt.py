###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

"""
A python wrapper to call Reaction Decoder Tool (RDT) for atom mapping, through a java wrapper script.
Will compile against the RDT jar if the pipette java wrapper is not already compiled. See env vars.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from flask_tools.pipette.smiles import split_reaction_smiles

RDT_JAR_ENV_VAR = "PIPETTE_RDT_JAR"
RDT_REPO_ENV_VAR = "PIPETTE_RDT_REPO"
RDT_HELPER_BUILD_ENV_VAR = "PIPETTE_RDT_HELPER_BUILD_DIR"
RDT_JAVA_BIN_ENV_VAR = "PIPETTE_RDT_JAVA_BIN"
RDT_JAVAC_BIN_ENV_VAR = "PIPETTE_RDT_JAVAC_BIN"
RDT_MAIN_CLASS = "flask_tools.pipette.java.PipetteAtomMapperCli"


@dataclass(frozen=True)
class _PreparedReaction:
    # To remove and reintroduce agents during mapping
    original_smiles: str
    stripped_smiles: str
    agents_smiles: str


def _default_rdt_repo_path() -> Path:
    # Almost the same lvl as the flask_tools repo, under a lib folder. A really arbitrary default.
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root.parent / "lib" / "ReactionDecoder"


def _helper_source_path() -> Path:
    return (
        Path(__file__).resolve().parent.with_name("java") / "PipetteAtomMapperCli.java"
    )


def _helper_build_dir() -> Path:
    env_path = os.environ.get(RDT_HELPER_BUILD_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().with_name("_java_build")


def _default_java_bin() -> str:
    """Return the configured Java executable."""
    return os.environ.get(RDT_JAVA_BIN_ENV_VAR, "java")


def _default_javac_bin() -> str:
    """Return the configured javac executable."""
    return os.environ.get(RDT_JAVAC_BIN_ENV_VAR, "javac")


JAR_GLOB: str = "target/*-jar-with-dependencies.jar"


def _resolve_jar_from_repo(repo_path: Path) -> Path | None:
    jar_candidates = sorted(repo_path.glob(JAR_GLOB))
    if not jar_candidates:
        return None
    return jar_candidates[-1]


def resolve_rdt_jar_path(
    jar_path: str | Path | None = None,
    repo_path: str | Path | None = None,
) -> Path:
    if jar_path is not None:
        resolved = Path(jar_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"RDT jar does not exist: {resolved}")
        return resolved

    env_jar = os.environ.get(RDT_JAR_ENV_VAR)
    if env_jar:
        return resolve_rdt_jar_path(env_jar)

    candidate_repos: list[Path] = []
    if repo_path is not None:
        candidate_repos.append(Path(repo_path).expanduser().resolve())

    env_repo = os.environ.get(RDT_REPO_ENV_VAR)
    if env_repo:
        candidate_repos.append(Path(env_repo).expanduser().resolve())

    candidate_repos.append(_default_rdt_repo_path().resolve())

    for candidate_repo in candidate_repos:
        jar_candidate = _resolve_jar_from_repo(candidate_repo)
        if jar_candidate is not None:
            return jar_candidate.resolve()

    searched = ", ".join(str(path) for path in candidate_repos)
    raise FileNotFoundError(
        "Could not locate an RDT fat jar. Set "
        f"{RDT_JAR_ENV_VAR}, pass jar_path=..., or build one in a default location ({searched}), matching the glob string {JAR_GLOB}"
    )


def _prepare_reaction_smiles(reaction_smiles: str) -> _PreparedReaction:
    reactants, agents, products = split_reaction_smiles(reaction_smiles)
    return _PreparedReaction(
        original_smiles=reaction_smiles,
        stripped_smiles=f"{reactants}>>{products}",
        agents_smiles=agents,
    )


def _restore_agents(mapped_reaction_smiles: str, agents_smiles: str) -> str:
    if not agents_smiles:
        return mapped_reaction_smiles
    reactants, _agents, products = split_reaction_smiles(mapped_reaction_smiles)
    return f"{reactants}>{agents_smiles}>{products}"


def ensure_rdt_helper_compiled(
    *,
    jar_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    javac_bin: str | None = None,
    build_dir: str | Path | None = None,
) -> Path:
    """Compile the local Java helper against the selected RDT jar."""
    resolved_jar = resolve_rdt_jar_path(jar_path=jar_path, repo_path=repo_path)
    source_path = _helper_source_path()
    javac_bin = javac_bin or _default_javac_bin()
    output_dir = (
        Path(build_dir).expanduser().resolve()
        if build_dir is not None
        else _helper_build_dir()
    )
    class_path = (
        output_dir / "flask_tools" / "pipette" / "java" / "PipetteAtomMapperCli.class"
    )

    if (
        class_path.exists()
        and class_path.stat().st_mtime >= source_path.stat().st_mtime
    ):
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            javac_bin,
            "-cp",
            str(resolved_jar),
            "-d",
            str(output_dir),
            str(source_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        details = proc.stderr.strip() or proc.stdout.strip() or "no compiler output"
        raise RuntimeError(f"Failed to compile the local RDT helper: {details}")
    return output_dir


def map_reaction_smiles_list_with_rdt(
    reaction_smiles_list: Sequence[str],
    *,
    jar_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    java_bin: str | None = None,
    javac_bin: str | None = None,
) -> list[str]:
    """Map a batch of reaction SMILES strings with the RDT helper CLI."""
    if not reaction_smiles_list:
        return []

    prepared = [_prepare_reaction_smiles(smiles) for smiles in reaction_smiles_list]
    resolved_jar = resolve_rdt_jar_path(jar_path=jar_path, repo_path=repo_path)
    java_bin = java_bin or _default_java_bin()
    helper_build_dir = ensure_rdt_helper_compiled(
        jar_path=resolved_jar,
        javac_bin=javac_bin,
    )
    command = [
        java_bin,
        "-cp",
        os.pathsep.join([str(helper_build_dir), str(resolved_jar)]),
        RDT_MAIN_CLASS,
    ]
    proc = subprocess.run(
        command,
        input="\n".join(item.stripped_smiles for item in prepared) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )

    stderr = proc.stderr.strip()
    stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0:
        details = stderr or "no stderr output"
        raise RuntimeError(
            f"RDT batch process failed with exit code {proc.returncode}: {details}"
        )

    if len(stdout_lines) != len(prepared):
        raise RuntimeError(
            "RDT returned an unexpected number of records: "
            f"expected {len(prepared)}, got {len(stdout_lines)}. stderr={stderr!r}"
        )

    mapped_smiles_list: list[str] = []
    errors: list[str] = []
    for expected_index, (line, original) in enumerate(
        zip(stdout_lines, prepared, strict=True)
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"RDT returned invalid JSON on line {expected_index + 1}: {line}"
            ) from exc

        actual_index = record.get("index")
        if actual_index != expected_index:
            raise RuntimeError(
                "RDT returned out-of-order records: "
                f"expected index {expected_index}, got {actual_index}"
            )

        error = record.get("error")
        mapped_smiles = record.get("mapped_smiles")
        if error:
            errors.append(f"[{expected_index}] {original.original_smiles}: {error}")
            continue
        if not isinstance(mapped_smiles, str) or not mapped_smiles:
            errors.append(
                f"[{expected_index}] {original.original_smiles}: missing mapped_smiles"
            )
            continue

        mapped_smiles_list.append(
            _restore_agents(mapped_smiles, original.agents_smiles)
        )

    if errors:
        joined_errors = "\n".join(errors[:10])
        if len(errors) > 10:
            joined_errors += f"\n... and {len(errors) - 10} more errors"
        raise RuntimeError(f"RDT failed to map one or more reactions:\n{joined_errors}")

    return mapped_smiles_list


def map_reaction_smiles_with_rdt(
    reaction_smiles: str,
    *,
    jar_path: str | Path | None = None,
    repo_path: str | Path | None = None,
    java_bin: str | None = None,
    javac_bin: str | None = None,
) -> str:
    """Map one reaction SMILES string with the RDT helper CLI."""
    return map_reaction_smiles_list_with_rdt(
        [reaction_smiles],
        jar_path=jar_path,
        repo_path=repo_path,
        java_bin=java_bin,
        javac_bin=javac_bin,
    )[0]


def _load_reaction_smiles_file(path: str | Path) -> list[str]:
    file_path = Path(path).expanduser().resolve()
    return [
        line.strip()
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Map one or more reaction SMILES with RDT in a single Java batch."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--rxn-smi",
        nargs="+",
        dest="rxn_smi",
        help="One or more reaction SMILES strings.",
    )
    input_group.add_argument(
        "-f",
        "--file",
        dest="file",
        help="Text file containing one reaction SMILES per line.",
    )
    parser.add_argument(
        "--jar-path",
        help=f"Path to the RDT fat jar. Overrides {RDT_JAR_ENV_VAR}.",
    )
    parser.add_argument(
        "--repo-path",
        help=f"Path to the RDT repository. Used to resolve target/*-jar-with-dependencies.jar. Overrides {RDT_REPO_ENV_VAR}.",
    )
    parser.add_argument(
        "--java-bin",
        default=_default_java_bin(),
        help=f"Java executable to use. Defaults to {RDT_JAVA_BIN_ENV_VAR} or 'java'.",
    )
    parser.add_argument(
        "--javac-bin",
        default=_default_javac_bin(),
        help=f"javac executable to use for compiling the local helper. Defaults to {RDT_JAVAC_BIN_ENV_VAR} or 'javac'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output records instead of plain mapped SMILES lines.",
    )
    args = parser.parse_args()

    reaction_smiles_list = args.rxn_smi or _load_reaction_smiles_file(args.file)
    mapped_smiles_list = map_reaction_smiles_list_with_rdt(
        reaction_smiles_list,
        jar_path=args.jar_path,
        repo_path=args.repo_path,
        java_bin=args.java_bin,
        javac_bin=args.javac_bin,
    )

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "input_reaction_smiles": input_smiles,
                        "mapped_reaction_smiles": mapped_smiles,
                    }
                    for input_smiles, mapped_smiles in zip(
                        reaction_smiles_list, mapped_smiles_list, strict=True
                    )
                ],
                indent=2,
            )
        )
    else:
        for mapped_smiles in mapped_smiles_list:
            print(mapped_smiles)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
