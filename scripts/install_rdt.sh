#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /path/to/ReactionDecoder" >&2
    exit 1
fi

RDT_REPO=$1

if [[ -n "${RDT_MODULE_INIT:-}" && -f "${RDT_MODULE_INIT}" ]]; then
    # Optional cluster hook, e.g. /etc/profile.d/modules.sh
    # shellcheck disable=SC1090
    source "${RDT_MODULE_INIT}"
fi

if command -v module >/dev/null 2>&1; then
    if [[ -n "${RDT_JAVA_MODULE:-}" ]]; then
        module load "${RDT_JAVA_MODULE}"
    fi
    if [[ -n "${RDT_MAVEN_MODULE:-}" ]]; then
        module load "${RDT_MAVEN_MODULE}"
    fi
fi

if ! command -v java >/dev/null 2>&1; then
    echo "java was not found on PATH." >&2
    exit 1
fi

if ! command -v mvn >/dev/null 2>&1; then
    echo "mvn was not found on PATH." >&2
    exit 1
fi

if ! command -v javac >/dev/null 2>&1; then
    echo "javac was not found on PATH." >&2
    exit 1
fi

if [[ ! -d "${RDT_REPO}" ]]; then
    echo "RDT repo not found: ${RDT_REPO}" >&2
    exit 1
fi

echo "Building RDT from ${RDT_REPO}"
(
    cd "${RDT_REPO}"
    mvn -P local package -DskipTests=true ${RDT_MAVEN_ARGS:-}
)

JAR_PATH=$(find "${RDT_REPO}/target" -maxdepth 1 -type f -name '*-jar-with-dependencies.jar' | sort | tail -n 1)
if [[ -z "${JAR_PATH}" ]]; then
    echo "Build completed but no fat jar was found under ${RDT_REPO}/target" >&2
    exit 1
fi

HELPER_SOURCE="${REPO_ROOT}/flask_tools/pipette/java/PipetteAtomMapperCli.java"
HELPER_BUILD_DIR="${PIPETTE_RDT_HELPER_BUILD_DIR:-${REPO_ROOT}/flask_tools/pipette/_java_build}"

mkdir -p "${HELPER_BUILD_DIR}"
javac -cp "${JAR_PATH}" -d "${HELPER_BUILD_DIR}" "${HELPER_SOURCE}"

echo
echo "RDT fat jar:"
echo "  ${JAR_PATH}"
echo "Local helper classes:"
echo "  ${HELPER_BUILD_DIR}"
echo
echo "Export this before using the Python wrapper on another machine:"
echo "  export PIPETTE_RDT_JAR=\"${JAR_PATH}\""
echo "  export PIPETTE_RDT_HELPER_BUILD_DIR=\"${HELPER_BUILD_DIR}\""
echo
echo "If your cluster needs custom Maven SSL or mirror flags, set them with:"
echo "  export RDT_MAVEN_ARGS='...'"
