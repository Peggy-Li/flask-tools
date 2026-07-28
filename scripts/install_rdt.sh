#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
DEFAULT_RDT_REPO="/usr/WS2/li54/flask/lib/ReactionDecoder"
DEFAULT_LOCAL_JAVA_MAJOR=11
DEFAULT_JAVA_WS="/usr/workspace/li54/java_ws"
DEFAULT_MAVEN_VERSION="3.6.3"
DEFAULT_MAVEN_HOME="${DEFAULT_JAVA_WS}/apache-maven-${DEFAULT_MAVEN_VERSION}"
DEFAULT_MAVEN_REPO="${DEFAULT_JAVA_WS}/.m2/repository"
DEFAULT_JAVA_CACERTS="/etc/pki/java/cacerts"

usage() {
    cat >&2 <<EOF
Usage: $0 [/path/to/ReactionDecoder]

Defaults:
  RDT repo: ${DEFAULT_RDT_REPO}

Overrides:
  RDT_REPO                Explicit ReactionDecoder repo path.
  RDT_JAVA_WS             Workspace directory for user-local Java tooling/cache.
  RDT_JAVA_HOME           Preferred Java home for the RDT build.
  RDT_MAVEN_HOME          Preferred Maven home for the RDT build.
  RDT_MAVEN_REPO          Maven local repository directory.
  RDT_JAVA_CACERTS        Java truststore path for Maven HTTPS downloads.
  RDT_MODULE_INIT         Optional module init script to source before loading modules.
  RDT_JAVA_MODULE         Optional Java module to load.
  RDT_MAVEN_MODULE        Optional Maven module to load.
  RDT_POM_FILE            Explicit pom file to build with.
  RDT_MAVEN_ARGS          Extra Maven arguments.
EOF
}

if [[ $# -gt 1 ]]; then
    usage
    exit 1
fi

RDT_REPO="${1:-${RDT_REPO:-${DEFAULT_RDT_REPO}}}"
RDT_JAVA_WS="${RDT_JAVA_WS:-${DEFAULT_JAVA_WS}}"
RDT_MAVEN_REPO="${RDT_MAVEN_REPO:-${DEFAULT_MAVEN_REPO}}"
RDT_JAVA_CACERTS="${RDT_JAVA_CACERTS:-${DEFAULT_JAVA_CACERTS}}"

prepend_path() {
    local dir=$1
    if [[ -d "${dir}" && ":${PATH}:" != *":${dir}:"* ]]; then
        PATH="${dir}:${PATH}"
    fi
}

java_major_from_bin() {
    local java_bin=$1
    local version_line
    version_line=$("${java_bin}" -version 2>&1 | head -n 1)
    if [[ "${version_line}" =~ version[[:space:]]+\"1\.([0-9]+)\. ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "${version_line}" =~ version[[:space:]]+\"([0-9]+)\. ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo 0
    fi
}

java_home_from_bin() {
    local java_bin=$1
    dirname "$(dirname "$(readlink -f "${java_bin}")")"
}

select_java_home() {
    local min_major=$1
    local best_major=0
    local best_home=""
    local candidate=""
    local candidate_bin=""
    local candidate_major=0
    local -a candidates=(
        "${RDT_JAVA_HOME:-}"
        "${JAVA_HOME:-}"
        "${RDT_JAVA_WS}/jdk-25"
        "${RDT_JAVA_WS}/jdk-25-openjdk"
        "${RDT_JAVA_WS}/java-25"
        "${RDT_JAVA_WS}/jdk-21"
        "${RDT_JAVA_WS}/jdk-21-openjdk"
        "${RDT_JAVA_WS}/java-21"
        "/usr/lib/jvm/java-25-openjdk"
        "/usr/lib/jvm/java-25"
        "/usr/lib/jvm/jdk-25"
        "/usr/lib/jvm/java-21-openjdk"
        "/usr/lib/jvm/java-21"
        "/usr/lib/jvm/java-17-openjdk"
        "/usr/lib/jvm/java-17"
        "/usr/lib/jvm/java-11-openjdk"
        "/usr/lib/jvm/java-11"
    )

    if command -v java >/dev/null 2>&1; then
        candidates+=("$(java_home_from_bin "$(command -v java)")")
    fi

    for candidate in "${candidates[@]}"; do
        [[ -n "${candidate}" ]] || continue
        candidate_bin="${candidate}/bin/java"
        if [[ ! -x "${candidate_bin}" || ! -x "${candidate}/bin/javac" ]]; then
            continue
        fi
        candidate_major=$(java_major_from_bin "${candidate_bin}")
        if (( candidate_major >= min_major && candidate_major > best_major )); then
            best_major=${candidate_major}
            best_home=${candidate}
        fi
    done

    if [[ -n "${best_home}" ]]; then
        echo "${best_home}"
    fi
}

configure_java_home() {
    local java_home=$1
    export JAVA_HOME="${java_home}"
    prepend_path "${JAVA_HOME}/bin"
    hash -r
}

select_maven_home() {
    local candidate=""
    local -a candidates=(
        "${RDT_MAVEN_HOME:-}"
        "${MAVEN_HOME:-}"
        "${DEFAULT_MAVEN_HOME}"
        "/usr/share/maven"
        "/opt/maven"
        "/usr/local/apache-maven"
    )

    for candidate in "${candidates[@]}"; do
        [[ -n "${candidate}" ]] || continue
        if [[ -x "${candidate}/bin/mvn" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    if command -v mvn >/dev/null 2>&1; then
        dirname "$(dirname "$(readlink -f "$(command -v mvn)")")"
    fi
}

configure_maven_home() {
    local maven_home=$1
    export MAVEN_HOME="${maven_home}"
    prepend_path "${MAVEN_HOME}/bin"
    hash -r
}

configure_java_truststore() {
    if [[ -f "${RDT_JAVA_CACERTS}" ]]; then
        local trust_opts="-Djavax.net.ssl.trustStore=${RDT_JAVA_CACERTS} -Djavax.net.ssl.trustStorePassword=changeit"
        if [[ -n "${MAVEN_OPTS:-}" ]]; then
            export MAVEN_OPTS="${trust_opts} ${MAVEN_OPTS}"
        else
            export MAVEN_OPTS="${trust_opts}"
        fi
    fi
}

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

SELECTED_JAVA_HOME=$(select_java_home 25 || true)
JAVA_MAJOR=0
POM_FILE="${RDT_POM_FILE:-}"
MAVEN_BUILD_ARGS=(-Dmaven.test.skip=true)

if [[ -n "${SELECTED_JAVA_HOME}" ]]; then
    configure_java_home "${SELECTED_JAVA_HOME}"
    JAVA_MAJOR=$(java_major_from_bin "${JAVA_HOME}/bin/java")
fi

if [[ -z "${POM_FILE}" ]]; then
    if (( JAVA_MAJOR >= 25 )); then
        POM_FILE="${RDT_REPO}/pom.xml"
        MAVEN_BUILD_ARGS=(-P local -Dmaven.test.skip=true)
    elif [[ -f "${RDT_REPO}/pom-local.xml" ]]; then
        SELECTED_JAVA_HOME=$(select_java_home "${DEFAULT_LOCAL_JAVA_MAJOR}" || true)
        if [[ -n "${SELECTED_JAVA_HOME}" ]]; then
            configure_java_home "${SELECTED_JAVA_HOME}"
            JAVA_MAJOR=$(java_major_from_bin "${JAVA_HOME}/bin/java")
            POM_FILE="${RDT_REPO}/pom-local.xml"
        fi
    else
        POM_FILE="${RDT_REPO}/pom.xml"
        MAVEN_BUILD_ARGS=(-P local -Dmaven.test.skip=true)
    fi
fi

if [[ -z "${POM_FILE}" ]]; then
    echo "No usable Java toolchain was found." >&2
    echo "This machine currently exposes JDKs up to 21 under /usr/lib/jvm." >&2
    echo "Provide Java 25 with RDT_JAVA_HOME/RDT_JAVA_MODULE, or use a repo that includes pom-local.xml." >&2
    exit 1
fi

SELECTED_MAVEN_HOME=$(select_maven_home || true)
if [[ -n "${SELECTED_MAVEN_HOME}" ]]; then
    configure_maven_home "${SELECTED_MAVEN_HOME}"
fi

configure_java_truststore

if ! command -v java >/dev/null 2>&1; then
    echo "java was not found on PATH after toolchain setup." >&2
    exit 1
fi

if ! command -v javac >/dev/null 2>&1; then
    echo "javac was not found on PATH after toolchain setup." >&2
    exit 1
fi

if ! command -v mvn >/dev/null 2>&1; then
    echo "mvn was not found on PATH after toolchain setup." >&2
    echo "Install Maven under ${DEFAULT_JAVA_WS} or set RDT_MAVEN_HOME/RDT_MAVEN_MODULE." >&2
    exit 1
fi

if [[ ! -d "${RDT_REPO}" ]]; then
    echo "RDT repo not found: ${RDT_REPO}" >&2
    exit 1
fi

if [[ ! -f "${POM_FILE}" ]]; then
    echo "POM file not found: ${POM_FILE}" >&2
    exit 1
fi

if [[ "${POM_FILE}" == "${RDT_REPO}/pom.xml" && ${JAVA_MAJOR} -lt 25 ]]; then
    echo "pom.xml requires Java 25, but the selected Java is ${JAVA_MAJOR}." >&2
    echo "Set RDT_JAVA_HOME or RDT_JAVA_MODULE to a JDK 25 installation, or unset RDT_POM_FILE to allow pom-local.xml fallback." >&2
    exit 1
fi

if [[ "${POM_FILE}" == "${RDT_REPO}/pom-local.xml" && ${JAVA_MAJOR} -lt ${DEFAULT_LOCAL_JAVA_MAJOR} ]]; then
    echo "pom-local.xml requires at least Java ${DEFAULT_LOCAL_JAVA_MAJOR}, but the selected Java is ${JAVA_MAJOR}." >&2
    exit 1
fi

mkdir -p "${RDT_MAVEN_REPO}"

echo "Building RDT from ${RDT_REPO}"
JAVA_VERSION_LINE=$(java -version 2>&1 | sed -n '1p')
MAVEN_VERSION_LINES=$(mvn -version 2>&1 | sed -n '1,2p')
echo "Using Java: $(command -v java)"
echo "${JAVA_VERSION_LINE}"
echo "Using Maven: $(command -v mvn)"
echo "${MAVEN_VERSION_LINES}"
echo "Using POM: ${POM_FILE}"
echo "Using Maven repo: ${RDT_MAVEN_REPO}"
if [[ -f "${RDT_JAVA_CACERTS}" ]]; then
    echo "Using Java truststore: ${RDT_JAVA_CACERTS}"
fi
(
    cd "${RDT_REPO}"
    mvn -Dmaven.repo.local="${RDT_MAVEN_REPO}" -f "${POM_FILE}" compile assembly:single "${MAVEN_BUILD_ARGS[@]}" ${RDT_MAVEN_ARGS:-}
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
