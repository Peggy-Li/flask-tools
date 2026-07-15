###############################################################################
## Copyright 2025-2026 Lawrence Livermore National Security, LLC.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
###############################################################################

from __future__ import annotations

import asyncio
import os
import threading
from typing import TYPE_CHECKING, Awaitable, Literal, Any
from urllib.parse import urlsplit, urlunsplit

from charge.clients.agentframework import AgentFrameworkBackend
from charge.tasks.task import Task
from pydantic import BaseModel

if TYPE_CHECKING:
    from charge.clients.agentframework import AgentFrameworkAgent

BACKEND: AgentFrameworkBackend | None = None


def _normalize_base_url(url: str | None) -> str | None:
    if not url:
        return url

    parts = urlsplit(url)
    normalized_path = parts.path.rstrip("/")

    # Pipette configs historically stored a full endpoint path, while the
    # model clients expect the API base URL and append their own route.
    for suffix in ("/chat/completions", "/responses"):
        if normalized_path.endswith(suffix):
            normalized_path = normalized_path[: -len(suffix)] or "/"
            break

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            normalized_path,
            parts.query,
            parts.fragment,
        )
    )


def _backend_matches(
    backend: AgentFrameworkBackend | None,
    *,
    model: str | None,
    backend_name: str,
    url: str | None,
    reasoning_effort: Literal["low", "medium", "high"],
) -> bool:
    if backend is None:
        return False
    return (
        backend.model == model
        and backend.backend == backend_name
        and backend.base_url == url
        and backend.reasoning_effort == reasoning_effort
    )


def set_agent_backend(
    model: str | None = "gpt-5.4",
    api_key: str | None = None,
    backend: str = "livai",
    url: str | None = None,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
) -> None:
    global BACKEND
    resolved_api_key = os.getenv("FLASK_ORCHESTRATOR_API_KEY", api_key)
    resolved_model = os.getenv("FLASK_ORCHESTRATOR_MODEL", model)
    resolved_backend = os.getenv("FLASK_ORCHESTRATOR_BACKEND", backend)
    resolved_url = _normalize_base_url(os.getenv("FLASK_ORCHESTRATOR_URL", url))

    BACKEND = AgentFrameworkBackend(
        model=resolved_model,
        backend=resolved_backend,
        api_key=resolved_api_key,
        base_url=resolved_url,
        use_responses_api=True,
        reasoning_effort=reasoning_effort,
    )


def get_agentframework_backend(
    model: str | None = "gpt-5.4",
    api_key: str | None = None,
    backend: str = "livai",
    url: str | None = None,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
) -> AgentFrameworkBackend:
    resolved_model = os.getenv("FLASK_ORCHESTRATOR_MODEL", model)
    resolved_backend = os.getenv("FLASK_ORCHESTRATOR_BACKEND", backend)
    resolved_url = _normalize_base_url(os.getenv("FLASK_ORCHESTRATOR_URL", url))

    if not _backend_matches(
        BACKEND,
        model=resolved_model,
        backend_name=resolved_backend,
        url=resolved_url,
        reasoning_effort=reasoning_effort,
    ):
        set_agent_backend(
            model=model,
            api_key=api_key,
            backend=backend,
            url=url,
            reasoning_effort=reasoning_effort,
        )
    return BACKEND  # noqa


async def query_task_async(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    url: str | None = None,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    structured_output_schema: type[BaseModel] | None = None,
    agent_name: str = "Pipette",
    max_retries: int = 1,
    max_tool_calls: int = 1,
) -> str:
    task = Task(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured_output_schema=structured_output_schema,
    )
    backend = get_agentframework_backend(
        model=model,
        api_key=api_key,
        url=url,
        reasoning_effort=reasoning_effort,
    )

    agent: AgentFrameworkAgent = backend.create_agent(
        task=task,
        agent_name=agent_name,
        max_retries=max_retries,
        max_tool_calls=max_tool_calls,
    )
    result = await agent.run()
    return str(result)


def _run_coroutine_sync(coro: Awaitable) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, str] = {}
    error: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - only hit inside active loop.
            error["value"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def query_task(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_key: str,
    url: str | None = None,
    reasoning_effort: Literal["low", "medium", "high"] = "medium",
    structured_output_schema: type[BaseModel] | None = None,
    agent_name: str = "Pipette",
    max_retries: int = 1,
    max_tool_calls: int = 1,
) -> str:
    return _run_coroutine_sync(
        query_task_async(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            api_key=api_key,
            url=url,
            reasoning_effort=reasoning_effort,
            structured_output_schema=structured_output_schema,
            agent_name=agent_name,
            max_retries=max_retries,
            max_tool_calls=max_tool_calls,
        )
    )
