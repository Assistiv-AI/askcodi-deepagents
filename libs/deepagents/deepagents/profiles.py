"""Profile-based deep agent construction.

This module adds an additive profile layer on top of ``create_deep_agent`` so
users can choose capability presets without changing the default constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
    TodoListMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import SystemMessage

from deepagents.backends import StateBackend
from deepagents.graph import (
    BASE_AGENT_PROMPT,
    create_deep_agent,
    get_default_model,
)
from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.read_skill import create_read_skill_tool
from deepagents.middleware.web_fetch import create_web_fetch_tool
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    GENERAL_PURPOSE_SUBAGENT,
    CompiledSubAgent,
    SubAgent,
    SubAgentMiddleware,
)
from deepagents.middleware.summarization import (
    _compute_summarization_defaults,
    _DeepAgentsSummarizationMiddleware,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain.agents.middleware.types import AgentMiddleware
    from langchain.agents.structured_output import ResponseFormat
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.cache.base import BaseCache
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer

    from deepagents.backends.protocol import BackendFactory, BackendProtocol

DeepAgentProfile = Literal[
    "full_agent",
    "external_tools_only",
    "no_shell",
    "no_filesystem",
    "local_subagent",
]
"""Supported profile names for ``create_deep_agent_with_profile``."""

LOCAL_SUBAGENT_NAME = "local-subagent"
LOCAL_SUBAGENT_DESCRIPTION = (
    "Use this subagent for internal workspace memory management and local file "
    "operations. It is optimized for creating, reading, and editing persistent "
    "agent notes and artifacts."
)
LOCAL_SUBAGENT_PROMPT = (
    "You are a local workspace memory manager. Use filesystem tools to maintain "
    "agent memory and artifacts. Prefer concise, deterministic updates and return "
    "a short report of what changed."
)


@dataclass(frozen=True)
class DeepAgentProfileOptions:
    """Capability options for profile-based agent construction.

    Attributes:
        include_main_filesystem: Whether the main orchestrator gets filesystem tools.
        include_main_execute: Whether the main orchestrator gets ``execute``.
        include_subagent_filesystem: Whether default subagents get filesystem tools.
        include_subagent_execute: Whether default subagents get ``execute``.
        include_subagents: Whether the main orchestrator gets the ``task`` tool.
        include_general_purpose_subagent: Whether to include the built-in
            ``general-purpose`` subagent.
        include_local_subagent: Whether to include a fixed internal
            ``local-subagent`` with filesystem capabilities.
        include_todo_list: Whether to include ``TodoListMiddleware``.
        include_web_fetch: Whether to include the ``web_fetch`` tool.
    """

    include_main_filesystem: bool = True
    include_main_execute: bool = True
    include_subagent_filesystem: bool = True
    include_subagent_execute: bool = True
    include_subagents: bool = True
    include_general_purpose_subagent: bool = True
    include_local_subagent: bool = False
    expose_main_filesystem_tools: bool = True
    include_todo_list: bool = True
    include_web_fetch: bool = True


PROFILE_OPTIONS: dict[DeepAgentProfile, DeepAgentProfileOptions] = {
    "full_agent": DeepAgentProfileOptions(),
    "external_tools_only": DeepAgentProfileOptions(
        include_main_filesystem=False,
        include_main_execute=False,
        include_subagent_filesystem=False,
        include_subagent_execute=False,
        include_todo_list=False,
    ),
    "no_shell": DeepAgentProfileOptions(
        include_main_filesystem=True,
        include_main_execute=False,
        include_subagent_filesystem=True,
        include_subagent_execute=False,
    ),
    "no_filesystem": DeepAgentProfileOptions(
        include_main_filesystem=False,
        include_main_execute=False,
        include_subagent_filesystem=False,
        include_subagent_execute=False,
        include_todo_list=False,
    ),
    "local_subagent": DeepAgentProfileOptions(
        include_main_filesystem=True,
        include_main_execute=False,
        include_subagent_filesystem=False,
        include_subagent_execute=False,
        include_local_subagent=True,
        expose_main_filesystem_tools=False,
    ),
}
"""Default capability settings for each built-in profile."""


def _resolve_model(model: str | BaseChatModel | None) -> BaseChatModel:
    """Resolve a model input into a concrete ``BaseChatModel`` instance."""
    if model is None:
        return get_default_model()
    if isinstance(model, str):
        return init_chat_model(model)
    return model


def _create_filesystem_middleware(
    *,
    backend: BackendProtocol | BackendFactory,
    include_execute: bool,
) -> FilesystemMiddleware:
    """Create filesystem middleware with optional execute tool removal."""
    middleware = FilesystemMiddleware(backend=backend)
    if include_execute:
        return middleware

    filtered_tools = [tool for tool in middleware.tools if tool.name != "execute"]
    middleware.tools = filtered_tools
    return middleware


def _hide_filesystem_tools(middleware: FilesystemMiddleware) -> FilesystemMiddleware:
    """Hide all filesystem tools while keeping filesystem state available.

    This is used for profiles like ``local_subagent`` where orchestrator state must
    still carry files written by internal workers, but the main model must not see
    filesystem tools directly.
    """
    middleware.tools = []
    middleware._custom_system_prompt = ""
    return middleware


def _build_subagent_middleware(
    *,
    model: BaseChatModel,
    backend: BackendProtocol | BackendFactory,
    include_filesystem: bool,
    include_execute: bool,
    skills_sources: list[str] | None,
) -> list[AgentMiddleware]:
    """Build default middleware stack for profile-managed subagents."""
    summarization_defaults = _compute_summarization_defaults(model)
    middleware: list[AgentMiddleware] = [TodoListMiddleware()]

    if include_filesystem:
        middleware.append(
            _create_filesystem_middleware(
                backend=backend,
                include_execute=include_execute,
            )
        )

    middleware.extend(
        [
            _DeepAgentsSummarizationMiddleware(
                model=model,
                backend=backend,
                trigger=summarization_defaults["trigger"],
                keep=summarization_defaults["keep"],
                trim_tokens_to_summarize=None,
                truncate_args_settings=summarization_defaults[
                    "truncate_args_settings"
                ],
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )

    if skills_sources is not None:
        middleware.append(SkillsMiddleware(backend=backend, sources=skills_sources))

    return middleware


def _build_local_subagent(
    *,
    model: BaseChatModel,
    backend: BackendProtocol | BackendFactory,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None,
) -> SubAgent:
    """Build the fixed local subagent used by the ``local_subagent`` profile."""
    local_middleware = _build_subagent_middleware(
        model=model,
        backend=backend,
        include_filesystem=True,
        include_execute=True,
        skills_sources=None,
    )
    if interrupt_on is not None:
        local_middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    return {
        "name": LOCAL_SUBAGENT_NAME,
        "description": LOCAL_SUBAGENT_DESCRIPTION,
        "system_prompt": LOCAL_SUBAGENT_PROMPT,
        "model": model,
        "tools": [],
        "middleware": local_middleware,
    }


def _process_subagents_for_profile(
    *,
    model: BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    subagents: list[SubAgent | CompiledSubAgent] | None,
    backend: BackendProtocol | BackendFactory,
    skills: list[str] | None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None,
    profile_options: DeepAgentProfileOptions,
) -> list[SubAgent | CompiledSubAgent]:
    """Build profile-managed subagent list for ``SubAgentMiddleware``."""
    processed_subagents: list[SubAgent | CompiledSubAgent] = []

    if profile_options.include_general_purpose_subagent:
        general_purpose_middleware = _build_subagent_middleware(
            model=model,
            backend=backend,
            include_filesystem=profile_options.include_subagent_filesystem,
            include_execute=profile_options.include_subagent_execute,
            skills_sources=skills,
        )
        if interrupt_on is not None:
            general_purpose_middleware.append(
                HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
            )

        processed_subagents.append(
            {
                **GENERAL_PURPOSE_SUBAGENT,
                "model": model,
                "tools": tools or [],
                "middleware": general_purpose_middleware,
            }
        )

    for spec in subagents or []:
        if "runnable" in spec:
            processed_subagents.append(spec)
            continue

        subagent_model: str | BaseChatModel = spec.get("model", model)
        if isinstance(subagent_model, str):
            subagent_model = init_chat_model(subagent_model)

        subagent_middleware = _build_subagent_middleware(
            model=subagent_model,
            backend=backend,
            include_filesystem=profile_options.include_subagent_filesystem,
            include_execute=profile_options.include_subagent_execute,
            skills_sources=spec.get("skills"),
        )
        subagent_middleware.extend(spec.get("middleware", []))

        processed_subagents.append(
            {
                **spec,
                "model": subagent_model,
                "tools": spec.get("tools", tools or []),
                "middleware": subagent_middleware,
            }
        )

    if profile_options.include_local_subagent:
        local_subagent = _build_local_subagent(
            model=model,
            backend=backend,
            interrupt_on=interrupt_on,
        )
        processed_subagents.append(local_subagent)

    return processed_subagents


def _compose_system_prompt(
    system_prompt: str | SystemMessage | None,
) -> str | SystemMessage:
    """Compose user-provided system prompt with Deep Agents base prompt."""
    if system_prompt is None:
        return BASE_AGENT_PROMPT

    if isinstance(system_prompt, SystemMessage):
        return SystemMessage(
            content=[
                *system_prompt.content_blocks,
                {"type": "text", "text": f"\n\n{BASE_AGENT_PROMPT}"},
            ]
        )

    return system_prompt + "\n\n" + BASE_AGENT_PROMPT


def _append_read_skill_tool_if_needed(
    *,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    skills: list[str] | None,
    backend: BackendProtocol | BackendFactory,
    profile_options: DeepAgentProfileOptions,
) -> list[BaseTool | Callable | dict[str, Any]] | None:
    """Add read_skill for skills-enabled profiles without main filesystem tools."""
    if not skills:
        return list(tools) if tools is not None else None

    # If main filesystem tools are exposed, read_file already exists.
    has_main_read_file = (
        profile_options.include_main_filesystem
        and profile_options.expose_main_filesystem_tools
    )
    if has_main_read_file:
        return list(tools) if tools is not None else None

    resolved_tools = list(tools) if tools is not None else []
    tool_names: set[str] = set()
    for t in resolved_tools:
        if hasattr(t, "name"):
            name = t.name
            if isinstance(name, str):
                tool_names.add(name)
            continue
        if isinstance(t, dict):
            name = t.get("name")
            if isinstance(name, str):
                tool_names.add(name)
    if "read_skill" in tool_names:
        return resolved_tools

    resolved_tools.append(
        create_read_skill_tool(
            backend=backend,
            allowed_prefixes=skills,
        )
    )
    return resolved_tools


def _append_web_fetch_tool_if_needed(
    *,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None,
    profile_options: DeepAgentProfileOptions,
    allowed_domains: Sequence[str] | None = None,
) -> list[BaseTool | Callable | dict[str, Any]] | None:
    """Add ``web_fetch`` tool when the profile enables it."""
    if not profile_options.include_web_fetch:
        return list(tools) if tools is not None else None

    resolved_tools = list(tools) if tools is not None else []
    tool_names: set[str] = set()
    for t in resolved_tools:
        if hasattr(t, "name"):
            name = t.name
            if isinstance(name, str):
                tool_names.add(name)
            continue
        if isinstance(t, dict):
            name = t.get("name")
            if isinstance(name, str):
                tool_names.add(name)
    if "web_fetch" in tool_names:
        return resolved_tools

    resolved_tools.append(create_web_fetch_tool(allowed_domains=allowed_domains))
    return resolved_tools


def create_deep_agent_with_profile(
    *,
    profile: DeepAgentProfile = "full_agent",
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: list[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    response_format: ResponseFormat | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
    profile_options: DeepAgentProfileOptions | None = None,
    allowed_domains: Sequence[str] | None = None,
) -> CompiledStateGraph:
    """Create a deep agent using capability profiles.

    This function is additive and does not change ``create_deep_agent`` behavior.

    Args:
        profile: Built-in profile name.
        model: The model to use.
        tools: External tools attached to main and profile-managed subagents.
        system_prompt: Optional custom system prompt.
        middleware: Extra middleware appended after profile middleware.
        subagents: Optional additional subagent specifications.
        skills: Optional skill source paths.
        memory: Optional memory paths.
        response_format: Structured output format.
        context_schema: Optional context schema.
        checkpointer: Optional graph checkpointer.
        store: Optional persistent store.
        backend: Optional backend instance or factory.
        interrupt_on: Optional tool interrupt mapping.
        debug: Whether debug mode is enabled.
        name: Optional graph name.
        cache: Optional graph cache.
        profile_options: Optional profile override options.

    Returns:
        A configured deep agent graph.
    """
    if profile_options is None:
        profile_options = PROFILE_OPTIONS[profile]

    resolved_backend: BackendProtocol | BackendFactory = (
        backend if backend is not None else (lambda rt: StateBackend(rt))
    )
    resolved_tools = _append_read_skill_tool_if_needed(
        tools=tools,
        skills=skills,
        backend=resolved_backend,
        profile_options=profile_options,
    )
    resolved_tools = _append_web_fetch_tool_if_needed(
        tools=resolved_tools,
        profile_options=profile_options,
        allowed_domains=allowed_domains,
    )

    if profile == "full_agent" and profile_options == PROFILE_OPTIONS["full_agent"]:
        return create_deep_agent(
            model=model,
            tools=resolved_tools,
            system_prompt=system_prompt,
            middleware=middleware,
            subagents=subagents,
            skills=skills,
            memory=memory,
            response_format=response_format,
            context_schema=context_schema,
            checkpointer=checkpointer,
            store=store,
            backend=backend,
            interrupt_on=interrupt_on,
            debug=debug,
            name=name,
            cache=cache,
        )

    resolved_model = _resolve_model(model)

    profile_subagents = _process_subagents_for_profile(
        model=resolved_model,
        tools=resolved_tools,
        subagents=subagents,
        backend=resolved_backend,
        skills=skills,
        interrupt_on=interrupt_on,
        profile_options=profile_options,
    )

    summarization_defaults = _compute_summarization_defaults(resolved_model)
    deepagent_middleware: list[AgentMiddleware] = []
    if profile_options.include_todo_list:
        deepagent_middleware.append(TodoListMiddleware())

    if memory is not None:
        deepagent_middleware.append(
            MemoryMiddleware(backend=resolved_backend, sources=memory)
        )

    if skills is not None:
        deepagent_middleware.append(
            SkillsMiddleware(backend=resolved_backend, sources=skills)
        )

    if profile_options.include_main_filesystem:
        main_filesystem = _create_filesystem_middleware(
            backend=resolved_backend,
            include_execute=profile_options.include_main_execute,
        )
        if not profile_options.expose_main_filesystem_tools:
            main_filesystem = _hide_filesystem_tools(main_filesystem)

        deepagent_middleware.append(main_filesystem)

    if profile_options.include_subagents:
        deepagent_middleware.append(
            SubAgentMiddleware(
                backend=resolved_backend,
                subagents=profile_subagents,
            )
        )

    deepagent_middleware.extend(
        [
            _DeepAgentsSummarizationMiddleware(
                model=resolved_model,
                backend=resolved_backend,
                trigger=summarization_defaults["trigger"],
                keep=summarization_defaults["keep"],
                trim_tokens_to_summarize=None,
                truncate_args_settings=summarization_defaults[
                    "truncate_args_settings"
                ],
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]
    )

    if middleware:
        deepagent_middleware.extend(middleware)

    if interrupt_on is not None:
        deepagent_middleware.append(
            HumanInTheLoopMiddleware(interrupt_on=interrupt_on)
        )

    return create_agent(
        resolved_model,
        system_prompt=_compose_system_prompt(system_prompt),
        tools=resolved_tools,
        middleware=deepagent_middleware,
        response_format=response_format,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 1000})
