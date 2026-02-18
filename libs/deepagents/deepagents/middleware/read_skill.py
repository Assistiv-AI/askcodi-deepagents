"""Read-skill tool for loading skill files without full filesystem tool access.

This module is additive and does not modify FilesystemMiddleware/SkillsMiddleware
behavior. It provides a scoped `read_skill` tool that can be attached as an
external tool in no-filesystem profiles.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated

from langchain.agents.middleware.types import AgentState
from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from deepagents.middleware.filesystem import _validate_path

DEFAULT_READ_OFFSET = 0
DEFAULT_READ_LIMIT = 100

READ_SKILL_TOOL_DESCRIPTION = """Reads a skill file from the skills library.

Use this tool to open `SKILL.md` and referenced markdown files for a selected skill.
You should pass one of the paths shown in the skills list whenever possible.

Usage:
- By default, reads up to 100 lines starting at line 0.
- Use pagination for larger files: `read_skill(file_path=..., offset=100, limit=200)`.
- The tool only allows paths within configured skill source directories.
- Returns backend read output with line numbers for text files.
"""


def _normalize_skill_source_prefix(source_path: str) -> str:
    """Normalize a skill source path to canonical absolute prefix form."""
    normalized = PurePosixPath("/" + source_path.lstrip("/")).as_posix()
    if not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _resolve_backend(
    backend: BACKEND_TYPES,
    runtime: ToolRuntime[None, AgentState],
) -> BackendProtocol:
    """Resolve backend instance from object or factory."""
    if callable(backend):
        resolved = backend(runtime)
        if resolved is None:
            raise AssertionError("read_skill requires a valid backend instance")
        return resolved
    return backend


def create_read_skill_tool(
    *,
    backend: BACKEND_TYPES,
    allowed_prefixes: Sequence[str],
    description: str = READ_SKILL_TOOL_DESCRIPTION,
) -> BaseTool:
    """Create a scoped `read_skill` tool for reading skill files."""
    if not allowed_prefixes:
        raise ValueError("allowed_prefixes must not be empty for read_skill tool")

    normalized_prefixes = tuple(_normalize_skill_source_prefix(prefix) for prefix in allowed_prefixes)

    def sync_read_skill(
        file_path: Annotated[str, "Absolute path to the skill file to read."],
        runtime: ToolRuntime[None, AgentState],
        offset: Annotated[int, "Line number to start reading from (0-indexed)."] = DEFAULT_READ_OFFSET,
        limit: Annotated[int, "Maximum number of lines to read."] = DEFAULT_READ_LIMIT,
    ) -> str:
        """Synchronous wrapper for read_skill."""
        resolved_backend = _resolve_backend(backend, runtime)
        try:
            validated_path = _validate_path(file_path, allowed_prefixes=normalized_prefixes)
        except ValueError as e:
            return f"Error: {e}"
        return resolved_backend.read(validated_path, offset=offset, limit=limit)

    async def async_read_skill(
        file_path: Annotated[str, "Absolute path to the skill file to read."],
        runtime: ToolRuntime[None, AgentState],
        offset: Annotated[int, "Line number to start reading from (0-indexed)."] = DEFAULT_READ_OFFSET,
        limit: Annotated[int, "Maximum number of lines to read."] = DEFAULT_READ_LIMIT,
    ) -> str:
        """Asynchronous wrapper for read_skill."""
        resolved_backend = _resolve_backend(backend, runtime)
        try:
            validated_path = _validate_path(file_path, allowed_prefixes=normalized_prefixes)
        except ValueError as e:
            return f"Error: {e}"
        return await resolved_backend.aread(validated_path, offset=offset, limit=limit)

    return StructuredTool.from_function(
        name="read_skill",
        description=description,
        func=sync_read_skill,
        coroutine=async_read_skill,
    )


__all__ = ["create_read_skill_tool", "READ_SKILL_TOOL_DESCRIPTION"]
