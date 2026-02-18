"""Unit tests for the read_skill tool."""

from pathlib import Path

from langchain.tools import ToolRuntime

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.read_skill import create_read_skill_tool


def _runtime() -> ToolRuntime:
    return ToolRuntime(
        state={},
        context={},
        config={},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id="call_read_skill",
        store=None,
    )


def test_read_skill_tool_reads_file(tmp_path: Path) -> None:
    """read_skill should read files within allowed skill prefixes."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    skill_file = tmp_path / "skills" / "user" / "test-skill" / "SKILL.md"
    backend.upload_files(
        [
            (
                str(skill_file),
                b"---\nname: test-skill\ndescription: demo\n---\n\n# Test Skill\n",
            )
        ]
    )

    tool = create_read_skill_tool(
        backend=backend,
        allowed_prefixes=[str(tmp_path / "skills" / "user")],
    )

    assert tool.func is not None
    result = tool.func(
        file_path=str(skill_file),
        runtime=_runtime(),
        offset=0,
        limit=50,
    )

    assert "test-skill" in result
    assert "Test Skill" in result


def test_read_skill_tool_blocks_outside_prefix(tmp_path: Path) -> None:
    """read_skill should reject paths outside configured skill sources."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)

    outside_file = tmp_path / "outside.md"
    backend.upload_files([(str(outside_file), b"secret")])

    tool = create_read_skill_tool(
        backend=backend,
        allowed_prefixes=[str(tmp_path / "skills" / "user")],
    )

    assert tool.func is not None
    result = tool.func(
        file_path=str(outside_file),
        runtime=_runtime(),
    )

    assert isinstance(result, str)
    assert result.startswith("Error:")
    assert "Path must start with one of" in result


def test_read_skill_tool_with_virtual_mode_backend(tmp_path: Path) -> None:
    """read_skill should work with virtual-mode paths used by gateway."""
    backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=True)
    backend.upload_files(
        [
            (
                "/skills/user/database-optimizer/SKILL.md",
                b"---\nname: database-optimizer\ndescription: demo\n---\n",
            )
        ]
    )

    tool = create_read_skill_tool(
        backend=backend,
        allowed_prefixes=["/skills/user"],
    )

    assert tool.func is not None
    result = tool.func(
        file_path="/skills/user/database-optimizer/SKILL.md",
        runtime=_runtime(),
    )

    assert "database-optimizer" in result


def test_read_skill_tool_schema_is_json_serializable() -> None:
    """read_skill schema should be generated without runtime/callable fields."""
    backend = FilesystemBackend(root_dir="/", virtual_mode=True)
    tool = create_read_skill_tool(
        backend=backend,
        allowed_prefixes=["/skills/user"],
    )

    schema = tool.tool_call_schema.model_json_schema()
    properties = schema.get("properties", {})

    assert set(properties) == {"file_path", "offset", "limit"}
