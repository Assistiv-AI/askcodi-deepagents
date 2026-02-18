"""Tests for profile-based deep agent construction."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from deepagents.profiles import LOCAL_SUBAGENT_NAME, create_deep_agent_with_profile
from tests.unit_tests.chat_model import GenericFakeChatModel


@tool(description="Echo tool for profile tests.")
def echo_tool(message: str) -> str:
    """Return the provided message."""
    return message


def _tool_names(agent: object) -> set[str]:
    """Get tool names from a compiled agent."""
    return set(agent.nodes["tools"].bound._tools_by_name.keys())


def test_full_agent_profile_includes_default_tools() -> None:
    """The full profile should expose filesystem, subagent, and planning tools."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent_with_profile(profile="full_agent", model=model)
    tools = _tool_names(agent)

    assert "ls" in tools
    assert "read_file" in tools
    assert "write_file" in tools
    assert "edit_file" in tools
    assert "glob" in tools
    assert "grep" in tools
    assert "execute" in tools
    assert "task" in tools
    assert "write_todos" in tools


def test_external_tools_only_profile_exposes_no_filesystem_tools() -> None:
    """The external-tools-only profile should remove filesystem and shell tools."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent_with_profile(
        profile="external_tools_only",
        model=model,
        tools=[echo_tool],
    )
    tools = _tool_names(agent)

    assert "echo_tool" in tools
    assert "task" in tools
    assert "write_todos" in tools
    assert "ls" not in tools
    assert "read_file" not in tools
    assert "write_file" not in tools
    assert "edit_file" not in tools
    assert "glob" not in tools
    assert "grep" not in tools
    assert "execute" not in tools


def test_no_shell_profile_keeps_filesystem_but_removes_execute() -> None:
    """The no-shell profile should keep filesystem tools except ``execute``."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent_with_profile(profile="no_shell", model=model)
    tools = _tool_names(agent)

    assert "ls" in tools
    assert "read_file" in tools
    assert "write_file" in tools
    assert "edit_file" in tools
    assert "glob" in tools
    assert "grep" in tools
    assert "execute" not in tools
    assert "task" in tools


def test_no_filesystem_profile_removes_filesystem_tools() -> None:
    """The no-filesystem profile should remove all filesystem and shell tools."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent_with_profile(
        profile="no_filesystem",
        model=model,
        tools=[echo_tool],
    )
    tools = _tool_names(agent)

    assert "echo_tool" in tools
    assert "task" in tools
    assert "write_todos" in tools
    assert "ls" not in tools
    assert "read_file" not in tools
    assert "write_file" not in tools
    assert "edit_file" not in tools
    assert "glob" not in tools
    assert "grep" not in tools
    assert "execute" not in tools


def test_no_filesystem_profile_with_skills_adds_read_skill_tool() -> None:
    """Profiles with skills but no main filesystem tools should expose read_skill."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent_with_profile(
        profile="no_filesystem",
        model=model,
        tools=[echo_tool],
        skills=["/skills/user/"],
    )
    tools = _tool_names(agent)

    assert "echo_tool" in tools
    assert "read_skill" in tools
    assert "read_file" not in tools


def test_local_subagent_profile_has_internal_worker() -> None:
    """The local-subagent profile should expose internal worker in task choices."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    agent = create_deep_agent_with_profile(
        profile="local_subagent",
        model=model,
        tools=[echo_tool],
    )
    tools_by_name = agent.nodes["tools"].bound._tools_by_name
    tools = set(tools_by_name.keys())

    assert "echo_tool" in tools
    assert "task" in tools
    assert "ls" not in tools
    assert "read_file" not in tools
    assert "write_file" not in tools
    assert "execute" not in tools

    task_tool = tools_by_name["task"]
    assert LOCAL_SUBAGENT_NAME in task_tool.description


def test_local_subagent_profile_can_update_internal_files() -> None:
    """The local-subagent profile should allow internal filesystem updates."""
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "Write /memory/notes.txt with content hello",
                                "subagent_type": LOCAL_SUBAGENT_NAME,
                            },
                            "id": "call_task_local_subagent",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {
                                "file_path": "/memory/notes.txt",
                                "content": "hello",
                            },
                            "id": "call_write_note",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Saved memory note."),
                AIMessage(content="Finished."),
            ]
        )
    )
    agent = create_deep_agent_with_profile(
        profile="local_subagent",
        model=model,
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="Create an internal memory note.")]},
        config={"configurable": {"thread_id": "profile-local-subagent-thread"}},
    )

    assert "files" in result
    assert "/memory/notes.txt" in result["files"]
