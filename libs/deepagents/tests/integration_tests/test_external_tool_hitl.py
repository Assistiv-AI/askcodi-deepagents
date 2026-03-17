"""Integration tests for ExternalToolMiddleware — tool_result decision type."""

import uuid

from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from deepagents.graph import create_deep_agent


@tool(description="Search the web for information")
def web_search(query: str):
    """Stub — should never run server-side."""
    raise RuntimeError("web_search executed server-side; expected HITL interrupt")


@tool(description="Get the current weather for a location")
def get_weather(location: str):
    return f"The weather in {location} is sunny."


EXTERNAL_TOOL_CONFIG = {
    "web_search": {"allowed_decisions": ["tool_result"]},
}


class TestExternalToolMiddleware:
    """Tests for the tool_result decision path used by external/client-side tools."""

    def test_interrupt_on_external_tool(self):
        """Verify the agent interrupts when an external tool is called."""
        checkpointer = MemorySaver()
        agent = create_deep_agent(
            tools=[web_search, get_weather],
            interrupt_on=EXTERNAL_TOOL_CONFIG,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Search the web for 'LangGraph HITL patterns'"}]},
            config=config,
        )

        # Should have interrupted
        assert "__interrupt__" in result
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]
        assert any(ar["name"] == "web_search" for ar in action_requests)

        # Review config should only allow tool_result
        review_configs = interrupts["review_configs"]
        ws_config = next(rc for rc in review_configs if rc["action_name"] == "web_search")
        assert ws_config["allowed_decisions"] == ["tool_result"]

    def test_resume_with_tool_result(self):
        """Verify resume with tool_result decision injects client output and continues."""
        checkpointer = MemorySaver()
        agent = create_deep_agent(
            tools=[web_search, get_weather],
            interrupt_on=EXTERNAL_TOOL_CONFIG,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        # First invoke — triggers interrupt
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "Search the web for 'LangGraph HITL patterns'"}]},
            config=config,
        )
        assert "__interrupt__" in result

        # Resume with tool_result
        client_output = "LangGraph supports HITL via interrupt/resume with Command objects."
        result2 = agent.invoke(
            Command(resume={"decisions": [{"type": "tool_result", "output": client_output}]}),
            config=config,
        )

        # Should NOT be interrupted anymore
        assert "__interrupt__" not in result2

        # The tool result should appear in messages
        tool_messages = [m for m in result2["messages"] if m.type == "tool"]
        assert any(
            m.name == "web_search" and client_output in m.content
            for m in tool_messages
        )

    def test_mixed_external_and_internal_tools(self):
        """External tool interrupts; internal tool (get_weather) is auto-approved."""
        checkpointer = MemorySaver()
        agent = create_deep_agent(
            tools=[web_search, get_weather],
            interrupt_on=EXTERNAL_TOOL_CONFIG,
            checkpointer=checkpointer,
        )
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Search the web for 'latest news' and also get the weather in Tokyo",
                    }
                ]
            },
            config=config,
        )

        # Should interrupt for web_search
        assert "__interrupt__" in result
        interrupts = result["__interrupt__"][0].value
        action_requests = interrupts["action_requests"]
        # Only web_search should be in action_requests (get_weather is auto-approved)
        assert all(ar["name"] == "web_search" for ar in action_requests)

        # Resume
        result2 = agent.invoke(
            Command(resume={"decisions": [{"type": "tool_result", "output": "Breaking news: AI advances"}]}),
            config=config,
        )
        assert "__interrupt__" not in result2

        # Both tool results should be present
        tool_messages = [m for m in result2["messages"] if m.type == "tool"]
        tool_names = {m.name for m in tool_messages}
        assert "web_search" in tool_names
        assert "get_weather" in tool_names
