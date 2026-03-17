"""External tool middleware for client-side tool execution via HITL interrupt/resume.

Extends HumanInTheLoopMiddleware with a "tool_result" decision type that allows
injecting client-provided tool results without executing the stub server-side.
"""

from typing import Any

from langchain_core.messages import ToolCall, ToolMessage

from langchain.agents.middleware.human_in_the_loop import (
    DecisionType,
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)
from langchain.agents.middleware.types import ContextT, ResponseT, StateT


class ExternalToolMiddleware(HumanInTheLoopMiddleware[StateT, ContextT, ResponseT]):
    """HITL middleware that supports a "tool_result" decision for external tools.

    When the client sends a decision with type="tool_result", the middleware
    keeps the original tool_call in the AIMessage and injects a ToolMessage
    with the client-provided output — bypassing the server-side stub entirely.
    """

    @staticmethod
    def _process_decision(
        decision: dict[str, Any],
        tool_call: ToolCall,
        config: InterruptOnConfig,
    ) -> tuple[ToolCall | None, ToolMessage | None]:
        """Process a decision, adding support for 'tool_result' type."""
        allowed_decisions = config["allowed_decisions"]

        if decision["type"] == "tool_result" and "tool_result" in allowed_decisions:
            # Client executed the tool and is providing the result directly.
            # Keep the tool_call (so AIMessage remains valid) and inject a
            # ToolMessage with the client-provided output.
            tool_message = ToolMessage(
                content=decision.get("output", ""),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="success",
            )
            return tool_call, tool_message

        # Delegate all other decision types to the parent implementation
        return HumanInTheLoopMiddleware._process_decision(
            decision, tool_call, config
        )
