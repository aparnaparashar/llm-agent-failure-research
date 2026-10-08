"""
LangGraph routing logic.

Determines the next node to execute based on the current state.
This implements the agent's control flow:
  agent -> tool -> agent -> ... -> end
"""

import logging
from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def should_continue(state: dict) -> str:
    """
    Route after the agent node.
    
    Returns:
        "tools" - if the agent wants to call tools
        "end" - if the agent is done (no tool calls) or max steps reached
    """
    messages = state.get("messages", [])
    current_step = state.get("current_step", 0)
    max_steps = state.get("max_steps", 30)

    # Check step limit
    if current_step >= max_steps:
        logger.warning(f"Max steps ({max_steps}) reached. Terminating.")
        return "end"

    # Check if the last message has tool calls
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

    # No tool calls = agent is providing final answer
    return "end"


def after_tools(state: dict) -> str:
    """
    Route after the tool node.
    
    Always routes back to the agent so the LLM can:
    - See tool results
    - Decide next action (continue, retry, replan, terminate)
    """
    return "agent"
