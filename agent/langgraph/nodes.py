"""
LangGraph node definitions for the failure-research agent.

Each node is a function that takes the AgentState and returns
state updates. These are the actual processing steps in the
agent's execution graph.
"""

import json
import time
import hashlib
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

# System prompt for the research agent
SYSTEM_PROMPT = """You are a helpful AI assistant that can use tools to accomplish tasks.

When given a task:
1. Think about what information you need and which tools can help.
2. Use the available tools by calling them with appropriate arguments.
3. Analyze the results from tool calls.
4. If a tool call fails, consider retrying with different arguments or using a different approach.
5. When you have enough information, provide a final answer.

Always be systematic and thorough. If you encounter errors, try to recover gracefully.
When you have completed the task, provide your final answer clearly.

IMPORTANT: When you are done and have a final answer, respond with your answer directly without calling any more tools."""


def create_agent_node(llm, tools):
    """
    Create the agent/planner node.
    
    This node:
    1. Receives the current state (including message history)
    2. Calls the Ollama LLM with the conversation so far
    3. The LLM decides what to do: call a tool or provide a final answer
    """
    # Bind tools to the LLM so it can generate tool calls
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: dict) -> dict:
        """Agent node: LLM reasoning and decision-making."""
        messages = state.get("messages", [])
        current_step = state.get("current_step", 0)
        
        # Add system message if not present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        # Record timing
        start_time = time.time()
        
        try:
            # Call the actual Ollama LLM
            response = llm_with_tools.invoke(messages)
            llm_latency = time.time() - start_time
            
            # Build step telemetry record
            step_record = {
                "step": current_step,
                "timestamp": time.time(),
                "node": "agent",
                "llm_latency": round(llm_latency, 4),
                "has_tool_calls": bool(response.tool_calls),
                "response_length": len(response.content) if response.content else 0,
                "state_hash": _compute_state_hash(state),
            }

            # Estimate token usage from response metadata if available
            response_metadata = getattr(response, "response_metadata", {}) or {}
            if "prompt_eval_count" in response_metadata:
                step_record["input_tokens"] = response_metadata["prompt_eval_count"]
            if "eval_count" in response_metadata:
                step_record["output_tokens"] = response_metadata["eval_count"]

            # Update step telemetry
            step_telemetry = list(state.get("step_telemetry", []))
            step_telemetry.append(step_record)

            return {
                "messages": [response],
                "current_step": current_step + 1,
                "step_telemetry": step_telemetry,
            }

        except Exception as e:
            logger.error(f"Agent node error at step {current_step}: {e}")
            llm_latency = time.time() - start_time
            
            error_msg = AIMessage(
                content=f"I encountered an error: {str(e)}. Let me try a different approach."
            )
            
            step_telemetry = list(state.get("step_telemetry", []))
            step_telemetry.append({
                "step": current_step,
                "timestamp": time.time(),
                "node": "agent",
                "llm_latency": round(llm_latency, 4),
                "error": str(e),
                "has_tool_calls": False,
            })
            
            return {
                "messages": [error_msg],
                "current_step": current_step + 1,
                "error_count": state.get("error_count", 0) + 1,
                "consecutive_errors": state.get("consecutive_errors", 0) + 1,
                "step_telemetry": step_telemetry,
            }

    return agent_node


def create_tool_node(tool_executor):
    """
    Create the tool execution node.
    
    This node:
    1. Extracts tool calls from the last AI message
    2. Executes each tool through the ToolExecutor (with injection point)
    3. Returns tool results as ToolMessages
    """

    def tool_node(state: dict) -> dict:
        """Tool node: execute tool calls from the agent."""
        messages = state.get("messages", [])
        current_step = state.get("current_step", 0)
        execution_mode = state.get("execution_mode", "healthy")
        
        # Get the last AI message with tool calls
        last_message = messages[-1] if messages else None
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return {}

        tool_messages = []
        tool_telemetry = []
        error_count_delta = 0
        consecutive_errors = state.get("consecutive_errors", 0)
        last_tool_name = None
        last_tool_args = None
        last_tool_result = None
        last_tool_success = None

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"call_{current_step}")

            # Execute through the ToolExecutor (injection happens here!)
            exec_result = tool_executor.execute(
                tool_name=tool_name,
                tool_args=tool_args,
                step=current_step,
                execution_mode=execution_mode,
            )

            # Create ToolMessage for the LLM to see
            tool_messages.append(
                ToolMessage(
                    content=exec_result["result"],
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )

            # Record telemetry
            tool_telemetry.append({
                "step": current_step,
                "timestamp": time.time(),
                "node": "tool",
                "tool_name": tool_name,
                "tool_args": tool_args,
                "tool_success": exec_result["success"],
                "tool_latency": round(exec_result["latency"], 4),
                "tool_error": exec_result.get("error"),
                "injected": exec_result.get("injected", False),
                "injection_details": exec_result.get("injection_details"),
            })

            # Track errors
            if not exec_result["success"]:
                error_count_delta += 1
                consecutive_errors += 1
            else:
                consecutive_errors = 0

            last_tool_name = tool_name
            last_tool_args = tool_args
            last_tool_result = exec_result["result"]
            last_tool_success = exec_result["success"]

        # Update step telemetry
        step_telemetry = list(state.get("step_telemetry", []))
        step_telemetry.extend(tool_telemetry)

        # Check if injection happened
        fault_injected = state.get("fault_injected", False)
        fault_exposed = state.get("fault_exposed", False)
        for t in tool_telemetry:
            if t.get("injected", False):
                fault_injected = True
                fault_exposed = True  # The LLM will see this in the ToolMessage

        return {
            "messages": tool_messages,
            "last_tool_name": last_tool_name,
            "last_tool_args": last_tool_args,
            "last_tool_result": last_tool_result,
            "last_tool_success": last_tool_success,
            "error_count": state.get("error_count", 0) + error_count_delta,
            "consecutive_errors": consecutive_errors,
            "step_telemetry": step_telemetry,
            "fault_injected": fault_injected,
            "fault_exposed": fault_exposed,
        }

    return tool_node


def _compute_state_hash(state: dict) -> str:
    """Compute a hash of the current state for tracking."""
    # Hash key state elements for change detection
    key_elements = {
        "step": state.get("current_step", 0),
        "msg_count": len(state.get("messages", [])),
        "error_count": state.get("error_count", 0),
    }
    return hashlib.md5(json.dumps(key_elements).encode()).hexdigest()[:12]
