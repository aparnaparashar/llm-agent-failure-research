"""
LangGraph graph construction for the failure-research agent.

Assembles the complete agent graph:
    START -> agent -> {tools | END}
    tools -> agent

The graph uses the AgentState as its state schema and integrates
with the tool executor (which provides the failure injection point).
"""

import logging
from typing import Optional

from langgraph.graph import StateGraph, END

from agent.langgraph.state import AgentState
from agent.langgraph.nodes import create_agent_node, create_tool_node
from agent.langgraph.routing import should_continue
from agent.tools.registry import ToolRegistry
from agent.tools.executor import ToolExecutor
from agent.llm.ollama import OllamaLLM

logger = logging.getLogger(__name__)


def build_agent_graph(
    ollama_llm: OllamaLLM,
    tool_registry: Optional[ToolRegistry] = None,
    tool_executor: Optional[ToolExecutor] = None,
    max_steps: int = 30,
) -> StateGraph:
    """
    Build the complete LangGraph agent graph.
    
    Architecture:
        START -> agent_node -> should_continue -> tools_node -> agent_node -> ...
                                               -> END
    
    Args:
        ollama_llm: The Ollama LLM wrapper
        tool_registry: Registry of available tools (defaults to builtins)
        tool_executor: Tool executor with injection support
        max_steps: Maximum execution steps
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Setup tools
    if tool_registry is None:
        tool_registry = ToolRegistry()
    
    tools = tool_registry.get_tools()
    
    if tool_executor is None:
        tool_executor = ToolExecutor(tools=tools)
    
    # Get the LLM
    llm = ollama_llm.get_llm()
    
    # Create nodes
    agent_node = create_agent_node(llm, tools)
    tool_node = create_tool_node(tool_executor)
    
    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    
    # Set entry point
    graph.set_entry_point("agent")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    
    # Tools always route back to agent
    graph.add_edge("tools", "agent")
    
    # Compile the graph
    compiled = graph.compile()
    
    logger.info(
        f"Agent graph built with {len(tools)} tools: "
        f"{[t.name for t in tools]}"
    )
    
    return compiled


def create_initial_state(
    task_id: str,
    task_description: str,
    trajectory_id: str,
    execution_mode: str = "healthy",
    seed: int = 42,
    max_steps: int = 30,
    injection_config: Optional[dict] = None,
) -> dict:
    """
    Create the initial state for a LangGraph execution.
    
    Args:
        task_id: Unique task identifier
        task_description: The task the agent should perform
        trajectory_id: Unique trajectory identifier
        execution_mode: "healthy", "injected", or "organic"
        seed: Random seed for this run
        max_steps: Maximum number of steps
        injection_config: Configuration for failure injection (injected mode only)
    
    Returns:
        Initial state dict for the graph
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=task_description)],
        "task_id": task_id,
        "task_description": task_description,
        "current_step": 0,
        "max_steps": max_steps,
        "execution_mode": execution_mode,
        "last_tool_name": None,
        "last_tool_args": None,
        "last_tool_result": None,
        "last_tool_success": None,
        "error_count": 0,
        "retry_count": 0,
        "consecutive_errors": 0,
        "agent_status": "running",
        "final_answer": None,
        "trajectory_id": trajectory_id,
        "seed": seed,
        "injection_config": injection_config,
        "fault_injected": False,
        "fault_exposed": False,
        "injection_step": injection_config.get("injection_step") if injection_config else None,
        "step_telemetry": [],
    }
