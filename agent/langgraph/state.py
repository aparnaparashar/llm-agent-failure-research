"""
LangGraph agent state definition.

Defines the state that flows through the LangGraph graph.
This state is the single source of truth for the agent's execution.
"""

from typing import Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    State for the failure-research LangGraph agent.
    
    Extends MessagesState (which provides 'messages' list) with
    additional fields needed for telemetry, failure injection, and
    trajectory recording.
    """
    
    # Task information
    task_id: str
    task_description: str
    
    # Execution tracking
    current_step: int
    max_steps: int
    execution_mode: str  # "healthy", "injected", "organic"
    
    # Tool execution state
    last_tool_name: Optional[str]
    last_tool_args: Optional[dict]
    last_tool_result: Optional[str]
    last_tool_success: Optional[bool]
    
    # Error tracking
    error_count: int
    retry_count: int
    consecutive_errors: int
    
    # Agent status
    agent_status: str  # "running", "completed", "failed", "aborted"
    final_answer: Optional[str]
    
    # Trajectory metadata
    trajectory_id: str
    seed: int
    
    # Injection metadata (only populated in injected mode)
    injection_config: Optional[dict]
    fault_injected: bool
    fault_exposed: bool
    injection_step: Optional[int]
    
    # Telemetry collection point
    step_telemetry: list  # List of per-step telemetry records
