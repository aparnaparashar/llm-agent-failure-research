"""
Telemetry schema definition.

Defines the schema for per-step telemetry records and trajectory metadata.
Only fields that actually exist are populated — never fabricated.
"""

from typing import Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class StepTelemetry:
    """Per-step telemetry record. Only populated fields are logged."""
    
    # Identity
    trajectory_id: str = ""
    step: int = 0
    timestamp: float = 0.0
    
    # State
    state_hash: str = ""
    task_text: str = ""
    goal_representation: str = ""
    
    # LLM action
    llm_action: str = ""  # "tool_call", "final_answer", "error"
    
    # Tool information
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None
    tool_response: Optional[str] = None
    tool_success: Optional[bool] = None
    
    # Error tracking
    error_flag: bool = False
    error_type: Optional[str] = None
    retry_count: int = 0
    
    # Timing
    latency: float = 0.0
    llm_latency: float = 0.0
    tool_latency: float = 0.0
    
    # Token usage (only when available from Ollama)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    
    # Context
    context_length: Optional[int] = None
    message_count: int = 0
    
    # Termination
    termination_status: Optional[str] = None  # "running", "completed", "failed"
    
    def to_dict(self) -> dict:
        """Convert to dict, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class TrajectoryMetadata:
    """Complete trajectory metadata as specified in the project schema."""
    
    trajectory_id: str = ""
    task_id: str = ""
    benchmark: str = ""
    benchmark_version: str = ""
    source: str = ""
    source_version: str = ""
    agent_name: str = ""
    agent_version: str = ""
    
    # LLM
    llm_provider: str = ""
    llm_model: str = ""
    model_version: Optional[str] = None
    temperature: float = 0.0
    prompt_version: str = ""
    
    # Execution
    seed: int = 42
    execution_mode: str = "healthy"
    
    # Failure
    failure_module: Optional[str] = None
    failure_type: Optional[str] = None
    temporal_mechanism: Optional[str] = None
    fault_injected: bool = False
    fault_exposed: bool = False
    injection_step: Optional[int] = None
    actual_failure_step: Optional[int] = None
    severity: Optional[str] = None
    onset_ramp: Optional[str] = None
    agent_recovered: Optional[bool] = None
    agent_failed: bool = False
    task_success: bool = False
    
    # Statistics
    num_steps: int = 0
    total_latency: float = 0.0
    total_tokens: Optional[int] = None
    total_tool_calls: int = 0
    start_time: str = ""
    end_time: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)
