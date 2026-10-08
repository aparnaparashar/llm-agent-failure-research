"""
Main execution runner for the LangGraph agent.

Orchestrates a single trajectory execution through:
  Task -> LangGraph -> Ollama -> Tools -> Telemetry -> Trajectory

Supports three execution modes: healthy, injected, organic.
"""

import os
import json
import time
import uuid
import logging
from typing import Any, Optional
from datetime import datetime

import yaml
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

from agent.langgraph.graph import build_agent_graph, create_initial_state
from agent.llm.ollama import OllamaLLM, OllamaConfig
from agent.tools.registry import ToolRegistry
from agent.tools.executor import ToolExecutor

logger = logging.getLogger(__name__)


class TrajectoryRunner:
    """
    Runs a single task through the LangGraph agent and records
    the complete trajectory with telemetry.
    """

    def __init__(
        self,
        ollama_llm: Optional[OllamaLLM] = None,
        tool_registry: Optional[ToolRegistry] = None,
        config_path: Optional[str] = None,
        output_dir: str = "data/generated",
        max_steps: int = 30,
    ):
        # Load LLM
        if ollama_llm is None:
            llm_config_path = config_path or os.path.join("configs", "llm.yaml")
            self.ollama_llm = OllamaLLM(OllamaConfig.from_yaml(llm_config_path))
        else:
            self.ollama_llm = ollama_llm

        # Setup tools
        self.tool_registry = tool_registry or ToolRegistry()
        self.tool_executor = ToolExecutor(
            tools=self.tool_registry.get_tools(),
        )

        self.output_dir = output_dir
        self.max_steps = max_steps

        # Build the graph
        self.graph = build_agent_graph(
            ollama_llm=self.ollama_llm,
            tool_registry=self.tool_registry,
            tool_executor=self.tool_executor,
            max_steps=max_steps,
        )

    def run(
        self,
        task_id: str,
        task_description: str,
        execution_mode: str = "healthy",
        seed: int = 42,
        injection_config: Optional[dict] = None,
        injection_hook: Optional[Any] = None,
    ) -> dict:
        """
        Execute a single task and return the complete trajectory.
        
        Args:
            task_id: Unique task identifier
            task_description: The task text
            execution_mode: "healthy", "injected", or "organic"
            seed: Random seed
            injection_config: Injection parameters (injected mode only)
            injection_hook: Callable for failure injection (injected mode only)
        
        Returns:
            Complete trajectory dict with metadata, steps, and telemetry
        """
        trajectory_id = f"traj_{task_id}_{execution_mode}_{seed}_{uuid.uuid4().hex[:8]}"
        
        logger.info(
            f"Starting {execution_mode} run: task={task_id}, "
            f"trajectory={trajectory_id}, seed={seed}"
        )

        # Set up injection if in injected mode
        if execution_mode == "injected" and injection_hook is not None:
            self.tool_executor.set_injection_hook(injection_hook)
        else:
            self.tool_executor.clear_injection_hook()

        # Reset tool execution log
        self.tool_executor.reset_log()

        # Create initial state
        initial_state = create_initial_state(
            task_id=task_id,
            task_description=task_description,
            trajectory_id=trajectory_id,
            execution_mode=execution_mode,
            seed=seed,
            max_steps=self.max_steps,
            injection_config=injection_config,
        )

        # Execute the graph
        start_time = time.time()
        start_datetime = datetime.utcnow().isoformat()

        try:
            # Run the LangGraph agent
            final_state = self.graph.invoke(
                initial_state,
                config={"recursion_limit": self.max_steps * 2 + 10},
            )
            agent_status = "completed"
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            final_state = initial_state
            final_state["agent_status"] = "failed"
            agent_status = "failed"

        end_time = time.time()
        end_datetime = datetime.utcnow().isoformat()
        total_latency = end_time - start_time

        # Extract trajectory data
        trajectory = self._build_trajectory(
            final_state=final_state,
            task_id=task_id,
            task_description=task_description,
            trajectory_id=trajectory_id,
            execution_mode=execution_mode,
            seed=seed,
            injection_config=injection_config,
            start_time=start_datetime,
            end_time=end_datetime,
            total_latency=total_latency,
            agent_status=agent_status,
        )

        # Save trajectory
        self._save_trajectory(trajectory, execution_mode)

        # Clean up injection hook
        self.tool_executor.clear_injection_hook()

        logger.info(
            f"Completed {execution_mode} run: trajectory={trajectory_id}, "
            f"steps={trajectory['num_steps']}, "
            f"latency={total_latency:.2f}s, "
            f"status={agent_status}"
        )

        return trajectory

    def _build_trajectory(
        self,
        final_state: dict,
        task_id: str,
        task_description: str,
        trajectory_id: str,
        execution_mode: str,
        seed: int,
        injection_config: Optional[dict],
        start_time: str,
        end_time: str,
        total_latency: float,
        agent_status: str,
    ) -> dict:
        """Build the complete trajectory record from the final state."""
        messages = final_state.get("messages", [])
        step_telemetry = final_state.get("step_telemetry", [])
        tool_log = self.tool_executor.get_execution_log()

        # Extract steps from messages
        steps = []
        step_idx = 0
        for msg in messages:
            step = {
                "trajectory_id": trajectory_id,
                "step": step_idx,
                "timestamp": time.time(),
                "message_type": type(msg).__name__,
            }

            if isinstance(msg, SystemMessage):
                step["role"] = "system"
                step["content"] = msg.content[:500]
            elif isinstance(msg, HumanMessage):
                step["role"] = "human"
                step["content"] = msg.content[:500]
                step["task_text"] = task_description
            elif isinstance(msg, AIMessage):
                step["role"] = "assistant"
                step["content"] = msg.content[:500] if msg.content else ""
                step["has_tool_calls"] = bool(msg.tool_calls)
                if msg.tool_calls:
                    step["tool_calls"] = [
                        {
                            "name": tc["name"],
                            "args": tc.get("args", {}),
                            "id": tc.get("id", ""),
                        }
                        for tc in msg.tool_calls
                    ]
            elif isinstance(msg, ToolMessage):
                step["role"] = "tool"
                step["tool_name"] = getattr(msg, "name", None)
                step["content"] = msg.content[:500] if msg.content else ""
                step["tool_call_id"] = getattr(msg, "tool_call_id", None)

            steps.append(step)
            step_idx += 1

        # Extract final answer
        final_answer = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_answer = msg.content
                break

        # Count tool calls
        total_tool_calls = sum(
            1 for msg in messages
            if isinstance(msg, AIMessage) and msg.tool_calls
            for _ in msg.tool_calls
        )

        # Determine task success (heuristic: completed without errors at end)
        task_success = (
            agent_status == "completed"
            and final_answer is not None
            and final_state.get("consecutive_errors", 0) == 0
        )

        # Build the trajectory metadata
        llm_metadata = self.ollama_llm.get_metadata()

        trajectory = {
            # Metadata
            "trajectory_id": trajectory_id,
            "task_id": task_id,
            "task_description": task_description,
            "benchmark": "builtin",
            "benchmark_version": "v1.0",
            "source": "langgraph_ollama",
            "source_version": "0.1.0",
            "agent_name": "failure-research-agent",
            "agent_version": "0.1.0",
            
            # LLM metadata
            "llm_provider": llm_metadata["llm_provider"],
            "llm_model": llm_metadata["llm_model"],
            "model_version": llm_metadata.get("model_version"),
            "temperature": llm_metadata["temperature"],
            "prompt_version": llm_metadata["prompt_version"],
            
            # Execution
            "seed": seed,
            "execution_mode": execution_mode,
            "agent_status": agent_status,
            "task_success": task_success,
            "final_answer": final_answer[:1000] if final_answer else None,
            
            # Failure injection
            "failure_module": injection_config.get("module") if injection_config else None,
            "failure_type": injection_config.get("failure_type") if injection_config else None,
            "temporal_mechanism": injection_config.get("temporal_mechanism") if injection_config else None,
            "fault_injected": final_state.get("fault_injected", False),
            "fault_exposed": final_state.get("fault_exposed", False),
            "injection_step": injection_config.get("injection_step") if injection_config else None,
            "actual_failure_step": None,  # Determined post-hoc from telemetry analysis
            "severity": injection_config.get("severity") if injection_config else None,
            "onset_ramp": injection_config.get("onset_ramp") if injection_config else None,
            "agent_recovered": None,  # Determined post-hoc
            "agent_failed": agent_status == "failed",
            
            # Statistics
            "num_steps": len(steps),
            "total_latency": round(total_latency, 4),
            "total_tool_calls": total_tool_calls,
            "error_count": final_state.get("error_count", 0),
            "retry_count": final_state.get("retry_count", 0),
            "start_time": start_time,
            "end_time": end_time,
            
            # Steps and telemetry
            "steps": steps,
            "step_telemetry": step_telemetry,
            "tool_execution_log": tool_log,
        }

        return trajectory

    def _save_trajectory(self, trajectory: dict, execution_mode: str):
        """Save trajectory to the appropriate directory."""
        mode_dir = os.path.join(self.output_dir, execution_mode)
        os.makedirs(mode_dir, exist_ok=True)

        filename = f"{trajectory['trajectory_id']}.json"
        filepath = os.path.join(mode_dir, filename)

        with open(filepath, "w") as f:
            json.dump(trajectory, f, indent=2, default=str)

        logger.info(f"Trajectory saved to {filepath}")
