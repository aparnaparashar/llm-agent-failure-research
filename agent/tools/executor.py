"""
Tool executor with failure injection integration point.

The executor wraps tool calls to:
1. Record execution telemetry (timing, success/failure, etc.)
2. Provide the injection point for controlled failure injection
3. Handle retries and error recording

This is where failure injection MUST happen — during actual execution,
not after the fact.
"""

import time
import json
import logging
from typing import Any, Optional, Callable

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


class ToolExecutor:
    """
    Executes tools with telemetry recording and failure injection support.
    
    The injection hook is called DURING tool execution, ensuring that:
    - The failure happens while the agent is running
    - The LLM sees the failure as part of the tool observation
    - The agent reacts to the failure causally
    """

    def __init__(
        self,
        tools: list[StructuredTool],
        injection_hook: Optional[Callable] = None,
        max_retries: int = 2,
        timeout: float = 60.0,
    ):
        self._tools = {t.name: t for t in tools}
        self._injection_hook = injection_hook
        self._max_retries = max_retries
        self._timeout = timeout
        self._execution_log: list[dict] = []

    def set_injection_hook(self, hook: Callable):
        """Set the failure injection hook. Called during tool execution."""
        self._injection_hook = hook

    def clear_injection_hook(self):
        """Remove any active injection hook."""
        self._injection_hook = None

    def execute(
        self,
        tool_name: str,
        tool_args: dict,
        step: int,
        execution_mode: str = "healthy",
    ) -> dict:
        """
        Execute a tool call with telemetry and optional failure injection.
        
        Returns a dict with:
        - result: the tool's output (or injected failure)
        - success: bool
        - latency: float (seconds)
        - error: optional error message
        - injected: whether a fault was injected at this step
        """
        record = {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "step": step,
            "execution_mode": execution_mode,
            "success": False,
            "result": None,
            "error": None,
            "latency": 0.0,
            "injected": False,
            "injection_details": None,
        }

        tool = self._tools.get(tool_name)
        if tool is None:
            record["error"] = f"Tool '{tool_name}' not found in registry."
            record["result"] = json.dumps({
                "status": "error",
                "error": f"Tool '{tool_name}' is not available.",
            })
            self._execution_log.append(record)
            return record

        start_time = time.time()
        try:
            # Execute the actual tool
            raw_result = tool.invoke(tool_args)
            record["latency"] = time.time() - start_time
            record["result"] = str(raw_result)
            record["success"] = True

            # === FAILURE INJECTION POINT ===
            # This is the critical integration point.
            # In "injected" mode, the hook can MUTATE the result before
            # it's returned to the LLM, ensuring the LLM reacts to the
            # injected failure causally.
            if (
                execution_mode == "injected"
                and self._injection_hook is not None
            ):
                injection_result = self._injection_hook(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    tool_result=record["result"],
                    step=step,
                )
                if injection_result is not None:
                    record["result"] = injection_result["mutated_result"]
                    record["success"] = injection_result.get("success", False)
                    record["injected"] = True
                    record["injection_details"] = injection_result.get("details", {})
                    record["error"] = injection_result.get("error_message")
                    logger.info(
                        f"Fault injected at step {step}, tool={tool_name}: "
                        f"{injection_result.get('fault_type', 'unknown')}"
                    )

        except Exception as e:
            record["latency"] = time.time() - start_time
            record["error"] = str(e)
            record["result"] = json.dumps({
                "status": "error",
                "error": str(e),
            })
            logger.warning(f"Tool execution error at step {step}: {e}")

        self._execution_log.append(record)
        return record

    def get_execution_log(self) -> list[dict]:
        """Get the full execution log."""
        return list(self._execution_log)

    def reset_log(self):
        """Clear the execution log."""
        self._execution_log = []

    def get_available_tools(self) -> list[str]:
        """Get names of available tools."""
        return list(self._tools.keys())
