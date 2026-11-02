"""
ToolBench Benchmark Loader (Direct Hugging Face Stream).
Connects directly to https://huggingface.co/datasets/Maurus/ToolBench
and provides realistic multi-step tool-use tasks and dynamic API tools.
"""

import os
import ast
import json
import logging
from typing import Dict, Any, List, Optional
from datasets import load_dataset
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


def sanitize_tool_name(name: str) -> str:
    """Sanitizes tool name to be alphanumeric + underscore for LLM compatibility."""
    valid_chars = [c if c.isalnum() or c == '_' else '_' for c in name]
    sanitized = "".join(valid_chars).strip('_')
    return sanitized or "tool"


def create_dynamic_tool(api_spec: dict) -> StructuredTool:
    """Dynamically converts a ToolBench API definition into a LangChain tool."""
    tool_name = sanitize_tool_name(f"{api_spec.get('tool_name', 'tool')}_{api_spec.get('api_name', 'api')}")
    desc = api_spec.get("api_description", "").strip() or f"Execute API endpoint {api_spec.get('api_name')}"
    req_params = [
        p.get("name") for p in api_spec.get("required_parameters", [])
        if isinstance(p, dict) and p.get("name")
    ]

    def tool_func(**kwargs) -> str:
        # Check required parameters for organic tool validation
        missing = [p for p in req_params if p not in kwargs]
        if missing:
            return json.dumps({
                "status": "error",
                "error": f"Missing required parameters: {missing}",
                "tool": tool_name
            })
        return json.dumps({
            "status": "success",
            "tool": tool_name,
            "data": f"Executed endpoint {api_spec.get('api_name')} with parameters {kwargs}",
        })

    return StructuredTool.from_function(
        func=tool_func,
        name=tool_name,
        description=desc[:200]
    )


class ToolBenchTaskSuite:
    """
    Direct-streaming benchmark suite connecting to Maurus/ToolBench on Hugging Face.
    """

    def __init__(self, dataset_name: str = "Maurus/ToolBench"):
        self.dataset_name = dataset_name
        self._dataset = None
        self._stream_iter = None
        self._initialized = False

    def _init_stream(self):
        if not self._initialized:
            logger.info(f"Connecting directly to Hugging Face dataset: {self.dataset_name} (streaming)...")
            self._dataset = load_dataset(self.dataset_name, split="train", streaming=True)
            self._stream_iter = iter(self._dataset)
            self._initialized = True

    def generate_task_instance(self, index: int, seed: int = 42) -> Dict[str, Any]:
        """
        Streams the next valid task from Maurus/ToolBench.
        """
        self._init_stream()

        # Fetch next valid task with query and APIs
        while True:
            try:
                row = next(self._stream_iter)
            except StopIteration:
                # Reset stream if reached end
                self._stream_iter = iter(self._dataset)
                row = next(self._stream_iter)

            raw_apis = row.get("api_list", "[]")
            try:
                apis = ast.literal_eval(raw_apis) if isinstance(raw_apis, str) else raw_apis
            except Exception:
                apis = []

            query = row.get("query", "").strip()
            if query and apis and isinstance(apis, list):
                break

        query_id = row.get("query_id", index)
        task_id = f"tb_{query_id}_{index}"
        domain = row.get("domain", "general")

        # Create executable LangChain tools from the APIs
        dynamic_tools = []
        tool_names = []
        for api in apis[:5]:  # Top 5 APIs for the task
            try:
                t = create_dynamic_tool(api)
                dynamic_tools.append(t)
                tool_names.append(t.name)
            except Exception as e:
                logger.warning(f"Failed to create dynamic tool: {e}")

        return {
            "task_id": task_id,
            "category": domain,
            "description": query,
            "expected_tools": tool_names,
            "dynamic_tools": dynamic_tools,
            "api_list": apis,
            "difficulty": "medium",
            "seed": seed,
            "instance_index": index,
        }

    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Compatibility method: generates a sample slice."""
        return [self.generate_task_instance(i) for i in range(25)]

    def get_task_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.generate_task_instance(0)
