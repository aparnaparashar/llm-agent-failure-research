"""
Unified Benchmark Adapter.
Adapts external tasks (ToolBench, AgentErrorBench, custom suites) to agent execution tasks.
"""

from typing import Dict, Any, List, Optional
from benchmarks.toolbench import ToolBenchTaskSuite
from benchmarks.agenterrorbench import AgentErrorBenchLoader


class BenchmarkAdapter:
    """Unified task provider for agent experimental runs."""

    def __init__(self):
        self.toolbench = ToolBenchTaskSuite()
        self.agenterrorbench = AgentErrorBenchLoader()

    def get_tasks_for_experiment(
        self,
        benchmark: str = "toolbench",
        limit: Optional[int] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve standardized task dictionaries."""
        if benchmark == "toolbench":
            if category:
                tasks = self.toolbench.get_tasks_by_category(category)
            else:
                tasks = self.toolbench.get_all_tasks()
        elif benchmark == "agenterrorbench":
            raw_samples = self.agenterrorbench.load_or_fetch(max_samples=limit)
            tasks = [
                {
                    "task_id": item.get("id", f"aeb_{i}"),
                    "description": item.get("task", ""),
                    "category": item.get("failure_module", "general"),
                    "benchmark": "agenterrorbench",
                    "grounding": item,
                }
                for i, item in enumerate(raw_samples)
            ]
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        return tasks[:limit] if limit else tasks
