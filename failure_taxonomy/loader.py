"""
Failure taxonomy module.
Provides programmatic access to the unified failure taxonomy.
"""

import os
import yaml
from typing import Dict, Any, List, Optional


class FailureTaxonomy:
    """Unified failure taxonomy loader and validator."""

    def __init__(self, taxonomy_path: Optional[str] = None):
        if taxonomy_path is None:
            taxonomy_path = os.path.join(
                os.path.dirname(__file__), "taxonomy.yaml"
            )
        self.taxonomy_path = os.path.abspath(taxonomy_path)
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.taxonomy_path):
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_path}")
        with open(self.taxonomy_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("taxonomy", {})

    def get_modules(self) -> List[str]:
        """List all failure modules (e.g., reflection, action, memory, planning, system)."""
        return list(self.data.keys())

    def get_failures_in_module(self, module: str) -> Dict[str, Any]:
        """Get all failure types for a specific module."""
        return self.data.get(module, {})

    def get_failure_spec(self, module: str, failure_type: str) -> Optional[Dict[str, Any]]:
        """Get specification for a specific failure type."""
        return self.data.get(module, {}).get(failure_type)

    def is_injectable(self, module: str, failure_type: str) -> bool:
        """Check if a failure type is injectable in the experimental framework."""
        spec = self.get_failure_spec(module, failure_type)
        if not spec:
            return False
        return spec.get("injectability") in ["injectable", "partially_injectable"]

    def validate_failure(self, module: str, failure_type: str) -> bool:
        """Validate if a module and failure type pair exists in the taxonomy."""
        return failure_type in self.data.get(module, {})


def get_taxonomy() -> FailureTaxonomy:
    """Singleton helper to get the taxonomy."""
    return FailureTaxonomy()
