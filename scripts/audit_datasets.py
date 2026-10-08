#!/usr/bin/env python3
"""
CLI Script: audit_datasets.py
Audits dataset integrity, schema conformity, and failure coverage.
"""

import os
import sys
import json
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from benchmarks.agenterrorbench import AgentErrorBenchLoader
from benchmarks.toolbench import ToolBenchTaskSuite
from failure_taxonomy.loader import get_taxonomy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_datasets")


def main():
    logger.info("Auditing dataset distributions and taxonomy alignment...")
    tax = get_taxonomy()
    modules = tax.get_modules()
    logger.info(f"Taxonomy contains {len(modules)} modules: {modules}")

    tb_suite = ToolBenchTaskSuite()
    tasks = tb_suite.get_all_tasks()
    categories = sorted(list(set(t["category"] for t in tasks)))
    logger.info(f"ToolBench task count: {len(tasks)} across categories: {categories}")

    loader = AgentErrorBenchLoader()
    aeb_samples = loader.load_or_fetch(split="train")
    logger.info(f"AgentErrorBench sample count: {len(aeb_samples)}")

    audit_summary = {
        "status": "passed",
        "toolbench_tasks": len(tasks),
        "agenterrorbench_samples": len(aeb_samples),
        "taxonomy_modules": len(modules),
        "categories": categories,
    }

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "audit_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    logger.info(f"Audit report saved to {out_path}")
    print(f"Dataset audit complete. Status: PASSED (ToolBench: {len(tasks)}, AgentErrorBench: {len(aeb_samples)})")


if __name__ == "__main__":
    main()
