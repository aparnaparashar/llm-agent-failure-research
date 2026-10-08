"""
Telemetry logger.

Persists telemetry records to JSONL files for later analysis.
"""

import os
import json
import logging
from typing import Optional

from telemetry.schema import StepTelemetry

logger = logging.getLogger(__name__)


class TelemetryLogger:
    """Writes telemetry records to persistent storage."""

    def __init__(self, output_dir: str = "data/processed/telemetry"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_trajectory_telemetry(
        self,
        trajectory_id: str,
        records: list[dict],
    ):
        """Save all telemetry records for a trajectory."""
        filepath = os.path.join(self.output_dir, f"{trajectory_id}_telemetry.jsonl")
        
        with open(filepath, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
        
        logger.info(f"Saved {len(records)} telemetry records to {filepath}")

    def load_trajectory_telemetry(self, trajectory_id: str) -> list[dict]:
        """Load telemetry records for a trajectory."""
        filepath = os.path.join(self.output_dir, f"{trajectory_id}_telemetry.jsonl")
        
        if not os.path.exists(filepath):
            logger.warning(f"No telemetry file found at {filepath}")
            return []
        
        records = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        
        return records
