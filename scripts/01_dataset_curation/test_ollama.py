"""
Test Ollama connectivity and model availability.

STEP 2 verification: Ensures the LangGraph agent can communicate
with the Ollama LLM backend.

Expected output:
  - Ollama reachable: True
  - Model available: True
  - LLM response received: True
"""

import sys
import os
import json
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm.ollama import OllamaLLM, OllamaConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    print("=" * 60)
    print("STEP 2: Ollama Connectivity Test")
    print("=" * 60)

    # Load config
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "llm.yaml"
    )
    
    print(f"\nConfig path: {config_path}")
    
    if os.path.exists(config_path):
        config = OllamaConfig.from_yaml(config_path)
    else:
        print("Config file not found, using defaults.")
        config = OllamaConfig()

    print(f"Provider: {config.provider}")
    print(f"Model: {config.model}")
    print(f"Base URL: {config.base_url}")
    print(f"Temperature: {config.temperature}")

    # Create LLM and verify
    ollama = OllamaLLM(config)
    result = ollama.verify_connection()

    print(f"\n--- Connection Test Results ---")
    print(f"Ollama reachable: {result['ollama_reachable']}")
    print(f"Model available:  {result['model_available']}")
    print(f"Model version:    {result.get('model_version', 'N/A')}")

    if result.get("test_response"):
        resp = result["test_response"]
        print(f"Test response:    {resp['content'][:100]}")
        print(f"Latency:          {resp['latency_seconds']}s")
        print(f"\n[OK] LLM response received: True")
    else:
        print(f"\n[FAIL] LLM response received: False")

    if result.get("error"):
        print(f"\n[FAIL] ERROR: {result['error']}")
        sys.exit(1)

    # Print metadata
    print(f"\n--- Experiment Metadata ---")
    metadata = ollama.get_metadata()
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("STEP 2 PASSED: Ollama connectivity verified.")
    print("=" * 60)


if __name__ == "__main__":
    main()
