"""Quick test: verify Ollama connectivity and LLM response."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.llm.ollama import OllamaLLM, OllamaConfig

config = OllamaConfig()
ollama = OllamaLLM(config)

print("Verifying Ollama connection...")
result = ollama.verify_connection()

print("Reachable:", result["ollama_reachable"])
print("Model available:", result["model_available"])

if result.get("test_response"):
    resp = result["test_response"]
    print("Response:", resp["content"][:200])
    print("Latency:", resp["latency_seconds"], "seconds")

if result.get("error"):
    print("ERROR:", result["error"])
    sys.exit(1)

print("\nSTEP 2 PASSED: Ollama connectivity verified.")
