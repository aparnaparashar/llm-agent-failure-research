"""
Ollama LLM integration for the failure research agent.

Provides a configurable connection to Ollama-hosted models.
All experiments must record: llm_provider, llm_model, model_version, temperature, prompt_version.
"""

import os
import time
import logging
from typing import Optional

import yaml
import requests
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# Default configuration path
DEFAULT_LLM_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "llm.yaml"
)


class OllamaConfig:
    """Configuration for the Ollama LLM connection."""

    def __init__(
        self,
        provider: str = "ollama",
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 2,
        prompt_version: str = "v1.0",
    ):
        # Environment variables override config file values
        self.provider = os.environ.get("LLM_PROVIDER", provider)
        self.model = os.environ.get("LLM_MODEL", model)
        self.base_url = os.environ.get("OLLAMA_BASE_URL", base_url)
        self.temperature = float(os.environ.get("LLM_TEMPERATURE", temperature))
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.prompt_version = prompt_version
        self.model_version: Optional[str] = None  # Populated after connection

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "OllamaConfig":
        """Load configuration from YAML file."""
        path = config_path or DEFAULT_LLM_CONFIG
        path = os.path.abspath(path)

        if not os.path.exists(path):
            logger.warning(f"Config file not found at {path}, using defaults.")
            return cls()

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        llm_config = config.get("llm", {})
        return cls(
            provider=llm_config.get("provider", "ollama"),
            model=llm_config.get("model", "qwen3:8b"),
            base_url=llm_config.get("base_url", "http://localhost:11434"),
            temperature=llm_config.get("temperature", 0.0),
            max_tokens=llm_config.get("max_tokens", 4096),
            timeout=llm_config.get("timeout", 120),
            max_retries=llm_config.get("max_retries", 2),
            prompt_version=config.get("prompt_version", "v1.0"),
        )

    def to_dict(self) -> dict:
        """Serialize config for experiment logging."""
        return {
            "llm_provider": self.provider,
            "llm_model": self.model,
            "model_version": self.model_version,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "prompt_version": self.prompt_version,
        }


class OllamaLLM:
    """Manages the Ollama LLM connection and provides experiment metadata."""

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig.from_yaml()
        self._llm: Optional[ChatOllama] = None
        self._verified = False

    def verify_connection(self) -> dict:
        """
        Verify that Ollama is reachable and the model is available.
        Returns a dict with connection status details.
        """
        result = {
            "ollama_reachable": False,
            "model_available": False,
            "model_version": None,
            "test_response": None,
            "error": None,
        }

        # Check Ollama server
        try:
            resp = requests.get(
                f"{self.config.base_url}/api/tags",
                timeout=10,
            )
            resp.raise_for_status()
            result["ollama_reachable"] = True

            # Check model availability
            models_data = resp.json()
            available_models = [
                m.get("name", "") for m in models_data.get("models", [])
            ]

            # Match model name (with or without tag)
            model_name = self.config.model
            model_found = False
            model_details = None
            for m in models_data.get("models", []):
                name = m.get("name", "")
                if name == model_name or name.startswith(model_name.split(":")[0]):
                    model_found = True
                    model_details = m
                    break

            result["model_available"] = model_found
            if model_details:
                result["model_version"] = model_details.get("digest", "unknown")[:12]
                self.config.model_version = result["model_version"]

            if not model_found:
                result["error"] = (
                    f"Model '{model_name}' not found. "
                    f"Available: {available_models}"
                )
                return result

        except requests.ConnectionError:
            result["error"] = (
                f"Cannot connect to Ollama at {self.config.base_url}. "
                "Is Ollama running? Try: ollama serve"
            )
            return result
        except Exception as e:
            result["error"] = f"Ollama connection error: {e}"
            return result

        # Test with a simple prompt
        try:
            llm = self._create_llm()
            start_time = time.time()
            response = llm.invoke("Say 'hello' in one word.")
            elapsed = time.time() - start_time
            result["test_response"] = {
                "content": response.content[:200],
                "latency_seconds": round(elapsed, 3),
            }
            self._verified = True
        except Exception as e:
            result["error"] = f"LLM test invocation failed: {e}"

        return result

    def _create_llm(self) -> ChatOllama:
        """Create a ChatOllama instance."""
        # Disable Qwen3 thinking mode for faster, deterministic responses.
        # Thinking mode generates very long internal reasoning chains that
        # slow execution and make telemetry noisy. For research experiments,
        # we want direct tool-use behavior.
        return ChatOllama(
            model=self.config.model,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            timeout=self.config.timeout,
            extra_body={"options": {"num_ctx": 8192}},
        )

    def get_llm(self) -> ChatOllama:
        """Get the ChatOllama instance, creating it if necessary."""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm

    def get_metadata(self) -> dict:
        """Get metadata for experiment logging."""
        return self.config.to_dict()


def create_ollama_llm(config_path: Optional[str] = None) -> OllamaLLM:
    """Factory function to create an OllamaLLM instance from config."""
    config = OllamaConfig.from_yaml(config_path)
    return OllamaLLM(config)
