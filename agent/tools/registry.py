"""
Tool registry and built-in tools for the research agent.

Provides a registry for tool discovery and a set of built-in
tools that simulate realistic API interactions for ToolBench tasks.
"""

import json
import time
import logging
import hashlib
from typing import Any, Callable, Optional

import requests
from langchain_core.tools import tool, StructuredTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for tools available to the LangGraph agent.
    
    Tools can be registered statically or loaded from ToolBench
    task specifications.
    """

    def __init__(self):
        self._tools: dict[str, StructuredTool] = {}
        self._tool_metadata: dict[str, dict] = {}
        # Register built-in tools
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in research tools."""
        for t in get_builtin_tools():
            self._tools[t.name] = t
            self._tool_metadata[t.name] = {
                "source": "builtin",
                "description": t.description,
            }

    def register(self, tool_obj: StructuredTool, metadata: Optional[dict] = None):
        """Register a tool with optional metadata."""
        self._tools[tool_obj.name] = tool_obj
        self._tool_metadata[tool_obj.name] = metadata or {
            "source": "custom",
            "description": tool_obj.description,
        }

    def get_tools(self) -> list[StructuredTool]:
        """Get all registered tools as a list."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[StructuredTool]:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def get_tool_names(self) -> list[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())

    def get_metadata(self) -> dict:
        """Get metadata for all tools."""
        return dict(self._tool_metadata)


# ============================================================
# Built-in tools for research experiments
# These simulate realistic API/tool interactions that an agent
# would encounter in ToolBench-style tasks.
# ============================================================

@tool
def web_search(query: str) -> str:
    """Search the web for information about a given query. Returns search results as text."""
    try:
        # Use a real but lightweight search approach
        # For research purposes, we simulate realistic API latency and responses
        time.sleep(0.5)  # Simulate network latency
        
        # Generate a deterministic but varied response based on query
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        return json.dumps({
            "status": "success",
            "query": query,
            "results": [
                {
                    "title": f"Result for: {query}",
                    "snippet": f"Information about {query}. This is a search result containing relevant data.",
                    "url": f"https://example.com/result/{query_hash}"
                }
            ],
            "total_results": 1
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city. Returns weather data as JSON."""
    try:
        time.sleep(0.3)
        # Deterministic weather based on city name
        city_hash = sum(ord(c) for c in city.lower())
        temp = 15 + (city_hash % 25)
        conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
        condition = conditions[city_hash % len(conditions)]
        
        return json.dumps({
            "status": "success",
            "city": city,
            "temperature_celsius": temp,
            "condition": condition,
            "humidity": 40 + (city_hash % 50),
            "wind_speed_kmh": 5 + (city_hash % 30),
        })
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. Returns the result."""
    try:
        time.sleep(0.1)
        # Safe evaluation of mathematical expressions
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sum": sum, "len": len,
        }
        # Only allow basic math operations
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return json.dumps({
            "status": "success",
            "expression": expression,
            "result": result,
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "expression": expression,
            "error": str(e),
        })


@tool
def get_data(endpoint: str, params: Optional[str] = None) -> str:
    """
    Fetch data from an API endpoint. The endpoint should be a descriptive
    name like 'users', 'products', 'orders'. Returns data as JSON.
    """
    try:
        time.sleep(0.4)
        endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
        
        # Generate plausible data based on endpoint type
        data = {
            "status": "success",
            "endpoint": endpoint,
            "data": {
                "id": endpoint_hash,
                "name": f"Item from {endpoint}",
                "description": f"Data retrieved from the {endpoint} endpoint",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        }
        if params:
            data["params_used"] = params
        
        return json.dumps(data)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


@tool
def text_analysis(text: str, analysis_type: str = "summary") -> str:
    """
    Analyze text content. Supported analysis types: summary, sentiment,
    key_phrases, entities. Returns analysis results as JSON.
    """
    try:
        time.sleep(0.3)
        word_count = len(text.split())
        
        result = {
            "status": "success",
            "analysis_type": analysis_type,
            "word_count": word_count,
            "char_count": len(text),
        }
        
        if analysis_type == "summary":
            result["summary"] = text[:200] + "..." if len(text) > 200 else text
        elif analysis_type == "sentiment":
            # Simple heuristic sentiment
            positive_words = {"good", "great", "excellent", "positive", "happy", "best"}
            negative_words = {"bad", "terrible", "worst", "negative", "sad", "poor"}
            words_lower = set(text.lower().split())
            pos = len(words_lower & positive_words)
            neg = len(words_lower & negative_words)
            if pos > neg:
                result["sentiment"] = "positive"
            elif neg > pos:
                result["sentiment"] = "negative"
            else:
                result["sentiment"] = "neutral"
            result["confidence"] = 0.7
        elif analysis_type == "key_phrases":
            words = text.split()
            result["key_phrases"] = list(set(words[:5]))
        else:
            result["analysis"] = f"Analysis of type '{analysis_type}' completed."
        
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)})


def get_builtin_tools() -> list[StructuredTool]:
    """Get all built-in tools."""
    return [web_search, get_weather, calculate, get_data, text_analysis]
