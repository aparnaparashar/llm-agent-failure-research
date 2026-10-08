"""
ToolBench benchmark loader and task generator.
Provides realistic multi-step tool-use tasks for experimental evaluation.
Supports parametric task instance generation across multiple categories.
"""

import random
from typing import Dict, Any, List, Optional


class ToolBenchTaskSuite:
    """Suite of realistic multi-step tool-use tasks across categories."""

    TASKS = [
        # Category: Information Retrieval & Analysis
        {
            "task_id": "tb_info_01",
            "category": "information_retrieval",
            "description": "Search for the latest innovations in quantum computing. Analyze the text to identify top keywords and calculate an index score based on keyword density.",
            "expected_tools": ["web_search", "text_analysis", "calculate"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_info_02",
            "category": "information_retrieval",
            "description": "Search for recent space missions to Mars. Extract the launch dates and calculate the travel duration in months.",
            "expected_tools": ["web_search", "calculate"],
            "difficulty": "easy",
        },
        {
            "task_id": "tb_info_03",
            "category": "information_retrieval",
            "description": "Search for artificial intelligence breakthrough papers in 2025. Perform sentiment analysis on public reception.",
            "expected_tools": ["web_search", "text_analysis"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_info_04",
            "category": "information_retrieval",
            "description": "Search for global electric vehicle market shares. Calculate the combined percentage of the top 3 manufacturers.",
            "expected_tools": ["web_search", "calculate"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_info_05",
            "category": "information_retrieval",
            "description": "Search for renewable energy installation targets by 2030. Summarize key technical bottlenecks.",
            "expected_tools": ["web_search", "text_analysis"],
            "difficulty": "easy",
        },
        # Category: Geospatial & Meteorology
        {
            "task_id": "tb_geo_01",
            "category": "geospatial",
            "description": "What is the weather in Seattle and Miami? Calculate the difference in temperature and determine if either city has humidity over 70%.",
            "expected_tools": ["get_weather", "calculate"],
            "difficulty": "easy",
        },
        {
            "task_id": "tb_geo_02",
            "category": "geospatial",
            "description": "Get the weather forecast for Berlin, London, and Madrid. Calculate the average temperature across all three European cities.",
            "expected_tools": ["get_weather", "calculate"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_geo_03",
            "category": "geospatial",
            "description": "Check the current weather in Tokyo and Sydney. Calculate the temperature difference and convert Celsius to Fahrenheit.",
            "expected_tools": ["get_weather", "calculate"],
            "difficulty": "easy",
        },
        {
            "task_id": "tb_geo_04",
            "category": "geospatial",
            "description": "Fetch weather in Chicago, Boston, and Toronto. Calculate the minimum temperature and summarize wind conditions.",
            "expected_tools": ["get_weather", "calculate", "text_analysis"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_geo_05",
            "category": "geospatial",
            "description": "Check the temperature in Cairo and Dubai. Calculate whether the difference is within 5 degrees.",
            "expected_tools": ["get_weather", "calculate"],
            "difficulty": "easy",
        },
        # Category: E-Commerce & Financial Operations
        {
            "task_id": "tb_fin_01",
            "category": "ecommerce_finance",
            "description": "Get data from the 'products' API with params 'category=electronics'. Then search for reviews for the top product and calculate the discounted price with a 20% off coupon.",
            "expected_tools": ["get_data", "web_search", "calculate"],
            "difficulty": "hard",
        },
        {
            "task_id": "tb_fin_02",
            "category": "ecommerce_finance",
            "description": "Retrieve financial metrics from 'market_summary' endpoint. Calculate the return on investment (ROI) if initial capital was 10000 and final valuation was 14500.",
            "expected_tools": ["get_data", "calculate"],
            "difficulty": "easy",
        },
        {
            "task_id": "tb_fin_03",
            "category": "ecommerce_finance",
            "description": "Query the 'transactions' endpoint for customer accounts. Calculate total revenue and average transaction value.",
            "expected_tools": ["get_data", "calculate"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_fin_04",
            "category": "ecommerce_finance",
            "description": "Retrieve data from 'quarterly_earnings' endpoint. Calculate profit margin given revenue 500000 and expense 380000.",
            "expected_tools": ["get_data", "calculate"],
            "difficulty": "easy",
        },
        {
            "task_id": "tb_fin_05",
            "category": "ecommerce_finance",
            "description": "Fetch product listings from 'catalog' with params 'sort=price'. Calculate discount savings if buying 3 units at 15% off.",
            "expected_tools": ["get_data", "calculate"],
            "difficulty": "medium",
        },
        # Category: Data Synthesis & Text Processing
        {
            "task_id": "tb_synth_01",
            "category": "synthesis",
            "description": "Query the 'user_feedback' endpoint, perform text analysis to gauge sentiment polarity, and calculate the proportion of positive comments.",
            "expected_tools": ["get_data", "text_analysis", "calculate"],
            "difficulty": "hard",
        },
        {
            "task_id": "tb_synth_02",
            "category": "synthesis",
            "description": "Search for renewable energy adoption statistics, summarize the main challenges, and calculate the required growth rate from 20% to 50% over 5 years.",
            "expected_tools": ["web_search", "text_analysis", "calculate"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_synth_03",
            "category": "synthesis",
            "description": "Query the 'system_logs' endpoint, analyze error log messages for summary clusters, and calculate the error frequency per hour.",
            "expected_tools": ["get_data", "text_analysis", "calculate"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_synth_04",
            "category": "synthesis",
            "description": "Search for healthcare automation case studies, perform key phrase analysis, and summarize core clinical benefits.",
            "expected_tools": ["web_search", "text_analysis"],
            "difficulty": "medium",
        },
        {
            "task_id": "tb_synth_05",
            "category": "synthesis",
            "description": "Search for supply chain resilience strategies, analyze key risk factors, and calculate cost efficiency tradeoffs.",
            "expected_tools": ["web_search", "text_analysis", "calculate"],
            "difficulty": "hard",
        },
    ]

    CITIES = [
        "Seattle", "Miami", "Berlin", "London", "Madrid", "Tokyo", "Sydney",
        "Paris", "Toronto", "Chicago", "Boston", "San Francisco", "Rome", "Singapore",
        "Dubai", "Cairo", "Seoul", "Amsterdam", "Stockholm", "Vienna"
    ]

    QUERIES = [
        "quantum computing breakthroughs", "space exploration Mars rovers",
        "autonomous driving safety benchmarks", "clean fusion energy progress",
        "large language model agent architectures", "biotech CRISPR therapeutics",
        "semiconductor manufacturing lithography", "global logistics supply chain bottlenecks"
    ]

    ENDPOINTS = ["products", "market_summary", "user_feedback", "transactions", "system_logs", "catalog", "financials"]

    @classmethod
    def get_all_tasks(cls) -> List[Dict[str, Any]]:
        return cls.TASKS.copy()

    @classmethod
    def get_tasks_by_category(cls, category: str) -> List[Dict[str, Any]]:
        return [t for t in cls.TASKS if t["category"] == category]

    @classmethod
    def get_task_by_id(cls, task_id: str) -> Optional[Dict[str, Any]]:
        for t in cls.TASKS:
            if t["task_id"] == task_id:
                return t
        return None

    @classmethod
    def generate_task_instance(cls, index: int, seed: int = 42) -> Dict[str, Any]:
        """
        Deterministically generates a rich, valid ToolBench task instance for batch scaling.
        Ensures variety in tools, prompts, arguments, and expected behavior.
        """
        rng = random.Random(seed + index * 997)
        base_task = cls.TASKS[index % len(cls.TASKS)]
        cat = base_task["category"]

        if cat == "geospatial":
            c1, c2 = rng.sample(cls.CITIES, 2)
            desc = f"What is the weather in {c1} and {c2}? Calculate the temperature difference and evaluate humidity."
            task_id = f"tb_geo_{index:04d}"
            tools = ["get_weather", "calculate"]
        elif cat == "ecommerce_finance":
            ep = rng.choice(cls.ENDPOINTS)
            val1 = rng.randint(5000, 50000)
            val2 = int(val1 * rng.uniform(1.1, 1.6))
            desc = f"Retrieve data from endpoint '{ep}'. Calculate the growth return if initial balance was {val1} and final was {val2}."
            task_id = f"tb_fin_{index:04d}"
            tools = ["get_data", "calculate"]
        elif cat == "information_retrieval":
            q = rng.choice(cls.QUERIES)
            desc = f"Search the web for '{q}'. Summarize the key findings and evaluate numerical trends."
            task_id = f"tb_info_{index:04d}"
            tools = ["web_search", "text_analysis", "calculate"]
        else:
            ep = rng.choice(cls.ENDPOINTS)
            desc = f"Query endpoint '{ep}' and perform text analysis to identify key topics. Calculate average length."
            task_id = f"tb_synth_{index:04d}"
            tools = ["get_data", "text_analysis", "calculate"]

        return {
            "task_id": task_id,
            "category": cat,
            "description": desc,
            "expected_tools": tools,
            "difficulty": base_task.get("difficulty", "medium"),
            "seed": seed,
            "instance_index": index,
        }
