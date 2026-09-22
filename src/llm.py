"""InsightAI LLM Integration Module
Manages interactions with Groq API, local Ollama, and deterministic Demo Mode fallback.
Validates structured JSON output and guarantees application stability without API keys.
"""

from typing import Any, Dict, List, Optional
import os
import json
import re
from dotenv import load_dotenv
import requests

from src.prompts import (
    SYSTEM_PROMPT,
    format_qa_prompt,
    format_executive_summary_prompt,
    format_segment_prompt,
    format_anomaly_prompt,
    format_forecast_prompt
)

load_dotenv()


class LLMClient:
    """Enterprise LLM Client supporting Groq, Ollama, and robust Demo Mode."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ):
        # 1. Explicit arguments
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.provider = (provider or "").strip().lower()

        # 2. Check Streamlit secrets (Streamlit Community Cloud)
        try:
            import streamlit as st
            if not self.api_key and "GROQ_API_KEY" in st.secrets:
                self.api_key = str(st.secrets["GROQ_API_KEY"]).strip()
            if not self.model and "GROQ_MODEL" in st.secrets:
                self.model = str(st.secrets["GROQ_MODEL"]).strip()
            if not self.provider and "LLM_PROVIDER" in st.secrets:
                self.provider = str(st.secrets["LLM_PROVIDER"]).strip().lower()
        except Exception:
            pass

        # 3. Fallback to os.getenv / defaults
        if not self.api_key:
            self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not self.model:
            self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
        if not self.provider:
            self.provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def is_configured(self) -> bool:
        """Returns True if the LLM provider is properly configured."""
        if self.provider == "groq":
            return bool(self.api_key and len(self.api_key) > 5)
        elif self.provider == "ollama":
            return True
        return False

    def generate_response(self, user_prompt: str) -> Dict[str, Any]:
        """Calls the configured LLM provider and parses structured JSON output."""
        if not self.is_configured():
            return self._build_demo_response(user_prompt)

        try:
            if self.provider == "groq":
                raw_text = self._call_groq(user_prompt)
            elif self.provider == "ollama":
                raw_text = self._call_ollama(user_prompt)
            else:
                return self._build_demo_response(user_prompt)

            return self._parse_and_validate_json(raw_text)
        except Exception as e:
            # Graceful fallback on any network/API exception
            demo_fallback = self._build_demo_response(user_prompt)
            demo_fallback["api_notice"] = f"Live LLM call failed ({str(e)}). Displaying rule-based analytical insights."
            return demo_fallback

    def _call_groq(self, user_prompt: str) -> str:
        """Calls Groq Chat Completions endpoint via HTTP request."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1500,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Groq API Error {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _call_ollama(self, user_prompt: str) -> str:
        """Calls local Ollama instance."""
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.model or "llama3.2",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "format": "json"
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama Error {response.status_code}: {response.text}")
        data = response.json()
        return data.get("message", {}).get("content", "")

    def _parse_and_validate_json(self, raw_text: str) -> Dict[str, Any]:
        """Validates and extracts structured JSON response from LLM output."""
        try:
            # 1. Direct JSON parse
            data = json.loads(raw_text)
        except Exception:
            # 2. Extract JSON block inside markdown ```json ... ```
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(0))
                except Exception:
                    data = None
            else:
                data = None

        if isinstance(data, dict) and "summary" in data:
            return {
                "summary": str(data.get("summary", "")),
                "key_findings": list(data.get("key_findings", [])),
                "business_implications": list(data.get("business_implications", [])),
                "recommendations": list(data.get("recommendations", [])),
                "limitations": list(data.get("limitations", [])),
                "raw_text": raw_text,
                "is_fallback": False
            }

        # 3. Graceful text fallback if structure missing
        return {
            "summary": raw_text[:300] + "..." if len(raw_text) > 300 else raw_text,
            "key_findings": ["[FACT] " + line.strip() for line in raw_text.split("\n") if line.strip()][:3],
            "business_implications": ["[HYPOTHESIS] Strategic review recommended based on available trends."],
            "recommendations": ["[RECOMMENDATION] Audit commercial drivers against verified operational metrics."],
            "limitations": ["Output could not be fully parsed into structured JSON format."],
            "raw_text": raw_text,
            "is_fallback": True
        }

    def _build_demo_response(self, user_prompt: str) -> Dict[str, Any]:
        """Provides high-quality deterministic insights when running in Demo Mode without API key."""
        return {
            "summary": (
                "Verified analytical calculations completed successfully. AI insights running in Demo Mode. "
                "Configure your GROQ_API_KEY in the sidebar to activate live Groq LLM evaluations."
            ),
            "key_findings": [
                "[FACT] All numbers, charts, and metrics shown on screen are computed directly from the dataset using DuckDB.",
                "[PATTERN] Consistent demand is concentrated across the United Kingdom and top recurring retail product lines.",
                "[FACT] Preprocessing audit verified that all cancellations and invalid prices were isolated prior to metric aggregation."
            ],
            "business_implications": [
                "[HYPOTHESIS] Regional concentration reflects core domestic customer base, with untapped export opportunities in key EU territories."
            ],
            "recommendations": [
                "[RECOMMENDATION] Add your GROQ_API_KEY in the application settings to receive dynamic GenAI strategic summaries."
            ],
            "limitations": [
                "Demo Mode active: Rule-based fallback generated without live LLM inference."
            ],
            "is_fallback": True,
            "is_demo_mode": True
        }

    # High-Level Analytical Assistant Methods
    def generate_business_insight(self, query: str, intent: str, verified_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generates business insight for user natural-language questions."""
        prompt = format_qa_prompt(query, intent, verified_metrics)
        return self.generate_response(prompt)

    def generate_executive_summary(self, kpis: Dict[str, Any], trends: Dict[str, Any], segments: Dict[str, Any]) -> Dict[str, Any]:
        """Generates an executive board summary."""
        prompt = format_executive_summary_prompt(kpis, trends, segments)
        return self.generate_response(prompt)

    def explain_segment(self, segment_name: str, segment_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Explains customer segment behaviors."""
        prompt = format_segment_prompt(segment_name, segment_metrics)
        return self.generate_response(prompt)

    def explain_anomaly(self, anomaly_summary: Dict[str, Any], top_anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Explains Isolation Forest anomalies."""
        prompt = format_anomaly_prompt(anomaly_summary, top_anomalies)
        return self.generate_response(prompt)

    def explain_forecast(self, forecast_data: Dict[str, Any]) -> Dict[str, Any]:
        """Explains 3-month sales forecast."""
        prompt = format_forecast_prompt(forecast_data)
        return self.generate_response(prompt)
