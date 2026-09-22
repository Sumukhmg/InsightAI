"""InsightAI Prompt Engineering & Structured Response Schemas
Contains the enterprise system prompt and prompt templates for analytical explanations.
"""

SYSTEM_PROMPT = """You are InsightAI, an enterprise business analytics assistant built for senior retail executives and commercial directors.

You receive verified, deterministic analytical results calculated directly by Python and DuckDB.

STRICT OPERATIONAL RULES:
1. NEVER INVENT NUMBERS: Every metric, percentage, currency figure, and count you state MUST come directly from the verified results provided to you. If a number is not in the verified results, do not state it.
2. NEVER MODIFY NUMBERS: Do not round or alter numbers in ways that change their meaning.
3. NEVER CONFUSE CORRELATION WITH CAUSALITY: Do not assert that X caused Y unless the data explicitly proves it. Use cautious, evidence-grounded phrasing (e.g., "associated with", "coincides with").
4. CLEARLY DISTINGUISH:
   - [FACT]: Observed empirical metric directly present in the data.
   - [PATTERN]: Directional trend, concentration, or statistical distribution.
   - [HYPOTHESIS]: Plausible business rationale explaining why the pattern exists.
   - [RECOMMENDATION]: Actionable commercial decision or investigative step for management.
5. OBJECTIVITY: If the data is ambiguous or does not support a clear conclusion, explicitly state that the evidence is inconclusive.

RESPONSE FORMAT:
You must respond with valid JSON matching the following schema:
{
  "summary": "A concise, high-impact executive summary (2-3 sentences) synthesizing the analytical findings.",
  "key_findings": [
    "[FACT] Specific verified data point 1",
    "[PATTERN] Specific verified pattern 2"
  ],
  "business_implications": [
    "[HYPOTHESIS] Commercial implication or underlying mechanism"
  ],
  "recommendations": [
    "[RECOMMENDATION] Pragmatic, actionable commercial step for stakeholders"
  ],
  "limitations": [
    "Caveat regarding data scope, missing customer IDs, or temporal boundaries"
  ]
}
"""


def format_qa_prompt(query: str, intent: str, verified_metrics: dict) -> str:
    """Formats the prompt for ad-hoc business analyst questions."""
    return f"""A business stakeholder asked the following question:
"{query}"

The analytical engine identified the intent as: {intent}
The Python/DuckDB engine executed deterministic calculations and returned the following VERIFIED RESULTS:

```json
{verified_metrics}
```

Instructions:
1. Ground your answer entirely in the verified results.
2. Structure your answer using the required JSON schema with summary, key_findings, business_implications, recommendations, and limitations.
3. Use the tags [FACT], [PATTERN], [HYPOTHESIS], and [RECOMMENDATION] inside your bullet points.
"""


def format_executive_summary_prompt(kpis: dict, trends: dict, segments: dict) -> str:
    """Formats the prompt for the comprehensive executive overview report."""
    return f"""Please generate a comprehensive Executive Summary for the board of directors based on the following verified business data:

OVERALL BUSINESS KPIS:
```json
{kpis}
```

SALES & REVENUE TRENDS:
```json
{trends}
```

CUSTOMER SEGMENTATION SUMMARY:
```json
{segments}
```

Instructions:
1. Provide a senior executive-level evaluation of overall commercial health, revenue concentration, customer loyalty, and growth momentum.
2. Respond with the required JSON schema.
"""


def format_segment_prompt(segment_name: str, segment_metrics: dict) -> str:
    """Formats prompt to explain customer segments (RFM or K-Means)."""
    return f"""Explain the customer segment '{segment_name}' based on the following verified metrics:

```json
{segment_metrics}
```

Instructions:
1. Describe customer behavior, value contribution, and engagement frequency.
2. Provide targeted retention or upsell recommendations.
3. Respond with the required JSON schema.
"""


def format_anomaly_prompt(anomaly_summary: dict, top_anomalies: list) -> str:
    """Formats prompt to explain statistical anomalies flagged by Isolation Forest."""
    return f"""Explain the following statistically unusual transactions detected by the Isolation Forest model:

SUMMARY OF ANOMALIES:
```json
{anomaly_summary}
```

TOP FLAGGED TRANSACTIONS:
```json
{top_anomalies}
```

Instructions:
1. Clarify that these are 'statistically unusual transactions' based on quantity, price, or order value, NOT confirmed fraud.
2. Formulate hypotheses (e.g., bulk B2B purchases, wholesale clearance, recording corrections).
3. Recommend operational audit checks.
4. Respond with the required JSON schema.
"""


def format_forecast_prompt(forecast_data: dict) -> str:
    """Formats prompt to interpret the 3-month sales forecast."""
    return f"""Interpret the following 3-month sales revenue forecast:

```json
{forecast_data}
```

Instructions:
1. Clearly state the predicted trend direction (Increasing, Stable, Decreasing).
2. Explicitly label this as a predictive statistical model subject to market uncertainty, not a guaranteed fact.
3. Provide inventory and commercial planning recommendations based on the forecast.
4. Respond with the required JSON schema.
"""
