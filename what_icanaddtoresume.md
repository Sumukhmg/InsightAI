# What I Can Add to My Resume — InsightAI

This guide provides concrete, verified resume content, interview talking points, and technical skills derived directly from the real implementation of **InsightAI**.

---

## 📌 Recommended Project Title

### Option 1 (Strongest / Recommended)
**InsightAI — GenAI-Powered Retail Intelligence & Business Analytics Platform**
> **Why this is strongest**: Clearly highlights both the domain (Retail Intelligence) and the modern technical convergence (GenAI + Business Analytics). It positions the project as an enterprise SaaS product rather than an academic script.

### Option 2
**InsightAI — Enterprise Retail Analytics Engine with LLM Business Insights**
> **Why use this**: Emphasizes backend data engineering, analytical rigor, and deterministic computation.

### Option 3
**InsightAI — End-to-End Retail Data Platform (DuckDB, Scikit-Learn, Groq GenAI)**
> **Why use this**: Technology-forward; best for roles filtering heavily on modern stack keywords (DuckDB, LLMs).

---

## ⚡ One-Line Project Description

> *"Architected an enterprise retail intelligence platform utilizing Python, DuckDB, Scikit-Learn, and Groq LLMs to analyze 1M+ real transactions, compute customer RFM clusters, detect transaction anomalies, and synthesize hallucination-free executive insights."*

---

## 📄 Recommended Resume Bullets

Select 3–4 bullets tailored to your target job profile:

### For Analytics / Business Intelligence Roles
- **Architected a high-performance retail analytics platform** on the real UCI Online Retail II dataset (1,067,371 transactions across 2 years), tracking £20.91M in gross volume, 40,077 orders, and 5,878 transacting customer accounts.
- **Engineered an in-memory SQL analytics layer with DuckDB and Pandas**, reducing query latency to < 0.06s across 1M+ records and computing Month-over-Month (MoM) revenue velocity, AOV (£521.84), and catalog performance across 4,916 SKUs.
- **Implemented an automated Data Quality Auditing engine** evaluating missing customer IDs (22.8%), duplicates (12,133 rows), cancellations (19,433 rows), and pricing anomalies, producing a verifiable 90/100 Data Quality Score.

### For Data Science / Machine Learning Roles
- **Developed a multi-tier customer segmentation pipeline** combining heuristic RFM scoring with unsupervised K-Means clustering ($K=4$, $\log(1+x)$ feature transformation, StandardScaler) and PCA 2D projection capturing 95.1% of variance in 0.29s.
- **Implemented Isolation Forest anomaly detection** over multi-dimensional transactional features (Quantity, UnitPrice, Order Revenue), isolating 10,507 statistically unusual transactions without mislabeling operational edge cases as fraud.
- **Constructed an autoregressive monthly revenue forecasting model** utilizing dynamic lag features and Random Forest regressors, generating 3-month predictive scenario horizons with real-time trajectory indicators.

### For GenAI / AI Engineer Roles
- **Designed a zero-hallucination GenAI analytics pipeline** enforcing a strict architectural separation of concerns: Python/DuckDB executes 100% of mathematical calculations, passing only verified analytical metrics to Groq LLMs (`openai/gpt-oss-20b`).
- **Engineered an intent-routing NLP console** that maps natural language queries to deterministic SQL/ML routines, generating structured JSON responses categorized into empirical `[FACT]`, `[PATTERN]`, `[HYPOTHESIS]`, and `[RECOMMENDATION]` tags.
- **Built fault-tolerant LLM resiliency** incorporating schema validation, markdown JSON regex extraction, and deterministic Demo Mode fallback, ensuring zero application downtime during API rate limits or network failures.

---

## 🔑 ATS Keywords (Verified Technologies & Competencies)

```text
Python, DuckDB, Pandas, NumPy, Scikit-Learn, Streamlit, Plotly Express, Plotly Graph Objects, 
PyArrow, Apache Parquet, Groq API, Large Language Models (LLM), Prompt Engineering, 
Intent Classification, Structured JSON Schema, Customer Segmentation, RFM Analysis, 
K-Means Clustering, Principal Component Analysis (PCA), Isolation Forest, Anomaly Detection, 
Time-Series Forecasting, Autoregressive Modeling, Data Preprocessing, Data Quality Auditing, 
SQL Analytics, Business Intelligence (BI), KPI Engineering, Retail Analytics, Pytest
```

---

## 🛠️ Skills Demonstrated

### Technical & Engineering Skills
- **High-Performance Data Storage**: Converting 45.6MB multi-sheet Excel workbooks into columnar Apache Parquet caches (7.2MB), cutting load time from 47s to 0.43s.
- **In-Memory Analytical SQL**: DuckDB relation management, analytical aggregations, and sub-second table scans over 1M+ records.
- **Modular Software Engineering**: Clean folder hierarchy (`src/`, `tests/`, `data/`), singleton database patterns, and strict typing.
- **Automated Testing**: 11 unit tests using `pytest` validating preprocessing filters, KPI math, RFM quintiles, ML clustering, and anomaly scores in 2.20s.

### Analytics & BI Skills
- **Commercial Metric Formulation**: Gross Revenue ($Q \times P$), Invoices, AOV, MoM Revenue Growth %, Geographic Revenue Share, SKU velocity.
- **Data Integrity & Governance**: Explicit audit logging isolating 19,433 cancellations and 6,196 zero/negative prices without silent record dropping.
- **Executive Reporting**: One-click synthesis of strategic board memorandums downloadable as structured Markdown.

### Machine Learning Skills
- **Unsupervised Learning**: K-Means clustering on right-skewed transactional data, feature normalization, and automated persona attribution.
- **Dimensionality Reduction**: PCA 2D projections for cluster visualization and variance explained metrics.
- **Outlier Detection**: Isolation Forest modeling across multi-variate feature spaces (Volume, UnitPrice, Value).
- **Predictive Analytics**: Multi-step recursive forecasting with autoregressive lag features (`Lag_1`, `Lag_2`, rolling 2M window).

### GenAI & LLM Skills
- **Separation of Concerns**: Preventing LLM math hallucinations by decoupling numerical computation from natural language explanation.
- **Intent Detection & Routing**: Pattern-matching NLP router dispatching queries to deterministic analytics functions.
- **Structured Output Parsing**: Forcing LLM JSON schema responses with regex parsing and fallback recovery.
- **Provider Agnosticism**: Pluggable architecture supporting Groq, local Ollama, and deterministic Demo Mode.

### Business & Product Skills
- **Actionable Segmentation Strategy**: Translating cluster centroids (£ spend, purchase frequency, recency days) into targeted commercial tactics (VIP perks, churn win-back).
- **Nuanced Insight Communication**: Enforcing distinction between observed facts, statistical patterns, management hypotheses, and strategic recommendations.
- **User-Centric Enterprise UI**: Glassmorphic Streamlit interface, KPI cards, reactive cross-filtering, and dynamic Plotly charts.

---

## 🎙️ Interview Explanations

### 30-Second Elevator Pitch
> *"I built **InsightAI**, an enterprise retail intelligence platform that solves the biggest problem with GenAI in business analytics: LLM hallucination. Using the real UCI Online Retail II dataset with over 1 million transactions and £20.9M in revenue, I used Python, DuckDB, and Scikit-Learn to calculate 100% of the numbers—from executive KPIs and RFM customer clusters to Isolation Forest anomaly detection and 3-month sales forecasts. I then fed those verified results into Groq LLMs to synthesize executive summaries that strictly categorize observed facts, statistical patterns, hypotheses, and commercial recommendations."*

### 2-Minute Deep Dive
> *"When companies try to connect LLMs directly to databases, two things usually happen: the LLM hallucinates calculations, or it takes 30 seconds to run unoptimized SQL over raw data. For InsightAI, I engineered an architecture that enforces a strict separation of concerns.*
>
> *First, for the data layer, I took the raw UCI Online Retail II dataset—which contains over 1 million transactions across two years—and built a preprocessing pipeline that audits duplicates, cancellations, and invalid unit prices, calculating a transparent 90/100 Data Quality Score. To make the app fast, I cached the cleaned dataset into an Apache Parquet file, reducing load time from 47 seconds to 0.43 seconds, and registered it into an in-memory DuckDB database for sub-second analytical queries.*
>
> *Second, on the machine learning side, I implemented RFM behavioral segmentation and unsupervised K-Means clustering. Because retail spend is heavily right-skewed, I applied a log transformation and standard scaling before clustering, and used PCA to project clusters into 2D with 95.1% explained variance. I also implemented Isolation Forest to flag statistically unusual transactions across quantity and price, and built a 3-month autoregressive revenue forecast.*
>
> *Finally, for the GenAI layer, I built an Intent Router. When a business user asks a question like 'Which countries generated the most revenue?', the pipeline identifies the intent, executes the exact DuckDB aggregation, verifies the facts, and passes only those verified metrics to the Groq API. Groq responds in a validated JSON schema that categorizes findings into Facts, Patterns, Hypotheses, and Recommendations. If the API key is missing or fails, the system seamlessly falls back to a deterministic Demo Mode without crashing.*
>
> *The entire codebase is modular, fully tested with 11 passing Pytest unit tests, and packaged into a responsive, enterprise-grade Streamlit application."*

---

## 🌟 Strongest Talking Points by Role

### For Analytics / BI Roles
1. **Zero Silent Data Deletion**: Emphasize how the preprocessing pipeline explicitly logs and counts every removed cancellation (19,433) and negative price (6,196), producing an auditable Data Quality Score (90/100).
2. **Deterministic Integrity**: Explain why you never let the LLM calculate metrics like Average Order Value (£521.84) or MoM Growth %, preventing executive mistrust.
3. **Columnar Acceleration**: Discuss how converting raw Excel into Parquet and DuckDB allows sub-second analytical scans across 1M+ rows.
4. **Actionable Personas**: Highlight how RFM segments were translated into distinct business actions (e.g. VIP loyalty programs vs. churn risk outreach).
5. **Cross-Filtering UX**: Walk through how the Streamlit dashboard provides coordinated date range and country filtering across all charts.

### For Data Science / ML Roles
1. **Handling Real-World Skew**: Discuss applying $\log(1+x)$ transformations before K-Means clustering to prevent high-spend wholesale outliers from distorting cluster centroids.
2. **PCA Dimensionality Reduction**: Explain using PCA to validate that 2 components explained 95.1% of the feature variance in the customer landscape.
3. **Responsible Anomaly Detection**: Highlight why you used Isolation Forest across Quantity, UnitPrice, and Order Value, and why you explicitly labeled flagged records as "statistically unusual" rather than making unsubstantiated fraud claims.
4. **Autoregressive Feature Engineering**: Discuss building multi-step lag features (`Lag_1`, `Lag_2`, rolling window) for monthly forecasting and evaluating in-sample MAPE.
5. **Automated Unit Testing**: Mention your 11 Pytest tests covering data quality math, RFM quintile assignment, and model stability.

### For AI / GenAI Roles
1. **Separation of Computation from Narration**: The core design thesis: Python computes the ground truth; the LLM provides contextual explanation.
2. **Intent Routing Architecture**: How regex and semantic intent classification dynamically dispatch questions to verified Python functions.
3. **Structured JSON Enforcement**: Using system prompt schema boundaries, markdown extraction regex, and schema validation.
4. **Resiliency & Demo Mode**: The application never crashes—it gracefully recovers from network drops, rate limits, or missing keys with high-quality fallback logic.
5. **Groq Inference Speed**: Why you utilized Groq's LPUs (`openai/gpt-oss-20b`) for near-instantaneous executive response generation.

### For Product / Strategy Roles
1. **Executive Communication**: The four-tier insight framework: distinguishing what the data proves (`[FACT]`) from statistical trends (`[PATTERN]`), commercial theories (`[HYPOTHESIS]`), and actions (`[RECOMMENDATION]`).
2. **SaaS-Grade UX**: Designing modern glassmorphic cards, contextual badges, expandable methodology disclaimers, and loading feedback.
3. **Self-Service Ad-Hoc Analytics**: How the suggested question pills empower non-technical executives to explore complex data without writing SQL.
4. **One-Click Board Memorandums**: How the Executive Report generator automates hours of manual slide deck preparation into downloadable Markdown.
5. **Data Governance Disclaimers**: Incorporating model limitations (e.g., guest checkouts, forecasting assumptions) to foster executive trust.

---

## 🛑 Claims You Should NOT Make (Protecting Your Credibility)

To ensure your resume and interview answers remain 100% credible and defensible under rigorous technical probing, **DO NOT** make the following claims:

1. **❌ Do NOT claim real-time streaming analytics**:
   - *Why*: The dataset is historical batch transaction data (2009–2011). You used high-speed in-memory batch processing (DuckDB/Parquet), not Kafka, Flink, or real-time WebSockets.
   - *What to say instead*: *"High-performance in-memory batch analytics with sub-second query latency across 1M+ records."*

2. **❌ Do NOT claim production cloud deployment or Kubernetes orchestration**:
   - *Why*: The application is architected for local or Streamlit Cloud execution; you did not build a multi-node Kubernetes cluster or CI/CD Docker swarm.
   - *What to say instead*: *"Packaged as a lightweight, deployable Streamlit application with modular dependencies and automated Pytest test coverage."*

3. **❌ Do NOT claim causal inference from correlations**:
   - *Why*: The ML models identify correlation and statistical proximity (K-Means, Isolation Forest), not randomized control trials (A/B testing) or causal DAGs.
   - *What to say instead*: *"Identified behavioral associations and directional patterns while strictly categorizing commercial hypotheses as distinct from verified facts."*

4. **❌ Do NOT claim healthcare or pharmaceutical expertise**:
   - *Why*: UCI Online Retail II is a British non-store gift/retail dataset. It contains no clinical trial, genomic, or patient data.
   - *What to say instead*: *"Domain-agnostic commercial transaction analytics demonstrated on retail e-commerce operations."*

5. **❌ Do NOT claim an autonomous multi-agent framework (e.g., AutoGen, CrewAI)**:
   - *Why*: You built a deterministic intent-routing pipeline with Groq LLM synthesis, not an autonomous agent swarm negotiating tasks.
   - *What to say instead*: *"An intent-routed GenAI analytics pipeline with deterministic Python/DuckDB tool execution."*

6. **❌ Do NOT claim enterprise MLOps (e.g., MLflow, Kubeflow, automated drift retraining)**:
   - *Why*: You utilized Scikit-learn for deterministic on-demand model execution, not an automated model registry or live drift-monitoring infrastructure.
   - *What to say instead*: *"Modular machine learning pipelines with reproducible parameterization, standardized feature scaling, and unit testing."*
