  Predictive Analytics Engine — System Architecture

  Overview

  A three-layer system: Ingestion → Intelligence → Output. The engine is schema-first and model-agnostic. A user uploads a dataset, asks a question in plain English,
  and the engine handles everything else.

  ---
  Layer 1 — Data Ingestion & Profiling

  ┌─────────────────────────────────────────────────────┐
  │                  Ingestion Layer                    │
  │                                                     │
  │  File Upload (CSV / Excel)                          │
  │        │                                            │
  │  Schema Profiler                                    │
  │    - column names, dtypes, null rates               │
  │    - cardinality, distributions                     │
  │    - numeric ranges, categorical levels             │
  │    - time columns, ID columns (auto-detect)         │
  │        │                                            │
  │  Schema Manifest (JSON)  ←── passed to agents       │
  └─────────────────────────────────────────────────────┘

  Key point: the profiler reads the full dataset once on upload to build the manifest. After that, the engine reasons entirely from the manifest — it does not re-scan
  rows for each question.

  ---
  Layer 2 — Intelligence (Multi-Agent Core)

  ┌─────────────────────────────────────────────────────────────┐
  │                    Intelligence Layer                       │
  │                                                             │
  │  User Question (natural language)                           │
  │        │                                                    │
  │  ┌─────▼──────────┐                                         │
  │  │ Intent Parser  │  "What predicts student stop-out?"      │
  │  │  (Claude API)  │  → task_type: classification            │
  │  └─────┬──────────┘    target: stop_out_flag                │
  │        │               candidate_features: [gpa, attend…]   │ 
  │        │                                                    │
  │  ┌─────▼──────────┐                                         │
  │  │ Model Selector │  schema manifest + task_type            │
  │  │    Agent       │  → selects: XGBoost + SHAP              │
  │  └─────┬──────────┘    (from model knowledge base)          │
  │        │                                                    │
  │  ┌─────▼──────────┐                                         │
  │  │  Code Generator│  dynamically writes Python              │
  │  │    Agent       │  preprocessing + model + eval code      │
  │  └─────┬──────────┘                                         │
  │        │                                                    │
  │  ┌─────▼──────────┐                                         │
  │  │ Execution      │  runs code in sandboxed env             │
  │  │ Sandbox        │  returns metrics + feature importances  │
  │  └─────┬──────────┘                                         │
  │        │                                                    │
  │  ┌─────▼──────────┐                                         │
  │  │ Interpreter    │  metrics + SHAP → plain English         │
  │  │    Agent       │  "Faculty support is the strongest      │
  │  │  (Claude API)  │   predictor of stop-out…"               │
  │  └────────────────┘                                         │
  └─────────────────────────────────────────────────────────────┘

  Model knowledge base — a registry mapping (task_type, data_shape, target_type) → ranked model list:

  ┌────────────────────────────────────────────────┬───────────────────────────────────────┐
  │                    Scenario                    │                Models                 │
  ├────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Binary classification, tabular, mixed features │ XGBoost → Logistic Regression         │
  ├────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Multi-class, high cardinality                  │ XGBoost → Random Forest               │
  ├────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Continuous target, regression                  │ Linear Regression → Gradient Boosting │
  ├────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Time series / forecasting                      │ Prophet → ARIMA                       │
  ├────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Cluster discovery, no target                   │ KMeans → DBSCAN                       │
  ├────────────────────────────────────────────────┼───────────────────────────────────────┤
  │ Survey construct analysis                      │ Factor Analysis → PCA                 │
  └────────────────────────────────────────────────┴───────────────────────────────────────┘

  ---
  Layer 3 — Output

  ┌────────────────────────────────────────┐
  │              Output Layer              │
  │                                        │
  │  Structured JSON result                │
  │    - model_used, metrics               │
  │    - feature_importances (SHAP)        │
  │    - risk_scores per student (if appl) │
  │    - sub-group breakdowns              │
  │                                        │
  │  Natural language narrative            │
  │    - what was found                    │
  │    - what it means                     │
  │    - recommended actions               │
  │                                        │
  │  Visualizations (optional)             │
  │    - SHAP summary plot                 │
  │    - confusion matrix                  │
  │    - risk distribution histogram       │
  └────────────────────────────────────────┘

  ---
  Tech Stack

  ┌────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
  │       Component        │                                 Technology                                  │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ Backend API            │ Python, FastAPI                                                             │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ ML / Stats             │ scikit-learn, XGBoost, statsmodels, prophet                                 │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ Explainability         │ SHAP                                                                        │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ Data handling          │ pandas, openpyxl                                                            │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ Intelligence agents    │ Claude API (claude-sonnet-4-6)                                              │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ Code execution sandbox │ subprocess + tempfile isolation (short term); Docker container (production) │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ Frontend               │ To be decided — can start with Streamlit for internal use                   │
  ├────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ Storage                │ Local filesystem (dev), S3-compatible (production)                          │
  └────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘

  ---
  Data Flow — End to End

  User uploads CSV/Excel
          │
          ▼
  Schema Profiler → Schema Manifest (JSON)
          │
          ▼
  User asks: "Which students are most at risk of stopping out?"
          │
          ▼
  Intent Parser → { task: classification, target: stop_out_flag }
          │
          ▼
  Model Selector → XGBoost + SHAP (from knowledge base)
          │
          ▼
  Code Generator → Python script (preprocessing + fit + eval)
          │
          ▼
  Execution Sandbox → { accuracy: 0.98, roc_auc: 0.999, shap_values: [...] }
          │
          ▼
  Interpreter Agent → Plain English narrative + ranked student risk list
          │
          ▼
  User sees results

  ---
  File Structure

  predictive-analytics-engine/
  ├── api/
  │   ├── main.py               # FastAPI app, routes
  │   └── models.py             # Pydantic schemas
  ├── engine/
  │   ├── profiler.py           # Schema profiler
  │   ├── intent_parser.py      # Claude API: question → task spec
  │   ├── model_selector.py     # Knowledge base + selector logic
  │   ├── code_generator.py     # Claude API: task spec → Python code
  │   ├── executor.py           # Sandboxed code runner
  │   └── interpreter.py        # Claude API: results → narrative
  ├── models/
  │   └── knowledge_base.yaml   # Model registry (task → model mapping)
  ├── data/
  │   └── synthetic/            # Synthetic datasets per use case
  ├── tests/
  │   └── use_cases/            # 15+ use case regression tests
  └── notebooks/                # Exploratory work (existing .ipynb files)

  ---
  Key Design Decisions

  1. Schema manifest is the contract — agents communicate via the manifest, not raw data. This makes the engine dataset-agnostic and fast.
  2. Claude API for 3 steps, not 1 — Intent parsing, code generation, and result interpretation each get a separate, focused prompt. Smaller prompts = cheaper, faster,
  more reliable.
  3. Execution isolation — generated code runs in a subprocess with no network access and a timeout. This is the highest-risk component and must be hardened before
  production.
  4. XGBoost + SHAP as default for tabular classification — validated by the existing notebook (0.98 accuracy, 0.999 AUC). SHAP replaces manual factor analysis for
  explainability.
  5. Logistic regression as the interpretable fallback — when stakeholders need a model they can audit or present to compliance teams, the selector drops to logistic
  regression and outputs odds ratios.

  ---
  Immediate Build Order

  1. profiler.py — schema manifest from any CSV/Excel
  2. knowledge_base.yaml — model registry with the 5 scenarios above
  3. model_selector.py — deterministic selector from manifest + task spec
  4. code_generator.py — Claude API call, templated prompt
  5. executor.py — sandboxed runner
  6. interpreter.py — Claude API call, structured results → narrative
  7.FastAPI wrapper
  8. Streamlit UI for internal demos