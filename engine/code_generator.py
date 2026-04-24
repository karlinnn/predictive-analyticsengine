import json
import os
import sys

try:
    from .llm_client import call_llm
except ImportError:
    from llm_client import call_llm


def _extract_python_code(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        lowered = cleaned.lower()
        if lowered.startswith("python"):
            cleaned = cleaned[6:].lstrip()
    return cleaned


def generate_training_code(
    schema_manifest: dict,
    task_plan: dict,
    data_file_path: str,
) -> str:
    system_prompt = (
        "You write one complete standalone Python script for ML training. "
        "You must obey task_plan and schema_manifest exactly. "
        "Do not invent models or explainability methods not implied by task_plan. "
        "Return ONLY valid Python source code. No markdown. No prose before or after the code."
    )

    user_prompt = (
        "Write a standalone Python ML training script.\n\n"
        "Inputs (use exactly for behavior; do not change target/features unless task_plan says so):\n\n"
        "schema_manifest (JSON):\n"
        f"{json.dumps(schema_manifest, ensure_ascii=True)}\n\n"
        "task_plan (JSON):\n"
        f"{json.dumps(task_plan, ensure_ascii=True)}\n\n"
        f"data_file_path (load the dataset from this path only): {json.dumps(data_file_path)}\n\n"
        "Library rules:\n"
        "- Use pandas, numpy as needed.\n"
        "- Use scikit-learn for preprocessing, splitting, and metrics where applicable.\n"
        "- Import and use xgboost, statsmodels, prophet, or sklearn clustering modules ONLY if the "
        "task_plan's selected top-priority model clearly requires them (infer from model_name and task_type).\n"
        "- Use shap or another explainability approach ONLY as specified in task_plan for the trained model "
        "(see selected_models[0] explainability_method or task_plan fields).\n"
        "- If using SHAP explainers (e.g. LinearExplainer), use the current keyword feature_perturbation; "
        "do not use the removed feature_dependence argument.\n\n"
        "Script requirements:\n"
        
        
        "- Load the CSV/Excel file from data_file_path (support .csv, .xlsx, .xls via pandas).\n"
        "- Select target column from task_plan['target_column'] and feature columns from "
        "task_plan['candidate_features'] (skip null target if any; align column names with the manifest names).\n"
        "Mandatory Preprocessing Rules (non-negotiable):\n"
        "- Detect numeric, categorical, and datetime columns from the schema_manifest.\n"
        "- Build a ColumnTransformer with a pipeline for each type:\n"
        "  Numeric pipeline (required): SimpleImputer(strategy='median') → StandardScaler()\n"
        "  Categorical pipeline (required): SimpleImputer(strategy='most_frequent') → OneHotEncoder(handle_unknown='ignore')\n"
        "  Datetime handling (required): convert datetime columns to numeric using .astype('int64') // 10**9, "
        "then treat them as numeric and pass through the numeric pipeline (imputer + scaler).\n"
        "- The model MUST be trained ONLY on the output of this ColumnTransformer. Training directly on raw columns is forbidden.\n"
        "- Do NOT use pandas.Series.view() anywhere. Only use .astype() for dtype conversion.\n"
        "- If task_plan['task_type'] is supervised (classification, regression), do train/test split and report metrics.\n"
        "- If clustering or unsupervised as per task_plan, skip supervised split/metrics where not applicable.\n"
        "- Train exactly ONE model: the highest-priority entry in task_plan['selected_models'] (lowest 'priority' "
        "number is rank 1). Implement that model_name faithfully.\n"
        "- Compute metrics appropriate to task_plan['task_type'].\n"
        "- Compute explainability using the method tied to that selected model in task_plan.\n"
        "- At the end, print to stdout valid JSON with this structure (use json.dumps on a dict): "
        '{"metrics": {...}, "feature_importance": {...}, "explainability_summary": str}\n'
        "- feature_importance should be a dict mapping feature names to numeric scores (or SHAP mean abs if used).\n"
        "- The script must use if __name__ == \"__main__\": to run training when executed directly.\n"
        "- Do not read environment variables for model choice; use only task_plan and schema_manifest.\n"
        "- Keep the script self-contained (no argparse required).\n"
    )

    response = call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    content = response["choices"][0]["message"]["content"]
    return _extract_python_code(content)


if __name__ == "__main__":
    try:
        from .pipeline import run_analytics_pipeline
    except ImportError:
        from pipeline import run_analytics_pipeline

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_data = os.path.join(root, "data", "dummy_students.csv")
    data_path = sys.argv[1] if len(sys.argv) > 1 else default_data
    question = (
        sys.argv[2]
        if len(sys.argv) > 2
        else (
            "Plan supervised learning on this dataset: infer task type, target, and features; "
            "then the generator should produce training code for the top-ranked model."
        )
    )
    bundle = run_analytics_pipeline(data_path, question)
    script = generate_training_code(
        bundle["schema_manifest"],
        bundle["plan"],
        data_path,
    )
    print(script)
