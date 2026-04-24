import json
import os
import sys

try:
    from .profiler import build_schema_manifest
    from .knowledge_base import generate_knowledge_base
    from .task_planner import plan_task
    from .code_generator import generate_training_code
    from .executor import execute_training_code
    from .interpreter import build_interpretation_input, interpret
except ImportError:
    from profiler import build_schema_manifest
    from knowledge_base import generate_knowledge_base
    from task_planner import plan_task
    from code_generator import generate_training_code
    from executor import execute_training_code
    from interpreter import build_interpretation_input, interpret


def _default_knowledge_base_path() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "knowledge_base.json")


def load_knowledge_base(
    json_path: str | None = None,
    prefer_live: bool = True,
) -> tuple[dict, str]:
    """
    Load model knowledge: try live LLM generation first, then fall back to JSON file.
    Returns (knowledge_base_dict, source) where source is \"live\" or \"file\".
    """
    path = json_path or _default_knowledge_base_path()
    if prefer_live:
        try:
            return generate_knowledge_base(), "live"
        except Exception:
            pass
    with open(path, encoding="utf-8") as f:
        return json.load(f), "file"


def run_analytics_pipeline(
    file_path: str,
    user_question: str,
    *,
    knowledge_base_json_path: str | None = None,
    prefer_live_knowledge_base: bool = True,
) -> dict:
    """
    End-to-end: schema manifest (profiler) -> knowledge base -> task plan (LLM).
    """
    schema_manifest = build_schema_manifest(file_path)
    knowledge_base, kb_source = load_knowledge_base(
        json_path=knowledge_base_json_path,
        prefer_live=prefer_live_knowledge_base,
    )
    plan = plan_task(user_question, schema_manifest, knowledge_base)
    return {
        "schema_manifest": schema_manifest,
        "knowledge_base": knowledge_base,
        "knowledge_base_source": kb_source,
        "plan": plan,
    }


def run_planning_pipeline(
    file_path: str,
    user_question: str,
    *,
    knowledge_base_json_path: str | None = None,
    prefer_live_knowledge_base: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Planning orchestrator entry point: profiler manifest, knowledge base, task plan.
    Returns (schema_manifest, knowledge_base, plan).
    """
    bundle = run_analytics_pipeline(
        file_path,
        user_question,
        knowledge_base_json_path=knowledge_base_json_path,
        prefer_live_knowledge_base=prefer_live_knowledge_base,
    )
    return bundle["schema_manifest"], bundle["knowledge_base"], bundle["plan"]


def run_full_pipeline(
    file_path: str,
    user_question: str,
    *,
    knowledge_base_json_path: str | None = None,
    prefer_live_knowledge_base: bool = True,
    execute_timeout: int = 180,
) -> dict:
    """
    Full engine pipeline:
    profiler -> knowledge base -> task planner -> code generator -> executor -> interpreter.
    """
    planning = run_analytics_pipeline(
        file_path,
        user_question,
        knowledge_base_json_path=knowledge_base_json_path,
        prefer_live_knowledge_base=prefer_live_knowledge_base,
    )
    schema_manifest = planning["schema_manifest"]
    plan = planning["plan"]

    training_code = generate_training_code(schema_manifest, plan, file_path)
    execution = execute_training_code(training_code, timeout=execute_timeout)

    interpretation_input = None
    interpretation = None
    if execution.get("success") and isinstance(execution.get("result_json"), dict):
        interpretation_input = build_interpretation_input(
            execution,
            task=plan.get("task_type", ""),
            target=plan.get("target_column") or "",
            schema_context={
                "n_rows": schema_manifest.get("n_rows"),
                "column_names": [c.get("name") for c in schema_manifest.get("columns", [])],
            },
        )
        interpretation = interpret(interpretation_input)

    return {
        "schema_manifest": schema_manifest,
        "knowledge_base": planning["knowledge_base"],
        "knowledge_base_source": planning["knowledge_base_source"],
        "plan": plan,
        "training_code": training_code,
        "execution": execution,
        "interpretation_input": interpretation_input,
        "interpretation": interpretation,
    }


if __name__ == "__main__":
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_dataset = os.path.join(_repo_root, "data", "dummy_students.csv")
    dataset = sys.argv[1] if len(sys.argv) > 1 else default_dataset
    question = (
        sys.argv[2]
        if len(sys.argv) > 2
        else (
            "Given this tabular dataset, identify the analytical task, likely target, "
            "and rank appropriate models from the knowledge base."
        )
    )
    result = run_full_pipeline(dataset, question)
    print(json.dumps(result, indent=2, ensure_ascii=True))
