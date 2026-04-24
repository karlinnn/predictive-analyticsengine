import json
import os

try:
    from .llm_client import call_llm
except ImportError:
    from llm_client import call_llm


def _extract_json_content(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def plan_task(user_question: str, schema_manifest: dict, knowledge_base: dict) -> dict:
    system_prompt = (
        "You are an ML architect planning how to solve a data analytics problem. "
        "All reasoning must use the provided user question, schema manifest, and knowledge base. "
        "Do not use external assumptions. Return only valid JSON."
    )

    user_prompt = (
        "Inputs:\n"
        f"user_question: {user_question}\n\n"
        "schema_manifest:\n"
        f"{json.dumps(schema_manifest, ensure_ascii=True)}\n\n"
        "knowledge_base:\n"
        f"{json.dumps(knowledge_base, ensure_ascii=True)}\n\n"
        "Instructions:\n"
        "- Infer the task_type from: classification | regression | clustering | forecasting | factor_analysis\n"
        "- Identify the most appropriate target_column from schema (or null)\n"
        "- Identify candidate_feature_columns\n"
        "- From the provided knowledge_base, select and rank suitable models\n"
        "- Choose explainability method from selected model info\n\n"
        "Return ONLY valid JSON in this exact structure:\n"
        "{\n"
        '  "task_type": str,\n'
        '  "target_column": str | null,\n'
        '  "candidate_features": list,\n'
        '  "selected_models": [\n'
        "    {\n"
        '      "model_name": str,\n'
        '      "priority": int,\n'
        '      "reason_for_selection": str,\n'
        '      "explainability_method": str\n'
        "    }\n"
        "  ],\n"
        '  "reasoning": str\n'
        "}"
    )

    response = call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    content = response["choices"][0]["message"]["content"]
    return _extract_json_content(content)


if __name__ == "__main__":
    import os as _os

    try:
        from .profiler import build_schema_manifest
        from .knowledge_base import generate_knowledge_base
    except ImportError:
        from profiler import build_schema_manifest
        from knowledge_base import generate_knowledge_base

    _root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    _demo_csv = _os.path.join(_root, "data", "dummy_students.csv")
    demo_schema_manifest = build_schema_manifest(_demo_csv)
    demo_knowledge_base = generate_knowledge_base()
    demo_question = (
        "What predicts student stop-out in this dataset, and which models should we consider?"
    )

    plan = plan_task(
        user_question=demo_question,
        schema_manifest=demo_schema_manifest,
        knowledge_base=demo_knowledge_base,
    )
    print(json.dumps(plan, indent=2, ensure_ascii=True))
