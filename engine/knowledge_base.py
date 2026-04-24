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


def generate_knowledge_base() -> dict:
    system_prompt = (
        "You are designing the model knowledge base for an automated predictive analytics engine. "
        "This engine works on tabular datasets (CSV/Excel) across domains like education, business, "
        "surveys, operations. Return only valid JSON."
    )

    user_prompt = (
        "List model families this engine should support.\n"
        "For each model, specify:\n"
        "- model_name\n"
        "- best_use_cases\n"
        "- data_requirements\n"
        "- strengths\n"
        "- weaknesses\n"
        "- explainability_method\n"
        "- task_types_supported (classification, regression, clustering, forecasting, factor_analysis)\n\n"
        "The output must be ONLY valid JSON in this exact structure:\n"
        "{\n"
        '  "models": [\n'
        "    {\n"
        '      "model_name": str,\n'
        '      "task_types_supported": list,\n'
        '      "best_use_cases": str,\n'
        '      "data_requirements": str,\n'
        '      "strengths": str,\n'
        '      "weaknesses": str,\n'
        '      "explainability_method": str\n'
        "    }\n"
        "  ]\n"
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


def save_knowledge_base(output_path: str = "models/knowledge_base.json") -> dict:
    knowledge_base = generate_knowledge_base()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(knowledge_base, file_obj, indent=2, ensure_ascii=True)
    return knowledge_base


if __name__ == "__main__":
    knowledge_base = save_knowledge_base()
    print("Saved knowledge base to models/knowledge_base.json")
    print(json.dumps(knowledge_base, indent=2, ensure_ascii=True))
