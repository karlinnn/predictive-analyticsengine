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


def validate_execution_result(result_json: dict, task_type: str, target_column: str) -> dict:
    """
    Validates model output from the executor before passing it to the interpreter.
    Returns a validation result with is_reliable flag and reasoning.
    """
    system_prompt = (
        "You are a model output validation agent in a predictive analytics pipeline. "
        "Your job is to strictly analyze the model result JSON and determine whether "
        "the outputs are statistically reasonable and trustworthy for the given task. "
        "Do not include any text outside valid JSON."
    )

    user_prompt = (
        "Analyze the model output below and validate whether the results are reliable.\n\n"
        f"task_type: {task_type}\n"
        f"target_column: {target_column}\n\n"
        "model_result_json:\n"
        f"{json.dumps(result_json, ensure_ascii=True)}\n\n"
        "Validation rules:\n"
        "- For classification: accuracy, precision, recall, f1_score must all be between 0.0 and 1.0. "
        "Accuracy below 0.5 is worse than random (unreliable). Extreme precision/recall imbalance "
        "(e.g. precision=1.0 and recall=0.0) is suspicious.\n"
        "- For regression: check that r2_score is not negative (model worse than baseline) and "
        "mae/rmse are not absurdly large relative to each other.\n"
        "- feature_importance must be present and contain at least one entry with a non-zero score.\n"
        "- If any metric is missing, NaN, null, or out of valid range, mark as unreliable.\n"
        "- If everything looks statistically reasonable, mark as reliable.\n\n"
        "Return ONLY valid JSON in this exact structure:\n"
        "{\n"
        '  "is_reliable": bool,\n'
        '  "checks": [\n'
        '    {"check": str, "passed": bool, "detail": str}\n'
        "  ],\n"
        '  "validation_summary": str\n'
        "}"
    )

    response = call_llm([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])

    content = response["choices"][0]["message"]["content"]
    return _extract_json_content(content)
