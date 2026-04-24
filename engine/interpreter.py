import json
import os

try:
    from .llm_client import call_llm
except ImportError:
    from llm_client import call_llm

MIN_ROWS = 50


def _feature_importance_to_list(
    value: list | dict,
) -> list[dict[str, str | float]]:
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict) and "feature" in item and "importance" in item:
                out.append(
                    {
                        "feature": str(item["feature"]),
                        "importance": float(item["importance"]),
                    }
                )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                out.append({"feature": str(item[0]), "importance": float(item[1])})
        return out
    if isinstance(value, dict):
        ranked = sorted(
            value.items(),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
        return [
            {"feature": str(name), "importance": float(score)}
            for name, score in ranked
        ]
    return []


def build_interpretation_input(
    executor_result: dict,
    task: str,
    target: str,
    predictions_summary: dict | None = None,
    schema_context: dict | None = None,
) -> dict:
    """
    Map execute_training_code output + plan fields into the interpreter's input contract.
    Uses only result_json from the executor when success and result_json is present.
    """
    payload: dict = {
        "task": task,
        "target": target,
        "metrics": {},
        "feature_importance": [],
    }
    if predictions_summary is not None:
        payload["predictions_summary"] = predictions_summary
    if schema_context is not None:
        payload["schema_context"] = schema_context

    rj = executor_result.get("result_json")
    if not isinstance(rj, dict):
        return payload

    metrics = rj.get("metrics")
    if isinstance(metrics, dict):
        payload["metrics"] = metrics

    fi = rj.get("feature_importance")
    payload["feature_importance"] = _feature_importance_to_list(
        fi if fi is not None else {}
    )

    return payload



def _parse_json_object(content: str) -> dict:
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


def _extract_row_count(interpretation_input: dict) -> int | None:
    schema_context = interpretation_input.get("schema_context")
    if not isinstance(schema_context, dict):
        return None
    for key in ("row_count", "n_rows", "rows"):
        value = schema_context.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def interpret(interpretation_input: dict) -> dict:
    row_count = _extract_row_count(interpretation_input)
    if row_count is not None and row_count < MIN_ROWS:
        raise ValueError(
            f"Dataset too small for reliable modelling: {row_count} rows. Minimum is {MIN_ROWS}."
        )

    text = json.dumps(interpretation_input, ensure_ascii=True)
    system = (
        "You are a senior data analyst answering a business question with model results. "
        "You ONLY use the single JSON object in the user message. "
        "Do not name features, causes, or metrics that are not present in that JSON. "
        "Do not invent numbers. No technical workflow terms. No code or library names. No outside facts. "
        "Write as if presenting findings to a business stakeholder — plain language, direct, confident. "
        "If the task is a prediction (e.g. retention rate, graduation probability), lead with the predicted "
        "value or rate, then the key influencing factors, then the confidence score. Nothing else. "
        "EXAMPLE for a prediction question: "
        "\"Predicted 2-year retention rate: 81.4%\\n"
        "Higher GPA, financial aid eligibility, and on-campus housing are associated with stronger retention.\\n"
        "Model confidence: AUC 0.87\" "
        "Return only valid JSON with keys: model_summary, performance_assessment, "
        "key_drivers, insights, recommendations (arrays of strings for the last three)."
    )
    user = (
        "Interpretation input JSON (use nothing else):\n" + text + "\n\n"
        "Respond as a data analyst presenting results to a business audience. "
        "Use only: task, target, metrics, feature_importance entries, and if present predictions_summary and schema_context. "
        "Do not suggest collecting new data. Do not use technical jargon. "
        "For prediction questions produce only: predicted value, key influencing factors, confidence metric. "
        "Return only valid JSON in this exact shape:\n"
        "{\n"
        '  "model_summary": str,\n'
        '  "performance_assessment": str,\n'
        '  "key_drivers": [str, ...],\n'
        '  "insights": [str, ...],\n'
        '  "recommendations": [str, ...]\n'
        "}"
    )
    out = call_llm(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    content = out["choices"][0]["message"]["content"]
    return _parse_json_object(content)


if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _dummy = os.path.join(_root, "data", "dummy_executor_result.json")
    with open(_dummy, encoding="utf-8") as _f:
        _executor = json.load(_f)
    _built = build_interpretation_input(
        _executor,
        task="regression",
        target="years to graduate",
    )
    print("=== interpretation_input ===")
    print(json.dumps(_built, indent=2, ensure_ascii=True))
    print("=== interpret() ===")
    print(json.dumps(interpret(_built), indent=2, ensure_ascii=True))
