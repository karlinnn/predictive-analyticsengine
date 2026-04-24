import json
import os

import pandas as pd

try:
    from .llm_client import call_llm
except ImportError:
    from llm_client import call_llm


def _load_dataframe(file_path: str) -> pd.DataFrame:
    lower_path = file_path.lower()
    if lower_path.endswith(".csv"):
        return pd.read_csv(file_path)
    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file format. Use CSV or Excel (.xlsx/.xls).")


def _safe_sample_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def extract_raw_stats(file_path: str) -> dict:
    df = _load_dataframe(file_path)
    columns = []

    for column_name in df.columns:
        series = df[column_name]
        columns.append(
            {
                "column_name": str(column_name),
                "dtype": str(series.dtype),
                "null_percentage": float(series.isna().mean() * 100.0),
                "unique_count": int(series.nunique(dropna=True)),
                "sample_values": [_safe_sample_value(v) for v in series.head(5).tolist()],
            }
        )

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": columns,
    }



def _extract_json_content(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def build_schema_manifest(file_path: str) -> dict:
    raw_stats = extract_raw_stats(file_path)

    system_prompt = (
        "You are a data profiling agent. "
        "Infer semantic types from raw stats only. "
        "Do not include any text outside valid JSON."
    )

    user_prompt = (
        "Given these raw dataset statistics, infer semantic_type for each column.\n"
        "Allowed semantic_type values: "
        '["numeric", "categorical", "datetime", "id_like", "text", "target_candidate"]\n'
        "Return ONLY valid JSON in this exact structure:\n"
        "{\n"
        '  "n_rows": int,\n'
        '  "n_columns": int,\n'
        '  "columns": [\n'
        "    {\n"
        '      "name": str,\n'
        '      "dtype": str,\n'
        '      "semantic_type": str,\n'
        '      "reasoning": str,\n'
        '      "null_percentage": float,\n'
        '      "unique_count": int,\n'
        '      "sample_values": list\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Raw stats JSON:\n"
        f"{json.dumps(raw_stats, ensure_ascii=True)}"
    )

    response = call_llm(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    content = response["choices"][0]["message"]["content"]
    manifest = _extract_json_content(content)
    return manifest


if __name__ == "__main__":
    dataset_path = input("Enter CSV/Excel file path: ").strip()
    schema_manifest = build_schema_manifest(dataset_path)
    print(json.dumps(schema_manifest, indent=2, ensure_ascii=True))
