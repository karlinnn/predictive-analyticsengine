import json
import os
import subprocess
import tempfile


def _parse_result_json(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def execute_training_code(training_code: str, timeout: int = 120) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "generated_script.py")
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(training_code)

        completed = subprocess.run(
            ["python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    result_json = _parse_result_json(stdout)
    success = completed.returncode == 0

    return {
        "success": success,
        "stdout": stdout,
        "stderr": stderr,
        "result_json": result_json,
    }
