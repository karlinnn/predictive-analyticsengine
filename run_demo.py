"""
End-to-end demo harness for the planning pipeline (testing only).
Not an agent; hardcoded demo inputs live here only.
"""

import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.pipeline import run_planning_pipeline

file_path = "data/dummy_students.csv"
question = "Which students are at risk of stopping out?"


if __name__ == "__main__":
    os.chdir(_ROOT)
    manifest, kb, plan = run_planning_pipeline(file_path, question)
    print("=== schema_manifest ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    print("=== knowledge_base ===")
    print(json.dumps(kb, indent=2, ensure_ascii=True))
    print("=== plan ===")
    print(json.dumps(plan, indent=2, ensure_ascii=True))
