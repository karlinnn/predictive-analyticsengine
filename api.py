import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from engine.pipeline import run_full_pipeline, run_planning_pipeline

app = FastAPI(title="Predictive Analytics Engine API")

_SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls"}


def _save_upload(file: UploadFile) -> tuple[str, str | None]:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        return "", f"Unsupported file type '{suffix}'. Accepted: {sorted(_SUPPORTED_SUFFIXES)}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
    finally:
        tmp.close()
    return tmp.name, None


@app.get("/health")
def health():
    return {"status": "success", "result": {"message": "ok"}}


@app.post("/plan")
async def plan(
    file: Annotated[UploadFile, File(description="CSV or Excel dataset")],
    question: Annotated[str, Form(description="Plain-English analytical question")],
    prefer_live_knowledge_base: Annotated[bool, Form()] = True,
):
    """
    Run the planning pipeline only (profiler → knowledge base → task plan).
    Fast — no code generation or model training.
    """
    tmp_path, err = _save_upload(file)
    if err:
        return JSONResponse(status_code=415, content={"status": "error", "message": err})
    try:
        schema_manifest, knowledge_base, plan_result = run_planning_pipeline(
            tmp_path,
            question,
            prefer_live_knowledge_base=prefer_live_knowledge_base,
        )
        return JSONResponse(content={
            "status": "success",
            "result": {
                "schema_manifest": schema_manifest,
                "knowledge_base": knowledge_base,
                "plan": plan_result,
            },
        })
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    finally:
        os.remove(tmp_path)


@app.post("/analyze")
async def analyze(
    file: Annotated[UploadFile, File(description="CSV or Excel dataset")],
    question: Annotated[str, Form(description="Plain-English analytical question")],
    prefer_live_knowledge_base: Annotated[bool, Form()] = True,
    execute_timeout: Annotated[int, Form(ge=10, le=600)] = 180,
):
    """
    Run the full pipeline: profiler → knowledge base → task plan →
    code generation → execution → interpretation.
    """
    tmp_path, err = _save_upload(file)
    if err:
        return JSONResponse(status_code=415, content={"status": "error", "message": err})
    try:
        result = run_full_pipeline(
            tmp_path,
            question,
            prefer_live_knowledge_base=prefer_live_knowledge_base,
            execute_timeout=execute_timeout,
        )
        return JSONResponse(content={"status": "success", "result": result})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    finally:
        os.remove(tmp_path)
