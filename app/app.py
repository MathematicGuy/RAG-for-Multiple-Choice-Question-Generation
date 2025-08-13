import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Import the user's RAGMCQ implementation
from together_gen import RAGMCQ

app = FastAPI(title="RAG MCQ Generator API")

# allow cross-origin requests (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global rag instance (created at startup)
rag: Optional[RAGMCQ] = None


class GenerateResponse(BaseModel):
    mcqs: dict
    validation: Optional[dict] = None


@app.on_event("startup")
def startup_event():
    global rag

    # Instantiate the heavy object once
    rag = RAGMCQ()
    print("RAGMCQ instance created on startup.")


@app.get("/health")
def health():
    return {"status": "ok", "ready": rag is not None}


def _save_upload_to_temp(upload: UploadFile) -> str:
    suffix = ".pdf"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as out_file:
        shutil.copyfileobj(upload.file, out_file)
    return path


def _normalize_mcqs(raw_mcqs: dict) -> dict:
    print(raw_mcqs)
    normalized = {}
    for k, item in raw_mcqs.items():
        if not isinstance(item, dict):
            normalized[k] = {"mcq": str(item), "options": {}, "correct": ""}
            continue

        # vietnamese schema used in utils.generate_mcqs_from_text
        if "câu hỏi" in item or "lựa chọn" in item or "đáp án" in item:
            q = item.get("câu hỏi") or item.get("Câu hỏi") or item.get("question")
            opts = item.get("lựa chọn") or item.get("lựa chọn:") or item.get("lựa chọn ") or item.get("options") or {}
            a = item.get("đáp án") or item.get("Đáp án") or item.get("answer") or item.get("đáp án ")
            normalized[k] = {"mcq": q, "options": opts, "correct": a}
            continue

    return normalized


@app.post("/generate", response_model=GenerateResponse)
async def generate_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    n_questions: int = Form(10),
    mode: str = Form("rag"),
    questions_per_page: int = Form(3),
    top_k: int = Form(3),
    temperature: float = Form(0.2),
    validate: bool = Form(False),
    debug=False,
):
    global rag
    if rag is None:
        raise HTTPException(status_code=503, detail="RAGMCQ not ready on server.")

    # basic file validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # save uploaded file to a temp location
    tmp_path = _save_upload_to_temp(file)

    # ensure file removed afterward
    def _cleanup(path: str):
        try:
            os.remove(path)
        except Exception:
            pass

    background_tasks.add_task(_cleanup, tmp_path)

    # generate
    try:
        mcqs = rag.generate_from_pdf(
            tmp_path,
            n_questions=n_questions,
            mode=mode,
            questions_per_page=questions_per_page,
            top_k=top_k,
            temperature=temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    validation_report = None

    if validate:
        try:
            normalized = _normalize_mcqs(mcqs)
            # validate_mcqs expects keys as strings and the normalized content
            validation_report = rag.validate_mcqs(normalized)
        except Exception as e:
            # don't fail the whole request for a validation error — return generator output and note the error
            validation_report = {"error": f"Validation failed: {e}"}

    if debug:
        import json
        with open("output.json", "w", encoding='utf-8') as f:
            result = {"mcqs": mcqs, "validation": validation_report}
            json.dump(result, f, ensure_ascii=False, indent=4)

    return {"mcqs": mcqs, "validation": validation_report}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
