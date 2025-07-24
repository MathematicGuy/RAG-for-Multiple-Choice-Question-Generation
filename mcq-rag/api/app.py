import time
from enhanced_rag_mcq import EnhancedRAGMCQGenerator, debug_prompt_templates, DifficultyLevel, QuestionType
import numpy as np
import os
import shutil
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from contextlib import asynccontextmanager



generator: Optional[EnhancedRAGMCQGenerator] = None
tmp_folder = "./tmp"
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

class GenerateRequest(BaseModel):
    topics: List[str] = Field(..., description="List of topics for MCQ generation")
    question_per_topic: int = Field(1, ge=1, le=10, description="Number of questions per topic")
    difficulty: Optional[DifficultyLevel] = Field(
        DifficultyLevel.MEDIUM,
        description="Difficulty level for generated questions"
    )
    qtype: Optional[QuestionType] = Field(
        QuestionType.DEFINITION,
        description="Type of question to generate"
    )

class MCQResponse(BaseModel):
    question: str
    options: Dict
    correct_answer: str
    confidence_score: float

class GenerateResponse(BaseModel):
    topics: List[str]
    generated: List[MCQResponse]
    avg_confidence: float
    generation_time: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global generator
    generator = EnhancedRAGMCQGenerator()
    try:
        generator.initialize_system()
        print("RAG system initialized.")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
    yield
    # Optional: Cleanup code after shutdown

app = FastAPI(
    title="Enhanced RAG MCQ Generation API",
    description="An API wrapping the RAG-based MCQ generator using FastAPI",
    version="1.0.0",
    lifespan=lifespan
)

#? cmd: uvicorn app:app --reload --reload-exclude unsloth_compiled_cache
@app.post("/generate/")
async def mcq_gen(
    file: UploadFile = File(...),
    topics: str = Form(...),
    n_questions: str = Form(...),
    difficulty: DifficultyLevel = Form(...),
    qtype: QuestionType = Form(...)
):
    if not generator:
        raise HTTPException(status_code=500, detail="Generator not initialized")

    topic_list = [t.strip() for t in topics.split(',') if t.strip()]
    if not topic_list:
        raise HTTPException(status_code=400, detail="At least one topic must be provided")

    # Save uploaded PDF to temporary folder
    filename = file.filename if file.filename else "uploaded_file"
    file_path = os.path.join(tmp_folder, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    file.file.close()

    try:
        # Load and index the uploaded document
        docs, _ = generator.load_documents(tmp_folder)
        generator.build_vector_database(docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing error: {e}")

    start_time = time.time()
    try:
        mcqs = generator.generate_batch(
            topics=topic_list,
            question_per_topic=int(n_questions)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    end_time = time.time()

    res_dict = [m.to_dict() for m in mcqs]

    responses = [
        MCQResponse(
            question=m["question"],
            options=m["options"],
            correct_answer=m["correct_answer"],
            confidence_score=m["confidence_score"]
        ) for m in res_dict
    ]
    avg_conf = sum(m["confidence_score"] for m in res_dict) / len(mcqs)
    # Clean up temporary files
    shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    return GenerateResponse(
        topics=topic_list,
        generated=responses,
        avg_confidence=avg_conf,
        generation_time=end_time - start_time
    )