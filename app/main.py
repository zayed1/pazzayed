from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings, BASE_DIR
from .ingest import SUPPORTED_SUFFIXES
from .vectorstore import Index, build_index
from . import rag, llm

app = FastAPI(title="قارئ الكتب الذكي")

STATIC_DIR = BASE_DIR / "static"

# Loaded index kept in memory.
_index: Index | None = None


def get_index() -> Index:
    global _index
    if _index is None:
        try:
            _index = Index.load()
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="الفهرس غير مبني بعد. شغّل البناء عبر /api/rebuild أو scripts/build_index.py",
            )
    return _index


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/api/status")
def status():
    data_files = [
        p.name for p in settings.data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    index_built = (settings.index_dir / "store.pkl").exists()
    return {
        "ollama_available": llm.is_available(),
        "retriever": settings.retriever,
        "model": settings.ollama_model,
        "index_built": index_built,
        "data_files": data_files,
    }


@app.post("/api/rebuild")
def rebuild():
    global _index
    try:
        _index = build_index()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "chunks": len(_index.chunks), "backend": _index.backend}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="يُسمح فقط بملفات PDF أو CSV أو XLSX.")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / Path(file.filename).name
    dest.write_bytes(await file.read())
    return {"ok": True, "saved": dest.name}


@app.post("/api/ask")
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="السؤال فارغ.")
    index = get_index()
    return rag.answer(index, question, top_k=req.top_k)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
