from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import settings
from .ingest import Chunk, load_chunks
from . import llm


class Index:
    """Search index over book chunks. Backend = 'tfidf' or 'ollama'."""

    def __init__(self, backend: str, chunks: list[Chunk]):
        self.backend = backend
        self.chunks = chunks
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None  # tfidf sparse matrix or dense embedding array

    # ---------- build ----------
    def build(self) -> None:
        texts = [c.text for c in self.chunks]
        if not texts:
            raise ValueError("لا توجد مقاطع لبناء الفهرس. أضف ملفات PDF أو CSV إلى مجلد data/.")

        if self.backend == "ollama":
            vectors = np.array(llm.embed(texts), dtype=np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.matrix = vectors / norms
        else:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
            )
            self.matrix = self.vectorizer.fit_transform(texts)

    # ---------- search ----------
    def search(self, query: str, top_k: int | None = None) -> list[tuple[Chunk, float]]:
        top_k = top_k or settings.top_k
        if self.backend == "ollama":
            qv = np.array(llm.embed([query])[0], dtype=np.float32)
            n = np.linalg.norm(qv) or 1.0
            qv = (qv / n).reshape(1, -1)
            scores = cosine_similarity(qv, self.matrix)[0]
        else:
            qv = self.vectorizer.transform([query])
            scores = cosine_similarity(qv, self.matrix)[0]

        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    # ---------- persistence ----------
    def save(self, index_dir: Path | None = None) -> None:
        index_dir = index_dir or settings.index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        meta = {"backend": self.backend, "count": len(self.chunks)}
        (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        with (index_dir / "store.pkl").open("wb") as f:
            pickle.dump(
                {
                    "chunks": [c.to_dict() for c in self.chunks],
                    "vectorizer": self.vectorizer,
                    "matrix": self.matrix,
                },
                f,
            )

    @classmethod
    def load(cls, index_dir: Path | None = None) -> "Index":
        index_dir = index_dir or settings.index_dir
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        with (index_dir / "store.pkl").open("rb") as f:
            data = pickle.load(f)
        chunks = [Chunk(**c) for c in data["chunks"]]
        idx = cls(meta["backend"], chunks)
        idx.vectorizer = data["vectorizer"]
        idx.matrix = data["matrix"]
        return idx


def build_index(backend: str | None = None) -> Index:
    backend = backend or settings.retriever
    chunks = load_chunks()
    idx = Index(backend, chunks)
    idx.build()
    idx.save()
    return idx
