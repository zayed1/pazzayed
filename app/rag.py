from __future__ import annotations

from .config import settings
from .vectorstore import Index
from . import llm


def _format_context(results) -> str:
    blocks = []
    for chunk, score in results:
        blocks.append(f"[المصدر: {chunk.source} — {chunk.locator}]\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def answer(index: Index, question: str, top_k: int | None = None) -> dict:
    results = index.search(question, top_k=top_k)

    sources = [
        {"source": c.source, "locator": c.locator, "score": round(s, 4), "excerpt": c.text}
        for c, s in results
    ]

    if not results:
        return {
            "answer": "لم أجد معلومات ذات صلة بسؤالك في الكتب المتاحة.",
            "sources": [],
            "used_model": False,
        }

    context = _format_context(results)

    if llm.is_available():
        try:
            text = llm.generate(question, context)
            return {"answer": text, "sources": sources, "used_model": True}
        except llm.OllamaError:
            pass  # fall back to excerpts

    # Fallback: no local model available — return the most relevant excerpts.
    top = results[0][0]
    fallback = (
        "النموذج المحلي غير متوفر حاليًا، لكن هذه أكثر المقاطع صلة بسؤالك:\n\n"
        f"المصدر الأقرب: {top.source} — {top.locator}\n\n{top.text}"
    )
    return {"answer": fallback, "sources": sources, "used_model": False}
