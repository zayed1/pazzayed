from __future__ import annotations

import httpx

from .config import settings


class OllamaError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def embed(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Ollama. Raises OllamaError if unavailable."""
    vectors: list[list[float]] = []
    try:
        with httpx.Client(timeout=settings.ollama_timeout) as client:
            for text in texts:
                r = client.post(
                    f"{settings.ollama_host}/api/embeddings",
                    json={"model": settings.ollama_embed_model, "prompt": text},
                )
                r.raise_for_status()
                vectors.append(r.json()["embedding"])
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"تعذّر الحصول على المتجهات من Ollama: {exc}") from exc
    return vectors


def generate(question: str, context: str) -> str:
    """Ask the local model to answer using only the provided context."""
    system = (
        "أنت مساعد ذكي يقرأ مقتطفات من كتب ومصادر ويجيب على الأسئلة بالاعتماد عليها فقط. "
        "أجب بالعربية بدقة ووضوح. إذا لم تكن الإجابة موجودة في المصادر المتاحة، قل بصراحة "
        "أن المعلومة غير متوفرة في المصادر. اذكر المصدر والموضع عند الإمكان."
    )
    prompt = (
        f"المصادر المتاحة:\n\n{context}\n\n"
        f"السؤال: {question}\n\n"
        "الإجابة المبنية على المصادر أعلاه:"
    )
    try:
        with httpx.Client(timeout=settings.ollama_timeout) as client:
            r = client.post(
                f"{settings.ollama_host}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                },
            )
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        raise OllamaError(f"تعذّر توليد الإجابة من Ollama: {exc}") from exc
