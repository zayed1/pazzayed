"""اختبار سريع يتأكد أن الاستيعاب والبحث يعملان دون الحاجة لنموذج خارجي."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import load_chunks  # noqa: E402
from app.vectorstore import Index  # noqa: E402
from app import rag  # noqa: E402


def test_ingest_loads_chunks():
    chunks = load_chunks()
    assert len(chunks) > 0, "يجب أن ينتج عن ملفات data/ مقاطع قابلة للفهرسة"


def test_index_search_returns_results():
    chunks = load_chunks()
    index = Index("tfidf", chunks)
    index.build()
    results = index.search("التاريخ الإسلامي", top_k=3)
    assert results, "يجب أن يرجع البحث نتائج ذات صلة"
    assert results[0][1] > 0, "يجب أن تكون درجة التطابق موجبة"


def test_rag_answer_fallback_without_model():
    chunks = load_chunks()
    index = Index("tfidf", chunks)
    index.build()
    out = rag.answer(index, "ما الكتب المتوفرة في السيرة النبوية؟", top_k=3)
    assert out["sources"], "يجب أن تتضمن الإجابة مصادر"
    assert isinstance(out["answer"], str) and out["answer"].strip()
