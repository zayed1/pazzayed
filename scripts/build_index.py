#!/usr/bin/env python3
"""بناء فهرس البحث من ملفات الكتب في مجلد data/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.vectorstore import build_index  # noqa: E402
from app.config import settings  # noqa: E402


def main() -> int:
    print(f"القراءة من: {settings.data_dir}")
    print(f"محرك البحث: {settings.retriever}")
    try:
        idx = build_index()
    except ValueError as exc:
        print(f"خطأ: {exc}")
        return 1
    print(f"تم بناء الفهرس: {len(idx.chunks)} مقطع. حُفظ في {settings.index_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
