#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
يكتشف تصنيفات جديدة وملائمة لكتب "متنوع - يحتاج تصنيف".

الفكرة: عناوين هذه الكتب في الملف غالباً بلا دلالة (كلمة "منوع")، لذا نجلب
بياناتها الحقيقية من archive.org (العنوان/المواضيع/الوصف)، ثم نجمّع الكتب
المتشابهة في مجموعات (clustering)، ونسمّي كل مجموعة بتصنيف عربي ملائم
(عبر نموذج Ollama المحلي إن توفّر، وإلا من أبرز الكلمات المفتاحية).

التشغيل:
  python scripts/discover_categories.py
  python scripts/discover_categories.py --num-categories 10 --use-ollama
  python scripts/discover_categories.py --no-fetch        # تجربة بالعناوين المحلية فقط
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import numpy as np
from openpyxl import Workbook
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.classify_books import (  # noqa: E402
    CACHE_DIR,
    COL_AUTHOR,
    COL_CATEGORY,
    COL_ID,
    COL_TITLE_ENH,
    COL_TITLE_ORIG,
    OUTPUT_DIR,
    USER_AGENT,
    fetch_metadata,
    find_default_input,
    load_cache,
    read_workbook,
    save_cache,
)
from app.config import settings  # noqa: E402

# كلمات عربية شائعة تُستبعد من تسمية المجموعات.
ARABIC_STOP = {
    "كتاب", "كتب", "في", "من", "على", "عن", "الى", "إلى", "the", "of", "and",
    "pdf", "ج", "جزء", "المجلد", "مجلد", "ال", "وال", "او", "أو", "مع", "هذا",
    "book", "books", "vol", "part", "منوع", "متنوع",
}
_WORD = re.compile(r"[A-Za-z؀-ۿ]{3,}")


def is_mix(rec: dict) -> bool:
    return str(rec.get(COL_CATEGORY) or "").strip().startswith("متنوع")


def discovery_text(rec: dict, meta: dict | None) -> str:
    """نص للتجميع: يعتمد بشكل أساسي على بيانات archive.org إن وُجدت."""
    parts: list[str] = []
    if meta:
        # نكرّر المواضيع لإعطائها وزناً أكبر في التجميع.
        parts += [meta.get("subject", ""), meta.get("subject", ""), meta.get("title", ""),
                  meta.get("description", "")]
    # العنوان المحلي قد يكون مفيداً إن لم يكن "منوع".
    for key in (COL_TITLE_ENH, COL_TITLE_ORIG, COL_AUTHOR):
        val = str(rec.get(key) or "").strip()
        if val and val not in ("منوع", "متنوع"):
            parts.append(val)
    return " ".join(p for p in parts if p).strip()


def choose_k(matrix, requested: int, n: int) -> int:
    if requested:
        return max(2, min(requested, n - 1))
    best_k, best_score = 6, -1.0
    upper = min(16, max(4, n // 8))
    for k in range(5, upper + 1):
        if k >= n:
            break
        labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels, sample_size=min(500, n), random_state=42)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def top_terms(vectorizer, centroid, n: int = 8) -> list[str]:
    feats = np.array(vectorizer.get_feature_names_out())
    order = centroid.argsort()[::-1]
    terms: list[str] = []
    for i in order:
        t = feats[i]
        if t.lower() in ARABIC_STOP or not _WORD.fullmatch(t):
            continue
        terms.append(t)
        if len(terms) >= n:
            break
    return terms


def name_with_ollama(client, terms: list[str], subjects: list[str], titles: list[str]) -> str | None:
    prompt = (
        "أنت أمين مكتبة. أعطِ اسم تصنيف عربي موجز (٢-٤ كلمات) يصف الموضوع المشترك "
        "لهذه الكتب. أعد الاسم فقط بدون أي شرح.\n\n"
        f"كلمات مفتاحية: {'، '.join(terms[:8])}\n"
        f"مواضيع archive: {'، '.join(subjects[:8])}\n"
        f"عناوين أمثلة: {' | '.join(titles[:5])}\n\n"
        "اسم التصنيف:"
    )
    try:
        r = client.post(
            f"{settings.ollama_host}/api/chat",
            json={"model": settings.ollama_model,
                  "messages": [{"role": "user", "content": prompt}],
                  "stream": False},
            timeout=settings.ollama_timeout,
        )
        r.raise_for_status()
        name = r.json()["message"]["content"].strip().strip('"').splitlines()[0]
        return name[:60] or None
    except Exception:
        return None


def fallback_name(terms: list[str], subjects: list[str]) -> str:
    pool = subjects[:3] or terms[:3]
    return "متنوع: " + ("، ".join(pool) if pool else "غير محدد")


def build_vectorizer(corpus: list[str]):
    """يبني TF-IDF مع min_df تكيّفي حتى لا تنهار المفردات على نصوص قليلة."""
    last = None
    for min_df in (3, 2, 1):
        vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
        X = vec.fit_transform(corpus)
        last = (vec, X)
        if X.shape[1] >= 30:
            break
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description="اكتشاف تصنيفات جديدة لكتب 'متنوع'")
    ap.add_argument("--input", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR / "متنوع-تصنيفات-جديدة.xlsx")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "archive_metadata.json")
    ap.add_argument("--num-categories", type=int, default=0, help="عدد التصنيفات الجديدة (0=تلقائي)")
    ap.add_argument("--use-ollama", action="store_true", help="استخدم نموذج Ollama لتسمية المجموعات")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--no-fetch", action="store_true", help="لا تجلب من archive.org")
    args = ap.parse_args()

    input_path = args.input or find_default_input()
    print(f"المدخل: {input_path}")
    header, records = read_workbook(input_path)
    mix_idx = [i for i, r in enumerate(records) if is_mix(r)]
    print(f"كتب 'متنوع - يحتاج تصنيف': {len(mix_idx)}")

    # 1) الإثراء من archive.org
    cache = load_cache(args.cache)
    client = None if args.no_fetch else httpx.Client(headers={"User-Agent": USER_AGENT})
    texts: dict[int, str] = {}
    metas: dict[int, dict | None] = {}
    fetched = 0
    try:
        for n, i in enumerate(mix_idx, start=1):
            rec = records[i]
            ident = str(rec.get(COL_ID) or "").strip()
            meta = None
            if not args.no_fetch and ident:
                if ident in cache:
                    meta = cache[ident]
                else:
                    meta = fetch_metadata(client, ident)
                    cache[ident] = meta
                    if n % 25 == 0:
                        save_cache(args.cache, cache)
                    time.sleep(args.sleep)
                if meta:
                    fetched += 1
            metas[i] = meta
            txt = discovery_text(rec, meta)
            if len(txt) >= 4:
                texts[i] = txt
            if n % 25 == 0 or n == len(mix_idx):
                print(f"  [{n}/{len(mix_idx)}] جُلب من archive: {fetched}")
    finally:
        if client:
            client.close()
        save_cache(args.cache, cache)

    usable = list(texts.keys())
    print(f"كتب فيها نص كافٍ للتجميع: {len(usable)} | بلا بيانات كافية: {len(mix_idx) - len(usable)}")
    if len(usable) < 5:
        sys.exit("بيانات غير كافية للتجميع. شغّل السكربت محلياً مع وصول لـ archive.org.")

    # 2) التجميع
    corpus = [texts[i] for i in usable]
    vectorizer, X = build_vectorizer(corpus)
    print(f"عدد السمات النصية: {X.shape[1]}")
    k = choose_k(X, args.num_categories, len(usable))
    print(f"عدد التصنيفات الجديدة: {k}")
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # 3) تسمية كل مجموعة
    name_client = None
    if args.use_ollama:
        try:
            if httpx.get(f"{settings.ollama_host}/api/tags", timeout=3).status_code == 200:
                name_client = httpx.Client()
                print("تسمية المجموعات عبر Ollama…")
        except Exception:
            print("Ollama غير متاح — سأستخدم تسمية بالكلمات المفتاحية.")

    cluster_books: dict[int, list[int]] = {}
    for pos, lbl in enumerate(labels):
        cluster_books.setdefault(int(lbl), []).append(usable[pos])

    cluster_name: dict[int, str] = {}
    for c, centroid in enumerate(km.cluster_centers_):
        terms = top_terms(vectorizer, centroid)
        subjects: list[str] = []
        titles: list[str] = []
        for i in cluster_books.get(c, []):
            m = metas.get(i) or {}
            if m.get("subject"):
                subjects += [s.strip() for s in re.split(r"[،,;|]", m["subject"]) if s.strip()]
            t = (m.get("title") or str(records[i].get(COL_TITLE_ORIG) or "")).strip()
            if t and t != "منوع":
                titles.append(t)
        top_subjects = [s for s, _ in Counter(subjects).most_common(8)]
        name = None
        if name_client:
            name = name_with_ollama(name_client, terms, top_subjects, titles)
        cluster_name[c] = name or fallback_name(terms, top_subjects)
    if name_client:
        name_client.close()

    # 4) الكتابة
    print("\n=== التصنيفات الجديدة المكتشفة ===")
    for c in sorted(cluster_books, key=lambda x: -len(cluster_books[x])):
        print(f"  • {cluster_name[c]}  ({len(cluster_books[c])} كتاب)")

    assigned: dict[int, str] = {}
    for c, books in cluster_books.items():
        for i in books:
            assigned[i] = cluster_name[c]

    extra = ["التصنيف_الجديد", "مصدر_البيانات", "وصف_archive", "مواضيع_archive"]
    out_header = list(header) + [c for c in extra if c not in header]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wb = Workbook(); ws = wb.active; ws.title = "متنوع-مصنّف"
    ws.append(out_header)
    for i in mix_idx:
        rec = dict(records[i])
        m = metas.get(i)
        rec["التصنيف_الجديد"] = assigned.get(i, "يحتاج مراجعة يدوية (بيانات غير كافية)")
        rec["مصدر_البيانات"] = "archive.org" if m else "العنوان المحلي فقط"
        rec["وصف_archive"] = (m or {}).get("description", "")[:500]
        rec["مواضيع_archive"] = (m or {}).get("subject", "")
        ws.append([rec.get(col, "") for col in out_header])
    wb.save(str(args.output))

    print(f"\nجُلب من archive.org: {fetched}/{len(mix_idx)}")
    print(f"صُنّف ضمن مجموعات: {len(assigned)} | بلا بيانات كافية: {len(mix_idx) - len(assigned)}")
    print(f"الملف الناتج: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
