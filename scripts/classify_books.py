#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
يصنّف الكتب الغامضة في فهرس المكتبة.

الخطوات:
  1. قراءة ملف Excel للمكتبة.
  2. تحديد الكتب الغامضة (تصنيفها "متنوع …" أو "يحتاج مراجعة" أو عمود يحتاج_مراجعة = نعم).
  3. جلب بيانات كل كتاب غامض من archive.org عبر المعرّف (مع تخزين مؤقت وإعادة محاولة).
  4. تدريب مُصنّف نصي على الكتب المصنّفة مسبقاً (بدون أي خدمة خارجية).
  5. توقّع التصنيف لكل كتاب غامض مع درجة ثقة، وكتابة ملف Excel ناتج جديد.

التشغيل:
  python scripts/classify_books.py
  python scripts/classify_books.py --include-general --limit 50
  python scripts/classify_books.py --confidence-threshold 0.4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import httpx
import openpyxl
from openpyxl import Workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"
OUTPUT_DIR = BASE_DIR / "output"

# تصنيفات تُعتبر "غامضة / تحتاج إعادة تصنيف".
AMBIGUOUS_CATEGORIES = {"متنوع - يحتاج تصنيف", "يحتاج مراجعة"}
GENERAL_CATEGORY = "عام"

# أعمدة الملف.
COL_CATEGORY = "التصنيف"
COL_TITLE_ENH = "العنوان_المحسّن"
COL_TITLE_ORIG = "العنوان_الأصلي"
COL_AUTHOR = "المؤلف"
COL_DOWNLOADS = "عدد_التحميلات"
COL_LINK = "الرابط"
COL_NEEDS_REVIEW = "يحتاج_مراجعة"
COL_ID = "المعرّف"

USER_AGENT = "library-book-classifier/1.0 (offline research; contact: zayed1@gmail.com)"
_HTML_TAG = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# قراءة Excel
# ---------------------------------------------------------------------------
def find_default_input() -> Path:
    xlsx = sorted(DATA_DIR.glob("*.xlsx"))
    if not xlsx:
        sys.exit(f"لم أجد أي ملف .xlsx في {DATA_DIR}")
    return xlsx[0]


def read_workbook(path: Path) -> tuple[list[str], list[dict]]:
    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    records: list[dict] = []
    for raw in rows:
        rec = {header[i]: raw[i] for i in range(len(header)) if i < len(raw)}
        records.append(rec)
    wb.close()
    return header, records


def is_ambiguous(rec: dict) -> bool:
    cat = str(rec.get(COL_CATEGORY) or "").strip()
    needs = str(rec.get(COL_NEEDS_REVIEW) or "").strip()
    return (
        cat in AMBIGUOUS_CATEGORIES
        or cat.startswith("متنوع")
        or needs == "نعم"
    )


# ---------------------------------------------------------------------------
# جلب بيانات archive.org (مع تخزين مؤقت)
# ---------------------------------------------------------------------------
def load_cache(cache_file: Path) -> dict:
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    return {}


def save_cache(cache_file: Path, cache: dict) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def _flatten(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ، ".join(str(v) for v in value)
    return str(value)


def fetch_metadata(client: httpx.Client, identifier: str, retries: int = 3) -> dict | None:
    """يرجّع dict فيه title/creator/subject/description أو None عند الفشل."""
    url = f"https://archive.org/metadata/{identifier}"
    delay = 2.0
    for attempt in range(retries):
        try:
            r = client.get(url, timeout=20, follow_redirects=True)
            if r.status_code == 200:
                md = r.json().get("metadata", {})
                return {
                    "title": _flatten(md.get("title")),
                    "creator": _flatten(md.get("creator")),
                    "subject": _flatten(md.get("subject")),
                    "description": _HTML_TAG.sub(" ", _flatten(md.get("description"))),
                    "language": _flatten(md.get("language")),
                }
            if r.status_code in (429, 503):
                time.sleep(delay)
                delay *= 2
                continue
            return None  # 403/404 وغيرها: لا فائدة من إعادة المحاولة
        except Exception:
            time.sleep(delay)
            delay *= 2
    return None


# ---------------------------------------------------------------------------
# نص قابل للتصنيف
# ---------------------------------------------------------------------------
def base_text(rec: dict) -> str:
    parts = [
        str(rec.get(COL_TITLE_ENH) or ""),
        str(rec.get(COL_TITLE_ORIG) or ""),
        str(rec.get(COL_AUTHOR) or ""),
    ]
    return " ".join(p for p in parts if p.strip())


def enriched_text(rec: dict, meta: dict | None) -> str:
    text = base_text(rec)
    if meta:
        extra = " ".join(
            meta.get(k, "") for k in ("title", "creator", "subject", "description")
        )
        text = f"{text} {extra}"
    return text.strip()


# ---------------------------------------------------------------------------
# المُصنّف
# ---------------------------------------------------------------------------
def build_classifier(labeled: list[dict], include_general: bool):
    excluded = set(AMBIGUOUS_CATEGORIES)
    if not include_general:
        excluded.add(GENERAL_CATEGORY)

    X, y = [], []
    for rec in labeled:
        cat = str(rec.get(COL_CATEGORY) or "").strip()
        if not cat or cat in excluded or cat.startswith("متنوع"):
            continue
        text = base_text(rec)
        if text:
            X.append(text)
            y.append(cat)

    if len(set(y)) < 2:
        sys.exit("لا توجد تصنيفات كافية لتدريب المُصنّف.")

    clf = make_pipeline(
        TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0),
    )
    clf.fit(X, y)
    return clf, sorted(set(y))


def predict(clf, text: str) -> tuple[str, float]:
    if not text.strip():
        return "غير معروف", 0.0
    proba = clf.predict_proba([text])[0]
    classes = clf.classes_
    best = proba.argmax()
    return str(classes[best]), float(proba[best])


# ---------------------------------------------------------------------------
# كتابة الناتج
# ---------------------------------------------------------------------------
def write_output(path: Path, header: list[str], records: list[dict], results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_cols = [
        "التصنيف_المقترح",
        "درجة_الثقة",
        "أُعيد_تصنيفه",
        "مصدر_البيانات",
        "وصف_archive",
        "مواضيع_archive",
    ]
    out_header = list(header) + [c for c in extra_cols if c not in header]

    wb = Workbook()
    ws = wb.active
    ws.title = "مصنفة"
    ws.append(out_header)

    for i, rec in enumerate(records):
        res = results.get(i)
        row = dict(rec)
        if res:
            # حدّث التصنيف الأساسي إذا تجاوز عتبة الثقة.
            if res["applied"]:
                row[COL_CATEGORY] = res["suggested"]
            row["التصنيف_المقترح"] = res["suggested"]
            row["درجة_الثقة"] = round(res["confidence"], 3)
            row["أُعيد_تصنيفه"] = "نعم" if res["applied"] else "لا (ثقة منخفضة)"
            row["مصدر_البيانات"] = res["data_source"]
            row["وصف_archive"] = res["meta_desc"]
            row["مواضيع_archive"] = res["meta_subject"]
        ws.append([row.get(c, "") for c in out_header])

    wb.save(str(path))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="تصنيف الكتب الغامضة في فهرس المكتبة")
    ap.add_argument("--input", type=Path, default=None, help="ملف Excel المدخل")
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR / "مكتبة-مصنفة-محدثة.xlsx")
    ap.add_argument("--cache", type=Path, default=CACHE_DIR / "archive_metadata.json")
    ap.add_argument("--include-general", action="store_true",
                    help="أعد تصنيف كتب 'عام' أيضاً")
    ap.add_argument("--confidence-threshold", type=float, default=0.4,
                    help="أقل درجة ثقة لاعتماد التصنيف الجديد")
    ap.add_argument("--sleep", type=float, default=0.5, help="مهلة بين طلبات archive.org (ثوانٍ)")
    ap.add_argument("--limit", type=int, default=0, help="حدّ عدد الكتب الغامضة (للتجربة)")
    ap.add_argument("--no-fetch", action="store_true",
                    help="تخطّ جلب archive.org وصنّف بالعنوان فقط")
    args = ap.parse_args()

    input_path = args.input or find_default_input()
    print(f"المدخل: {input_path}")
    header, records = read_workbook(input_path)
    print(f"إجمالي الكتب: {len(records)}")

    ambiguous_idx = [i for i, r in enumerate(records) if is_ambiguous(r)]
    if args.limit:
        ambiguous_idx = ambiguous_idx[: args.limit]
    print(f"الكتب الغامضة المستهدفة: {len(ambiguous_idx)}")

    print("تدريب المُصنّف على الكتب المصنّفة مسبقاً…")
    clf, classes = build_classifier(records, args.include_general)
    print(f"التصنيفات المتاحة ({len(classes)}): {'، '.join(classes)}")

    cache = load_cache(args.cache)
    results: dict[int, dict] = {}
    enriched = applied = low_conf = fetch_ok = 0

    client = None
    if not args.no_fetch:
        client = httpx.Client(headers={"User-Agent": USER_AGENT})

    try:
        for n, i in enumerate(ambiguous_idx, start=1):
            rec = records[i]
            ident = str(rec.get(COL_ID) or "").strip()
            meta = None

            if not args.no_fetch and ident:
                if ident in cache:
                    meta = cache[ident]
                else:
                    meta = fetch_metadata(client, ident)
                    cache[ident] = meta  # خزّن حتى None لتفادي إعادة المحاولة
                    if n % 25 == 0:
                        save_cache(args.cache, cache)
                    time.sleep(args.sleep)
                if meta:
                    fetch_ok += 1

            if meta:
                enriched += 1
            text = enriched_text(rec, meta)
            suggested, conf = predict(clf, text)
            ok = conf >= args.confidence_threshold
            if ok:
                applied += 1
            else:
                low_conf += 1

            results[i] = {
                "suggested": suggested,
                "confidence": conf,
                "applied": ok,
                "data_source": "archive.org" if meta else "العنوان فقط",
                "meta_desc": (meta or {}).get("description", "")[:500],
                "meta_subject": (meta or {}).get("subject", ""),
            }

            if n % 20 == 0 or n == len(ambiguous_idx):
                print(f"  [{n}/{len(ambiguous_idx)}] جُلب: {fetch_ok} | مُعتمد: {applied} | ثقة منخفضة: {low_conf}")
    finally:
        if client:
            client.close()
        save_cache(args.cache, cache)

    write_output(args.output, header, records, results)
    print("\n=== الملخص ===")
    print(f"كتب جُلبت بياناتها من archive.org: {fetch_ok}/{len(ambiguous_idx)}")
    print(f"أُعيد تصنيفها (ثقة ≥ {args.confidence_threshold}): {applied}")
    print(f"بقيت بثقة منخفضة (تحتاج مراجعة يدوية): {low_conf}")
    print(f"الملف الناتج: {args.output}")
    print(f"الذاكرة المؤقتة: {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
