"""Web app: upload a PDF, extract sentences, and classify claims.

Sentence extraction comes from the sibling Sentence-Extraction project
(``Sentence_extraction.py``); classification uses ``claim_classifier.py`` and
the saved model in this repository. Run with ``python app.py`` and open
http://127.0.0.1:5000.
"""

import io
import json
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import claim_classifier

ROOT = Path(__file__).resolve().parent
EXTRACTION_DIR = Path(os.environ.get("SENTENCE_EXTRACTION_DIR", ROOT.parent / "Sentence-Extraction")).resolve()
MODEL_PATH = Path(os.environ.get("CLAIM_MODEL_PATH", ROOT / "claim_classifier.joblib")).resolve()
UPLOAD_DIR = ROOT / "uploads"
MAX_UPLOAD_MB = 100

sys.path.insert(0, str(EXTRACTION_DIR))
import Sentence_extraction as extraction  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

jobs: dict[str, dict] = {}
# spaCy pipelines are not safe to share across threads, so jobs run one at a time.
pipeline_lock = threading.Lock()
_nlp = None


def _sentence_model():
    global _nlp
    if _nlp is None:
        _nlp = extraction._load_sentence_model()
    return _nlp


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    # Round-trip through JSON so NumPy scalars and NaN become plain JSON values.
    return json.loads(df[columns].to_json(orient="records"))


def _run_job(job_id: str) -> None:
    job = jobs[job_id]
    job_dir = UPLOAD_DIR / job_id
    pdf_path = job_dir / job["filename"]
    sentences_csv = job_dir / f"{pdf_path.stem}_sentences.csv"
    review_csv = job_dir / f"{pdf_path.stem}_review.csv"
    classified_csv = job_dir / f"{pdf_path.stem}_classified.csv"
    try:
        with pipeline_lock:
            job["stage"] = "loading_model" if _nlp is None else "extracting"
            nlp = _sentence_model()
            job["stage"] = "extracting"
            records = extraction.parse_pdf_to_sentences(pdf_path, nlp=nlp)
            ready = records[records["classification_ready"]]
            review = records[~records["classification_ready"]]
            ready.to_csv(sentences_csv, index=False, encoding="utf-8-sig")
            review.to_csv(review_csv, index=False, encoding="utf-8-sig")

            job["stage"] = "classifying"
            if len(ready):
                summary = claim_classifier.classify_csv(sentences_csv, MODEL_PATH, classified_csv)
                classified = pd.read_csv(classified_csv, encoding="utf-8-sig", keep_default_na=False)
            else:
                summary = {"auto_accept": 0, "review": 0, "discard": 0}
                classified = pd.DataFrame(columns=[
                    "page_number", "sentence_text", "claim_probability", "decision",
                    "claim_type", "claim_type_keywords",
                ])

        job["result"] = {
            "pages": int(records["page_number"].nunique()) if len(records) else 0,
            "extracted": len(records),
            "classified": len(ready),
            "extraction_review": len(review),
            "summary": summary,
            "sentences": _records(classified, [
                "page_number", "sentence_text", "claim_probability", "decision",
                "claim_type", "claim_type_keywords",
            ]),
            "review_rows": _records(review, ["page_number", "sentence_text", "content_type"]),
        }
        job["downloads"] = {"classified": classified_csv.name if len(ready) else None, "review": review_csv.name}
        job["stage"] = "done"
    except Exception as error:  # surfaced to the UI rather than lost in a thread
        traceback.print_exc()
        job["stage"] = "error"
        job["error"] = str(error)
    finally:
        job["elapsed"] = round(time.monotonic() - job["started"], 1)


@app.get("/")
def index():
    return render_template("index.html", max_upload_mb=MAX_UPLOAD_MB)


@app.post("/api/jobs")
def create_job():
    upload = request.files.get("pdf")
    if upload is None or not upload.filename:
        return jsonify(error="Choose a PDF file to upload."), 400
    filename = secure_filename(upload.filename) or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        return jsonify(error="Only PDF files are supported."), 400
    head = upload.stream.read(5)
    upload.stream.seek(0)
    if head != b"%PDF-":
        return jsonify(error="That file does not look like a PDF."), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)
    upload.save(job_dir / filename)
    jobs[job_id] = {"id": job_id, "filename": filename, "stage": "queued", "started": time.monotonic()}
    threading.Thread(target=_run_job, args=(job_id,), daemon=True).start()
    return jsonify(id=job_id), 202


def _job_or_404(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        abort(404)
    return job


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    job = _job_or_404(job_id)
    payload = {key: job.get(key) for key in ("id", "filename", "stage", "error", "result")}
    payload["elapsed"] = job.get("elapsed", round(time.monotonic() - job["started"], 1))
    return jsonify(payload)


@app.get("/api/jobs/<job_id>/pdf")
def job_pdf(job_id: str):
    job = _job_or_404(job_id)
    return send_file(UPLOAD_DIR / job_id / job["filename"], mimetype="application/pdf")


@app.get("/api/jobs/<job_id>/page/<int:page_number>.png")
def job_page_image(job_id: str, page_number: int):
    """Render one page, highlighting the ``q`` sentence where it can be found."""

    import pymupdf

    job = _job_or_404(job_id)
    with pymupdf.open(UPLOAD_DIR / job_id / job["filename"]) as document:
        if not 1 <= page_number <= document.page_count:
            abort(404)
        page = document[page_number - 1]
        sentence = " ".join(request.args.get("q", "").split())
        if sentence:
            # Extracted text is cleaned Markdown, so fall back to its opening
            # words when the full sentence does not match the PDF text layer.
            words = sentence.split()
            for candidate in (sentence, " ".join(words[:8]), " ".join(words[:4])):
                quads = page.search_for(candidate, quads=True)
                if quads:
                    page.add_highlight_annot(quads)
                    break
        image = page.get_pixmap(dpi=150).tobytes("png")
    return send_file(io.BytesIO(image), mimetype="image/png")


@app.get("/api/jobs/<job_id>/download/<kind>")
def job_download(job_id: str, kind: str):
    job = _job_or_404(job_id)
    name = (job.get("downloads") or {}).get(kind)
    if not name:
        abort(404)
    return send_file(UPLOAD_DIR / job_id / name, as_attachment=True, download_name=name)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error=f"PDF is larger than {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    if not MODEL_PATH.is_file():
        sys.exit(f"Saved model not found: {MODEL_PATH}")
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), threaded=True)
