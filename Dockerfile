FROM python:3.12-slim

# System tools for universal ingestion (ingest.py):
#   pandoc          docx/odt/epub/html/... -> Markdown
#   tesseract-ocr   OCR fallback for scanned PDFs (English + French)
RUN apt-get update && apt-get install -y --no-install-recommends \
        pandoc \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "markitdown[all]" pytesseract pillow pytest

COPY . .

# /data: mount your documents here; results land in /app/results
VOLUME ["/data", "/app/results"]

ENTRYPOINT ["python"]
CMD ["ingest.py", "--help"]
