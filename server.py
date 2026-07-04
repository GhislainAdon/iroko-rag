#!/usr/bin/env python3
"""iroko-rag web API — chat with your documents over HTTP.

Wraps PageIndexClient + the universal ingestion pipeline behind a small
FastAPI app, and serves a ready-to-use chat page at /.

    uvicorn server:app --host 0.0.0.0 --port 8000

Endpoints (CORS open, so any Angular/React/plain-JS frontend can call them):
    GET  /api/documents          list indexed documents
    POST /api/documents          multipart upload -> convert -> index
    POST /api/chat               {doc_id, question} -> {answer, sources}
"""

import json
import os
import shutil
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ingest import (MARKDOWN_EXTS, MARKITDOWN_EXTS, PANDOC_EXTS,
                    convert_with_markitdown, convert_with_pandoc,
                    ensure_headers, ocr_pdf_to_markdown, pdf_is_scanned)
from pageindex.client import PageIndexClient
from pageindex.utils import extract_json, llm_completion

WORKSPACE = os.getenv('IROKO_WORKSPACE', './results/workspace')
MODEL = os.getenv('MODEL') or None  # None -> pageindex/config.yaml default
OCR_LANG = os.getenv('OCR_LANG', 'eng')
# Cap the answer context so small local models don't overflow their window.
CONTEXT_MAX_CHARS = int(os.getenv('CONTEXT_MAX_CHARS', '40000'))

app = FastAPI(title='iroko-rag', docs_url='/api/docs', openapi_url='/api/openapi.json')
app.add_middleware(CORSMiddleware, allow_origins=['*'],
                   allow_methods=['*'], allow_headers=['*'])

client = PageIndexClient(model=MODEL, workspace=WORKSPACE)

PROVIDER_KEY_VARS = ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY',
                     'DEEPSEEK_API_KEY', 'OPENROUTER_API_KEY', 'AZURE_API_KEY',
                     'MISTRAL_API_KEY', 'GROQ_API_KEY')


def _require_llm():
    """Fail fast with a clear message instead of letting every LLM call
    retry into an empty response that looks like 'no relevant section'."""
    model = client.model or ''
    if 'ollama' in model:
        return
    if any(os.getenv(k) for k in PROVIDER_KEY_VARS):
        return
    raise HTTPException(
        status_code=503,
        detail='No LLM configured. Set OPENAI_API_KEY (or another provider '
               'key) in the environment / .env file, or use a local model: '
               "MODEL=ollama_chat/qwen3:8b docker compose --profile ollama "
               'up web ollama')


class ChatRequest(BaseModel):
    doc_id: str
    question: str


TREE_SEARCH_PROMPT = """\
You are given a question and the tree structure of a document.
Find all nodes that are likely to contain the answer.

Question: {question}

Document tree structure:
{tree}

Reply in the following JSON format:
{{
  "thinking": "<your reasoning about which nodes are relevant>",
  "node_list": ["<node_id1>", "<node_id2>"]
}}
"""

ANSWER_PROMPT = """\
Answer the question using ONLY the document excerpts below.
If the excerpts do not contain the answer, say so.
Answer in the same language as the question.

Question: {question}

Document excerpts:
{context}
"""


def _uploads_dir():
    path = os.path.join(WORKSPACE, 'uploads')
    os.makedirs(path, exist_ok=True)
    return path


def _to_indexable(upload_path):
    """Convert any uploaded file to something PageIndexClient can index
    (.pdf or .md), reusing the ingest.py pipeline."""
    ext = os.path.splitext(upload_path)[1].lower()
    base = os.path.splitext(upload_path)[0]
    if ext in MARKDOWN_EXTS:
        return upload_path
    if ext == '.pdf':
        if not pdf_is_scanned(upload_path):
            return upload_path
        md_path = base + '.md'
        ocr_pdf_to_markdown(upload_path, md_path, OCR_LANG)
        ensure_headers(md_path, os.path.basename(base))
        return md_path
    md_path = base + '.md'
    if ext in PANDOC_EXTS:
        convert_with_pandoc(upload_path, md_path)
    elif ext in MARKITDOWN_EXTS:
        convert_with_markitdown(upload_path, md_path)
    else:
        try:
            convert_with_markitdown(upload_path, md_path)
        except Exception:
            convert_with_pandoc(upload_path, md_path)
    ensure_headers(md_path, os.path.basename(base))
    return md_path


def _walk_nodes(nodes):
    for node in nodes or []:
        yield node
        yield from _walk_nodes(node.get('nodes'))


def _node_context(doc, node):
    """Best available text for a node: stored text (md), page contents
    (pdf), or the summary as a last resort."""
    if node.get('text'):
        return node['text']
    start, end = node.get('start_index'), node.get('end_index')
    pages = doc.get('pages')
    if pages and start and end:
        page_map = {p['page']: p['content'] for p in pages}
        return '\n'.join(page_map.get(p, '') for p in range(start, end + 1))
    return node.get('summary', '')


@app.get('/')
def home():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'webui', 'index.html'))


@app.get('/api/documents')
def list_documents():
    return [
        {'doc_id': doc_id,
         'doc_name': doc.get('doc_name', ''),
         'doc_description': doc.get('doc_description', ''),
         'type': doc.get('type', '')}
        for doc_id, doc in client.documents.items()
    ]


@app.post('/api/documents')
def upload_document(file: UploadFile):
    _require_llm()
    filename = os.path.basename(file.filename or 'document')
    upload_path = os.path.join(_uploads_dir(), filename)
    with open(upload_path, 'wb') as out:
        shutil.copyfileobj(file.file, out)
    try:
        indexable = _to_indexable(upload_path)
        doc_id = client.index(indexable)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f'Indexing failed: {e}')
    doc = client.documents[doc_id]
    return {'doc_id': doc_id,
            'doc_name': doc.get('doc_name', filename),
            'type': doc.get('type', '')}


@app.post('/api/chat')
def chat(req: ChatRequest):
    _require_llm()
    if req.doc_id not in client.documents:
        raise HTTPException(status_code=404, detail='Document not found')
    tree = client.get_document_structure(req.doc_id)

    # Step 1 — LLM tree search: which nodes can answer the question?
    search = llm_completion(client.model, TREE_SEARCH_PROMPT.format(
        question=req.question, tree=tree))
    if not search:
        raise HTTPException(
            status_code=502,
            detail='LLM call failed after retries — check your API key, '
                   'MODEL setting, and network (see server logs).')
    node_ids = [str(n) for n in extract_json(search).get('node_list', [])]

    doc = client.documents[req.doc_id]
    selected = [n for n in _walk_nodes(doc.get('structure'))
                if str(n.get('node_id')) in node_ids]
    if not selected:
        # Small models sometimes return an empty node list even when the
        # document is relevant (single-node docs especially). Answer from
        # the whole tree rather than refusing.
        selected = list(_walk_nodes(doc.get('structure')))
    if not selected:
        return {'answer': 'Ce document ne contient aucun contenu indexé. / '
                          'This document has no indexed content.',
                'sources': []}

    # Step 2 — answer from the selected nodes' content only.
    context = '\n\n'.join(
        f"[{n.get('title', '?')}]\n{_node_context(doc, n)}" for n in selected)
    if len(context) > CONTEXT_MAX_CHARS:
        context = context[:CONTEXT_MAX_CHARS] + '\n[... truncated ...]'
    answer = llm_completion(client.model, ANSWER_PROMPT.format(
        question=req.question, context=context))

    sources = [{'node_id': n.get('node_id'), 'title': n.get('title'),
                'pages': ([n.get('start_index'), n.get('end_index')]
                          if n.get('start_index') else None),
                'line_num': n.get('line_num')}
               for n in selected]
    return {'answer': answer, 'sources': sources}
