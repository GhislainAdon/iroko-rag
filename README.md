<div align="center">

# 🌳 iroko-rag

**Chat with any document — vectorless, reasoning-based RAG, ready to use.**

*Named after the iroko, the sacred great tree of West Africa: a fitting home for a tree-index RAG.*

Fork of [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex) that adds universal document ingestion (Office, scanned PDFs, plain text), a web chat UI + HTTP API, Docker packaging, local models via Ollama / llama.cpp, and fixes for several verified upstream bugs.

</div>

---

## ✨ Features

| | |
|---|---|
| 🌲 **Vectorless RAG** | No embeddings, no vector DB. Documents become a semantic **tree** (like a smart table of contents); an LLM *reasons* its way to the right sections. From upstream PageIndex. |
| 📥 **Universal ingestion** | PDF (native **and scanned** — Tesseract OCR), Word, PowerPoint, Excel, EPUB, HTML, plain `.txt` (structure recovered heuristically), Markdown… one command for any format. |
| 💬 **Chat without coding** | Built-in web UI: upload → ask → sourced answers. Plus an open HTTP API for your Angular/React/anything frontend. |
| 🦙 **Any LLM** | OpenAI/Anthropic/… (via LiteLLM), **Ollama** (with auto-pull of missing models), or any OpenAI-compatible server (llama.cpp, LM Studio, vLLM). |
| 🐳 **Docker-ready** | One image with all system deps (Pandoc, Tesseract). `demo`, `web`, `test`, `ollama` compose services. |
| 🔧 **Hardened** | 5 verified upstream bugs fixed with regression tests (36 tests), LLM call throttling, robust JSON parsing for small local models. |

---

## 🚀 Quick start (2 minutes)

```bash
git clone https://github.com/GhislainAdon/iroko-rag.git
cd iroko-rag
cp env.example .env        # put your OPENAI_API_KEY — or set MODEL=ollama_chat/gemma4:e2b for 100% local
docker compose up web
```

Open **http://localhost:8000** → upload a document → ask questions. That's it, no RAG code to write.

> 100% local, no API key: install [Ollama](https://ollama.com), set `MODEL=ollama_chat/gemma4:e2b` in `.env` — if the model isn't downloaded yet, **the server pulls it automatically** at startup.

---

## 📖 Usage examples

### 1. Web chat (no code)

```bash
docker compose up web        # → http://localhost:8000
```

Upload a PDF, a Word contract, a PowerPoint deck or even a scanned document; pick it in the list; ask *"quels sont les livrables ?"* — the answer cites the exact sections it used (`📎 Livrables attendus`).

### 2. HTTP API

```bash
# Index a document (any format)
curl -X POST http://localhost:8000/api/documents -F "file=@rapport.docx"
# → {"doc_id": "e29efba8-...", "doc_name": "rapport", "type": "md"}

# Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "e29efba8-...", "question": "Quel est le budget ?"}'
# → {"answer": "Le budget alloué est de 15 000 EUR...",
#    "sources": [{"node_id": "0004", "title": "Budget", ...}]}

# List documents / check server & model status
curl http://localhost:8000/api/documents
curl http://localhost:8000/api/health
```

Interactive API docs: **http://localhost:8000/api/docs**

### 3. Command line — build a tree index from any file

```bash
# Office documents (Pandoc keeps the heading hierarchy)
python3 ingest.py --input contrat.docx
python3 ingest.py --input presentation.pptx

# Scanned PDF (OCR — French)
python3 ingest.py --input scan.pdf --ocr-lang fra

# Plain text: ALL-CAPS lines & 'Section :' lines become tree nodes
python3 ingest.py --input notes.txt

# Just convert to Markdown, no LLM calls
python3 ingest.py --input rapport.docx --convert-only
```

Each run writes `results/<name>_structure.json` — the tree index. Native PDF/Markdown also work with the original entry point (`run_pageindex.py --pdf_path doc.pdf`).

### 4. Python

```python
from pageindex.client import PageIndexClient

client = PageIndexClient(model="ollama_chat/gemma4:e2b", workspace="./workspace")
doc_id = client.index("rapport.pdf")

print(client.get_document_structure(doc_id))   # the tree (titles + summaries)
print(client.get_page_content(doc_id, "5-7"))  # raw content of pages 5-7
```

### 5. Your own frontend (React / Angular / vanilla)

CORS is open — call the API directly from any dev server.

<details>
<summary><b>Vanilla JS</b></summary>

```html
<script>
async function ask(docId, question) {
  const r = await fetch('http://localhost:8000/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doc_id: docId, question })
  });
  const { answer, sources } = await r.json();
  console.log(answer, sources);
}
</script>
```
</details>

<details>
<summary><b>React</b></summary>

```jsx
import { useState } from 'react';

export function DocChat({ docId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  async function send(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question) return;
    setMessages(m => [...m, { role: 'user', text: question }]);
    setInput('');
    const r = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id: docId, question }),
    });
    const { answer, sources } = await r.json();
    setMessages(m => [...m, { role: 'bot', text: answer, sources }]);
  }

  return (
    <div>
      {messages.map((m, i) => <p key={i}><b>{m.role}:</b> {m.text}</p>)}
      <form onSubmit={send}>
        <input value={input} onChange={e => setInput(e.target.value)} />
        <button>Send</button>
      </form>
    </div>
  );
}
```
</details>

<details>
<summary><b>Angular</b></summary>

```ts
// doc-chat.service.ts
import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatResponse {
  answer: string;
  sources: { node_id: string; title: string; pages?: [number, number] }[];
}

@Injectable({ providedIn: 'root' })
export class DocChatService {
  private http = inject(HttpClient);
  private api = 'http://localhost:8000/api';

  documents(): Observable<any[]> {
    return this.http.get<any[]>(`${this.api}/documents`);
  }

  upload(file: File): Observable<{ doc_id: string }> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<{ doc_id: string }>(`${this.api}/documents`, form);
  }

  ask(docId: string, question: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.api}/chat`,
      { doc_id: docId, question });
  }
}
```
</details>

---

## 🧠 Choosing your LLM

Everything goes through [LiteLLM](https://docs.litellm.ai/docs/providers) — set `MODEL` in `.env`:

| Option | `.env` | Notes |
|---|---|---|
| **OpenAI** (default) | `OPENAI_API_KEY=sk-...` | best quality on complex PDFs |
| **Anthropic, Gemini, …** | `ANTHROPIC_API_KEY=...` + `MODEL=anthropic/claude-sonnet-4-6` | any LiteLLM provider |
| **Ollama** (local, free) | `MODEL=ollama_chat/gemma4:e2b` | missing models are **pulled automatically**; progress at `/api/health` |
| **llama.cpp / LM Studio / vLLM** | `MODEL=openai/local` + `OPENAI_API_BASE=http://host.docker.internal:8080/v1` + `OPENAI_API_KEY=dummy` | any OpenAI-compatible endpoint |

**Reusing models already downloaded by Ollama with llama.cpp** (they are plain GGUF blobs):

```bash
ollama show --modelfile gemma4:e2b | grep FROM
# FROM /root/.ollama/models/blobs/sha256-abc123...
llama-server -m /root/.ollama/models/blobs/sha256-abc123... --port 8080
```

> ⚠️ One-way only: Ollama's model directory is a content-addressed store (nameless blobs + manifests), **not** a folder of `.gguf` files. Never copy hand-downloaded models into it — keep them in an ordinary folder (e.g. `./models/`) and point `llama-server -m` at them.

Tuning for small local models: `PAGEINDEX_MAX_CONCURRENCY` (default 10) caps concurrent LLM calls; `CONTEXT_MAX_CHARS` (default 40000) caps answer context. Responses that wrap JSON in prose (common with small models) are handled by a balanced-brace fallback parser.

---

## 🌲 How it works

1. **Indexing** — the document is converted to a hierarchical **tree**: every section becomes a node with a title, a summary, and its location (pages or lines).

```jsonc
{
  "title": "Financial Stability",
  "node_id": "0006",
  "start_index": 21,
  "end_index": 22,
  "summary": "The Federal Reserve ...",
  "nodes": [
    { "title": "Monitoring Financial Vulnerabilities", "node_id": "0007", ... },
    { "title": "Domestic and International Cooperation", "node_id": "0008", ... }
  ]
}
```

2. **Retrieval** — no embeddings: the LLM reads the tree (titles + summaries) and *reasons* about which nodes can answer the question — like a human using a table of contents.

3. **Answer** — a second LLM call answers **only** from the selected sections and returns them as sources. If the selection comes back empty (small local models sometimes do that), the server falls back to the whole tree instead of refusing.

Ideal for long, structured documents: financial reports, contracts, tenders, technical manuals, academic papers.

---

## 🔧 Fixed upstream issues

Each one verified in code and covered by [`tests/test_upstream_issues.py`](tests/test_upstream_issues.py):

| Upstream issue | Fix |
|---|---|
| [#330](https://github.com/VectifyAI/PageIndex/issues/330) `get_leaf_nodes` KeyError on leaf nodes | `.get('nodes')` — `clean_node()` deletes the key on leaves |
| [#326](https://github.com/VectifyAI/PageIndex/issues/326) `extract_json` crash on non-strict model JSON (DeepSeek, Ollama, …) | balanced-brace fallback extracts the first `{...}`/`[...]` from prose-wrapped responses |
| [#283](https://github.com/VectifyAI/PageIndex/issues/283) unthrottled concurrent LLM calls → HTTP 429 | per-event-loop semaphore, `PAGEINDEX_MAX_CONCURRENCY` env |
| [#279](https://github.com/VectifyAI/PageIndex/issues/279)/[#296](https://github.com/VectifyAI/PageIndex/issues/296) `get_page_content` over-collects on Markdown comma lists | exact line matching instead of the `[min, max]` window |
| [#245](https://github.com/VectifyAI/PageIndex/issues/245) Markdown preamble silently dropped, headerless docs yield zero nodes | preamble captured as a node (frontmatter excluded); headerless docs become a single root node |

---

## 🧪 Tests & quality eval

```bash
docker compose run --rm test     # 36 unit/regression tests, no API key needed
python3 eval_chat.py             # end-to-end chat eval against the live server
```

The eval asks known questions about indexed documents and checks expected keywords and cited sources — use it to compare models (`gemma4:e2b` vs `mistral` vs GPT-4o) on *your* documents.

---

## 📁 Repository layout

```
ingest.py            # universal ingestion CLI (any format → tree)
server.py            # FastAPI: upload + chat API, serves the web UI
webui/index.html     # dependency-free chat page
eval_chat.py         # chat quality eval
docker-compose.yml   # web / demo / test / ollama services
pageindex/           # core engine (tree building, retrieval) — from upstream
examples/            # sample documents & tutorials — from upstream
tests/               # 36 regression tests
```

---

## 🙏 Credits & license

Core engine by [Vectify AI](https://vectify.ai) — [PageIndex](https://github.com/VectifyAI/PageIndex) ([docs](https://docs.pageindex.ai), [Discord](https://discord.com/invite/VuXuf29EUj)). They also offer a hosted [API & dashboard](https://pageindex.ai) with a proprietary long-context OCR.

Fork maintained by [@GhislainAdon](https://github.com/GhislainAdon). Same license as upstream — see [LICENSE](LICENSE).
