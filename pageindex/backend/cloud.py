# pageindex/backend/cloud.py
"""CloudBackend — connects to PageIndex cloud service (api.pageindex.ai).

API reference: https://github.com/VectifyAI/pageindex_sdk
"""
from __future__ import annotations
import json
import logging
import os
import re
import time
import urllib.parse
import requests
from typing import AsyncIterator

from .protocol import AgentTools
from ..errors import CloudAPIError, PageIndexError
from ..events import QueryEvent

logger = logging.getLogger(__name__)

API_BASE = "https://api.pageindex.ai"

_INTERNAL_TOOLS = frozenset({"ToolSearch", "Read", "Grep", "Glob", "Bash", "Edit", "Write"})


class CloudBackend:
    def __init__(self, api_key: str):
        self._api_key = api_key
        self._headers = {"api_key": api_key}
        self._folder_id_cache: dict[str, str | None] = {}
        self._folder_warning_shown = False

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _warn_folder_upgrade(self) -> None:
        if not self._folder_warning_shown:
            logger.warning(
                "Folders (collections) require a Max plan. "
                "All documents are stored in a single global space — collection names are ignored. "
                "Upgrade at https://dash.pageindex.ai/subscription"
            )
            self._folder_warning_shown = True

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{API_BASE}{path}"
        for attempt in range(3):
            try:
                resp = requests.request(method, url, headers=self._headers, timeout=30, **kwargs)
                if resp.status_code in (429, 500, 502, 503):
                    logger.warning("Cloud API %s %s returned %d, retrying...", method, path, resp.status_code)
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code != 200:
                    body = resp.text[:500] if resp.text else ""
                    raise CloudAPIError(f"Cloud API error {resp.status_code}: {body}")
                return resp.json() if resp.content else {}
            except requests.RequestException as e:
                if attempt == 2:
                    raise CloudAPIError(f"Cloud API request failed: {e}") from e
                time.sleep(2 ** attempt)
        raise CloudAPIError("Max retries exceeded")

    @staticmethod
    def _validate_collection_name(name: str) -> None:
        if not re.match(r'^[a-zA-Z0-9_-]{1,128}$', name):
            raise PageIndexError(
                f"Invalid collection name: {name!r}. "
                "Must be 1-128 chars of [a-zA-Z0-9_-]."
            )

    @staticmethod
    def _enc(value: str) -> str:
        return urllib.parse.quote(value, safe="")

    # ── Collection management (mapped to folders) ─────────────────────────

    def create_collection(self, name: str) -> None:
        self._validate_collection_name(name)
        try:
            resp = self._request("POST", "/folder/", json={"name": name})
            self._folder_id_cache[name] = resp.get("folder", {}).get("id")
        except CloudAPIError as e:
            if "403" in str(e):
                self._warn_folder_upgrade()
                self._folder_id_cache[name] = None
            else:
                raise

    def get_or_create_collection(self, name: str) -> None:
        self._validate_collection_name(name)
        try:
            data = self._request("GET", "/folders/")
            for folder in data.get("folders", []):
                if folder.get("name") == name:
                    self._folder_id_cache[name] = folder["id"]
                    return
            resp = self._request("POST", "/folder/", json={"name": name})
            self._folder_id_cache[name] = resp.get("folder", {}).get("id")
        except CloudAPIError as e:
            if "403" in str(e):
                self._warn_folder_upgrade()
                self._folder_id_cache[name] = None
            else:
                raise

    def _get_folder_id(self, name: str) -> str | None:
        """Resolve collection name to folder ID. Returns None if folders not available."""
        if name in self._folder_id_cache:
            return self._folder_id_cache.get(name)
        try:
            data = self._request("GET", "/folders/")
            for folder in data.get("folders", []):
                if folder.get("name") == name:
                    self._folder_id_cache[name] = folder["id"]
                    return folder["id"]
        except CloudAPIError:
            pass
        self._folder_id_cache[name] = None
        return None

    def list_collections(self) -> list[str]:
        data = self._request("GET", "/folders/")
        return [f["name"] for f in data.get("folders", [])]

    def delete_collection(self, name: str) -> None:
        folder_id = self._get_folder_id(name)
        if folder_id:
            self._request("DELETE", f"/folder/{self._enc(folder_id)}/")

    # ── Document management ───────────────────────────────────────────────

    def add_document(self, collection: str, file_path: str) -> str:
        folder_id = self._get_folder_id(collection)
        data = {"if_retrieval": "true"}
        if folder_id:
            data["folder_id"] = folder_id

        with open(file_path, "rb") as f:
            resp = self._request("POST", "/doc/", files={"file": f}, data=data)

        doc_id = resp["doc_id"]

        # Poll until indexing completes. The cloud API signals readiness via
        # status == "completed"; retrieval_ready is not a reliable indicator.
        for _ in range(120):  # 10 min max
            tree_resp = self._request("GET", f"/doc/{self._enc(doc_id)}/", params={"type": "tree"})
            status = tree_resp.get("status", "")
            if status == "completed":
                return doc_id
            if status == "failed":
                raise CloudAPIError(f"Document {doc_id} indexing failed")
            time.sleep(5)

        raise CloudAPIError(f"Document {doc_id} indexing timed out")

    def get_document(self, collection: str, doc_id: str, include_text: bool = False) -> dict:
        resp = self._request("GET", f"/doc/{self._enc(doc_id)}/metadata/")
        # Fetch structure in the same call via tree endpoint
        tree_resp = self._request("GET", f"/doc/{self._enc(doc_id)}/",
                                  params={"type": "tree", "summary": "true"})
        raw_tree = tree_resp.get("tree", tree_resp.get("structure", tree_resp.get("result", [])))
        return {
            "doc_id": resp.get("id", doc_id),
            "doc_name": resp.get("name", ""),
            "doc_description": resp.get("description", ""),
            "doc_type": "pdf",
            "status": resp.get("status", ""),
            "structure": self._normalize_tree(raw_tree),
        }

    def get_document_structure(self, collection: str, doc_id: str) -> list:
        resp = self._request("GET", f"/doc/{self._enc(doc_id)}/", params={"type": "tree", "summary": "true"})
        raw_tree = resp.get("tree", resp.get("structure", resp.get("result", [])))
        return self._normalize_tree(raw_tree)

    def get_page_content(self, collection: str, doc_id: str, pages: str) -> list:
        resp = self._request("GET", f"/doc/{self._enc(doc_id)}/", params={"type": "ocr", "format": "page"})
        # Filter to requested pages
        from ..index.utils import parse_pages
        page_nums = set(parse_pages(pages))
        all_pages = resp.get("pages", resp.get("ocr", resp.get("result", [])))
        if isinstance(all_pages, list):
            return [
                {"page": p.get("page", p.get("page_index")),
                 "content": p.get("content", p.get("markdown", ""))}
                for p in all_pages
                if p.get("page", p.get("page_index")) in page_nums
            ]
        return []

    @staticmethod
    def _normalize_tree(nodes: list) -> list:
        """Normalize cloud tree nodes to match local schema."""
        result = []
        for node in nodes:
            normalized = {
                "title": node.get("title", ""),
                "node_id": node.get("node_id", ""),
                "summary": node.get("summary", node.get("prefix_summary", "")),
                "start_index": node.get("start_index", node.get("page_index")),
                "end_index": node.get("end_index", node.get("page_index")),
            }
            if "text" in node:
                normalized["text"] = node["text"]
            children = node.get("nodes", [])
            if children:
                normalized["nodes"] = CloudBackend._normalize_tree(children)
            result.append(normalized)
        return result

    def list_documents(self, collection: str) -> list[dict]:
        folder_id = self._get_folder_id(collection)
        params = {"limit": 100}
        if folder_id:
            params["folder_id"] = folder_id
        data = self._request("GET", "/docs/", params=params)
        return [
            {"doc_id": d.get("id", ""), "doc_name": d.get("name", ""), "doc_type": "pdf"}
            for d in data.get("documents", [])
        ]

    def delete_document(self, collection: str, doc_id: str) -> None:
        self._request("DELETE", f"/doc/{self._enc(doc_id)}/")

    # ── Query (uses cloud chat/completions, no LLM key needed) ────────────

    def query(self, collection: str, question: str, doc_ids: list[str] | None = None) -> str:
        """Non-streaming query via cloud chat/completions."""
        doc_id = doc_ids if doc_ids else self._get_all_doc_ids(collection)
        resp = self._request("POST", "/chat/completions/", json={
            "messages": [{"role": "user", "content": question}],
            "doc_id": doc_id,
            "stream": False,
        })
        # Extract answer from response
        choices = resp.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return resp.get("content", resp.get("answer", ""))

    async def query_stream(self, collection: str, question: str,
                           doc_ids: list[str] | None = None) -> AsyncIterator[QueryEvent]:
        """Streaming query via cloud chat/completions SSE.

        Events are yielded in real-time as they arrive from the server.
        A background thread handles the blocking HTTP stream and pushes
        events through an asyncio.Queue for true async streaming.
        """
        import asyncio
        import threading

        doc_id = doc_ids if doc_ids else self._get_all_doc_ids(collection)
        headers = self._headers
        queue: asyncio.Queue[QueryEvent | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _stream():
            """Background thread: read SSE and push events to queue."""
            resp = requests.post(
                f"{API_BASE}/chat/completions/",
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": question}],
                    "doc_id": doc_id,
                    "stream": True,
                    "stream_metadata": True,
                },
                stream=True,
                timeout=120,
            )
            try:
                if resp.status_code != 200:
                    body = resp.text[:500] if resp.text else ""
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        QueryEvent(type="answer_done",
                                   data=f"Cloud streaming error {resp.status_code}: {body}"),
                    )
                    return

                current_tool_name = None
                current_tool_args: list[str] = []

                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    meta = chunk.get("block_metadata", {})
                    block_type = meta.get("type", "")
                    choices = chunk.get("choices", [])
                    delta = choices[0].get("delta", {}) if choices else {}
                    content = delta.get("content", "")

                    if block_type == "mcp_tool_use_start":
                        current_tool_name = meta.get("tool_name", "")
                        current_tool_args = []

                    elif block_type == "tool_use":
                        if content:
                            current_tool_args.append(content)

                    elif block_type == "tool_use_stop":
                        if current_tool_name and current_tool_name not in _INTERNAL_TOOLS:
                            args_str = "".join(current_tool_args)
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                QueryEvent(type="tool_call", data={
                                    "name": current_tool_name,
                                    "args": args_str,
                                }),
                            )
                        current_tool_name = None
                        current_tool_args = []

                    elif block_type == "text" and content:
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            QueryEvent(type="answer_delta", data=content),
                        )

            finally:
                resp.close()
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = threading.Thread(target=_stream, daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event is None:
                break
            yield event

        thread.join(timeout=5)

    def _get_all_doc_ids(self, collection: str) -> list[str]:
        """Get all document IDs in a collection."""
        docs = self.list_documents(collection)
        return [d["doc_id"] for d in docs]

    # ── Not used in cloud mode ────────────────────────────────────────────

    def get_agent_tools(self, collection: str, doc_ids: list[str] | None = None) -> AgentTools:
        """Not used in cloud mode — query goes through chat/completions."""
        return AgentTools()
