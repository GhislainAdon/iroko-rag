from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


CANDIDATE_FIELDS = ("domain", "topic")
MEMBERSHIP_LIMIT = 3
SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SemanticFolderPlanError(ValueError):
    pass


@dataclass(frozen=True)
class SemanticFolderBuildItem:
    item_id: str
    title: str
    summary: str
    domain: Any = None
    topic: Any = None


@dataclass(frozen=True)
class SemanticFolderMembership:
    item_id: str
    file_ref: str
    relative_path: str
    confidence: float | None = None
    canonical_segments: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class SemanticFolderValidatedPlan:
    template: list[str]
    canonical_values: list[dict[str, str]]
    memberships: list[SemanticFolderMembership]
    skipped: list[dict[str, str]]
    raw_plan: dict[str, Any]


class SemanticFolderPlanner(Protocol):
    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class OpenAISemanticFolderPlanner:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
    ):
        self.model = (
            model
            or os.environ.get("PIFS_SEMANTIC_FOLDER_MODEL")
            or os.environ.get("PIFS_METADATA_MODEL")
            or "gpt-5-nano"
        )
        self.base_url = (
            base_url
            if base_url is not None
            else os.environ.get("PIFS_METADATA_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
        )

    def plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        api_key = (
            os.environ.get("PIFS_SEMANTIC_FOLDER_API_KEY")
            or os.environ.get("PIFS_METADATA_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise SemanticFolderPlanError(
                "PIFS_SEMANTIC_FOLDER_API_KEY, PIFS_METADATA_API_KEY, or OPENAI_API_KEY "
                "is required for PIFS Semantic Folder planning"
            )

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=self.base_url or None)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Plan a PIFS Semantic Folder from document-level metadata. "
                        "Use only the provided transient item ids, title, summary, domain, and topic. "
                        "Do not infer from storage paths or original folders. "
                        "Choose a useful field/value folder template using domain and topic, "
                        "canonicalize display values, provide path-safe slugs, and reduce each "
                        "document to at most three semantic memberships. Return strict JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format=self._response_format(),
        )
        return json.loads(response.choices[0].message.content or "{}")

    @staticmethod
    def _response_format() -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "pifs_semantic_folder_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["template", "canonical_values", "memberships", "skipped"],
                    "properties": {
                        "template": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(CANDIDATE_FIELDS)},
                        },
                        "canonical_values": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["field", "display", "slug"],
                                "properties": {
                                    "field": {"type": "string", "enum": list(CANDIDATE_FIELDS)},
                                    "display": {"type": "string"},
                                    "slug": {"type": "string"},
                                },
                            },
                        },
                        "memberships": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["item_id", "paths"],
                                "properties": {
                                    "item_id": {"type": "string"},
                                    "paths": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "confidence": {"type": ["number", "null"]},
                                },
                            },
                        },
                        "skipped": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["item_id", "reason"],
                                "properties": {
                                    "item_id": {"type": "string"},
                                    "reason": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        }


def semantic_mount_path(source_scope: str) -> str:
    source_scope = _normalize_path(source_scope)
    return "/semantic" if source_scope == "/" else f"{source_scope}/semantic"


def validate_semantic_folder_plan(
    plan: dict[str, Any],
    *,
    item_file_refs: dict[str, str],
) -> SemanticFolderValidatedPlan:
    if not isinstance(plan, dict):
        raise SemanticFolderPlanError("Semantic Folder planner returned a non-object plan")
    template = _validate_template(plan.get("template"))
    canonical_values = _validate_canonical_values(plan.get("canonical_values"))
    canonical_lookup = {
        (item["field"], item["slug"]): item for item in canonical_values
    }
    memberships: list[SemanticFolderMembership] = []
    seen_item_paths: set[tuple[str, str]] = set()
    per_item_count: dict[str, int] = {}
    for item in _required_list(plan.get("memberships"), "memberships"):
        if not isinstance(item, dict):
            raise SemanticFolderPlanError("Semantic Folder membership entries must be objects")
        item_id = str(item.get("item_id") or "").strip()
        if item_id not in item_file_refs:
            raise SemanticFolderPlanError(f"Unknown Semantic Folder build item: {item_id}")
        paths = item.get("paths")
        if not isinstance(paths, list):
            raise SemanticFolderPlanError(f"Semantic Folder membership {item_id} paths must be a list")
        confidence = _optional_float(item.get("confidence"))
        for raw_path in paths:
            relative_path, canonical_segments = _validate_membership_path(
                raw_path,
                template=template,
                canonical_lookup=canonical_lookup,
            )
            key = (item_id, relative_path)
            if key in seen_item_paths:
                raise SemanticFolderPlanError(
                    f"Duplicate Semantic Folder membership for {item_id}: {relative_path}"
                )
            seen_item_paths.add(key)
            per_item_count[item_id] = per_item_count.get(item_id, 0) + 1
            if per_item_count[item_id] > MEMBERSHIP_LIMIT:
                raise SemanticFolderPlanError(
                    f"Semantic Folder membership limit exceeded for {item_id}: "
                    f"max {MEMBERSHIP_LIMIT}"
                )
            memberships.append(
                SemanticFolderMembership(
                    item_id=item_id,
                    file_ref=item_file_refs[item_id],
                    relative_path=relative_path,
                    confidence=confidence,
                    canonical_segments=canonical_segments,
                )
            )
    skipped = _validate_skipped(plan.get("skipped"), item_file_refs)
    if not memberships:
        raise SemanticFolderPlanError("No useful Semantic Folder hierarchy was planned")
    return SemanticFolderValidatedPlan(
        template=template,
        canonical_values=canonical_values,
        memberships=memberships,
        skipped=skipped,
        raw_plan=plan,
    )


def _validate_template(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SemanticFolderPlanError("Semantic Folder plan template must select at least one field")
    template: list[str] = []
    for field in value:
        field = str(field)
        if field not in CANDIDATE_FIELDS:
            raise SemanticFolderPlanError(f"Unsupported Semantic Folder field: {field}")
        if field in template:
            raise SemanticFolderPlanError(f"Duplicate Semantic Folder template field: {field}")
        template.append(field)
    return template


def _validate_canonical_values(value: Any) -> list[dict[str, str]]:
    rows = _required_list(value, "canonical_values")
    seen_slug: dict[tuple[str, str], str] = {}
    canonical_values: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SemanticFolderPlanError("Semantic Folder canonical values must be objects")
        field = str(row.get("field") or "").strip()
        display = str(row.get("display") or "").strip()
        slug = str(row.get("slug") or "").strip()
        if field not in CANDIDATE_FIELDS:
            raise SemanticFolderPlanError(f"Unsupported Semantic Folder canonical field: {field}")
        if not display:
            raise SemanticFolderPlanError("Semantic Folder canonical display value is required")
        _validate_segment(slug, label=f"{field} slug")
        key = (field, slug)
        previous = seen_slug.get(key)
        if previous is not None and previous != display:
            raise SemanticFolderPlanError(
                f"Semantic Folder segment collision for {field}/{slug}: "
                f"{previous!r} and {display!r}"
            )
        seen_slug[key] = display
        canonical_values.append({"field": field, "display": display, "slug": slug})
    return canonical_values


def _validate_membership_path(
    value: Any,
    *,
    template: list[str],
    canonical_lookup: dict[tuple[str, str], dict[str, str]],
) -> tuple[str, list[dict[str, str]]]:
    raw_path = str(value or "").strip()
    if not raw_path:
        raise SemanticFolderPlanError("Semantic Folder membership path is required")
    if raw_path.startswith("/"):
        raise SemanticFolderPlanError(f"Semantic Folder membership path must be relative: {raw_path}")
    parts = raw_path.split("/")
    if len(parts) % 2:
        raise SemanticFolderPlanError(
            f"Semantic Folder membership path must use field/value segments: {raw_path}"
        )
    canonical_segments: list[dict[str, str]] = []
    fields = parts[0::2]
    values = parts[1::2]
    if fields != template[: len(fields)]:
        raise SemanticFolderPlanError(
            f"Semantic Folder membership path does not match selected template: {raw_path}"
        )
    for field, slug in zip(fields, values):
        _validate_segment(field, label="field segment")
        _validate_segment(slug, label=f"{field} value segment")
        if field not in CANDIDATE_FIELDS:
            raise SemanticFolderPlanError(f"Unsupported Semantic Folder field segment: {field}")
        canonical = canonical_lookup.get((field, slug))
        if canonical is None:
            raise SemanticFolderPlanError(
                f"Semantic Folder path uses undeclared canonical value: {field}/{slug}"
            )
        canonical_segments.append(canonical)
    return "/".join(parts), canonical_segments


def _validate_segment(segment: str, *, label: str) -> None:
    if not segment or segment in {".", ".."}:
        raise SemanticFolderPlanError(f"Unsafe Semantic Folder {label}: {segment!r}")
    if "/" in segment or "\\" in segment or "=" in segment:
        raise SemanticFolderPlanError(f"Unsafe Semantic Folder {label}: {segment!r}")
    if segment.lower() in {"unknown", "misc", "uncategorized"}:
        raise SemanticFolderPlanError(
            f"Semantic Folder plan must skip missing values instead of using {segment!r}"
        )
    if not SEGMENT_RE.fullmatch(segment):
        raise SemanticFolderPlanError(f"Unsafe Semantic Folder {label}: {segment!r}")


def _validate_skipped(value: Any, item_file_refs: dict[str, str]) -> list[dict[str, str]]:
    skipped: list[dict[str, str]] = []
    for row in _required_list(value, "skipped"):
        if not isinstance(row, dict):
            raise SemanticFolderPlanError("Semantic Folder skipped entries must be objects")
        item_id = str(row.get("item_id") or "").strip()
        if item_id not in item_file_refs:
            raise SemanticFolderPlanError(f"Unknown skipped Semantic Folder build item: {item_id}")
        reason = str(row.get("reason") or "").strip() or "skipped"
        skipped.append({"item_id": item_id, "reason": reason})
    return skipped


def _required_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise SemanticFolderPlanError(f"Semantic Folder plan {name} must be a list")
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticFolderPlanError("Semantic Folder confidence must be numeric") from exc


def _normalize_path(path: str) -> str:
    parts = [part for part in str(path or "/").replace("\\", "/").split("/") if part and part != "."]
    return "/" + "/".join(parts) if parts else "/"
