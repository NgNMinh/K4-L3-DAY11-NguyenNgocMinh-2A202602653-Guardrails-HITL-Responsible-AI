"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store request text and its start time; return its correlation id."""
        request_id = request_id or uuid.uuid4().hex
        started = time.perf_counter()
        self._open[request_id] = started
        self.logs.append({
            "request_id": request_id,
            "user_id": str(user_id),
            "input": text,
            "started_at": utc_now_iso(),
        })
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete the matching request entry with its decision and latency."""
        if request_id is None:
            request_id = next(
                (entry["request_id"] for entry in reversed(self.logs)
                 if entry.get("user_id") == str(user_id)
                 and entry["request_id"] in self._open),
                None,
            )
        started = self._open.pop(request_id, None) if request_id else None
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 3)
            if started is not None else None
        )
        entry = next(
            (item for item in reversed(self.logs)
             if request_id and item.get("request_id") == request_id),
            None,
        )
        if entry is None:
            entry = {
                "request_id": request_id or uuid.uuid4().hex,
                "user_id": str(user_id),
                "started_at": None,
            }
            self.logs.append(entry)
        entry.update({
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
            "completed_at": utc_now_iso(),
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
