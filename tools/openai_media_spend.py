"""Fail-closed OpenAI media spend gate and per-profile ledger.

This module is deliberately limited to the two OpenAI API capabilities approved
for Hermes gateway agents: image generation and speech transcription.  It never
reads an admin key and never calls OpenAI.  The OpenAI Costs API reconciliation
remains the authority for actual billed spend.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

_ALLOWED = {"image_generation", "transcription"}
_TZ = ZoneInfo(os.getenv("TZ", "Europe/Zagreb"))
_PRICING_SOURCE = "https://developers.openai.com/api/docs/pricing"
_PRICING_CHECKED_AT = "2026-07-13"
_STT_INPUT_PER_M = 2.50
_STT_OUTPUT_PER_M = 10.00
_IMAGE_TEXT_INPUT_PER_M = 5.00
_IMAGE_INPUT_PER_M = 8.00
_IMAGE_CACHED_INPUT_PER_M = 2.00
_IMAGE_OUTPUT_PER_M = 30.00
_IMAGE_OUTPUT_FALLBACK_USD = {
    ("low", "1024x1024"): 0.009,
    ("low", "1024x1536"): 0.013,
    ("low", "1536x1024"): 0.013,
    ("medium", "1024x1024"): 0.034,
    ("medium", "1024x1536"): 0.050,
    ("medium", "1536x1024"): 0.050,
    ("high", "1024x1024"): 0.133,
    ("high", "1024x1536"): 0.165,
    ("high", "1536x1024"): 0.165,
}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_spend_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  iso_time TEXT NOT NULL,
  channel TEXT NOT NULL DEFAULT 'api',
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  caller TEXT NOT NULL,
  operation TEXT NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  cached_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  estimated_usd REAL NOT NULL DEFAULT 0,
  pricing_checked_at TEXT,
  pricing_source TEXT,
  estimated INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'recorded',
  metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_api_spend_events_ts ON api_spend_events(ts);
CREATE INDEX IF NOT EXISTS idx_api_spend_events_model ON api_spend_events(model);
"""


class SpendPolicyError(RuntimeError):
    """Raised before an API call when policy, tracking, or budget is invalid."""


def _policy_operations() -> set[str]:
    raw = os.getenv("OPENAI_API_ALLOWED_OPERATIONS", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _require_operation(operation: str) -> None:
    if operation not in _ALLOWED:
        raise SpendPolicyError(f"OpenAI API operation is not implemented by the media-only gate: {operation}")
    configured = _policy_operations()
    if operation not in configured:
        raise SpendPolicyError(f"OpenAI API operation is not allowed for this profile: {operation}")


def _caller() -> str:
    caller = os.getenv("HERMES_API_SPEND_CALLER", "").strip()
    if not caller:
        raise SpendPolicyError("HERMES_API_SPEND_CALLER is missing; refusing un-attributed OpenAI API spend")
    return caller


def _ledger_path() -> Path:
    raw = os.getenv("API_SPEND_LEDGER", "").strip()
    if not raw:
        raise SpendPolicyError("API_SPEND_LEDGER is missing; refusing untracked OpenAI API spend")
    return Path(raw).expanduser()


def _connect() -> sqlite3.Connection:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=15)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con


def _day_bounds() -> tuple[float, float]:
    now = datetime.now(_TZ)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def gate(operation: str, model: str, estimated_usd: float, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fail closed before a paid call if policy/tracking/budget is unavailable."""
    _require_operation(operation)
    caller = _caller()
    estimate = max(0.0, float(estimated_usd))
    hard = float(os.getenv("API_SPEND_DAILY_HARD_USD", os.getenv("HERMES_API_DAILY_HARD_USD", "5.00")))
    start, end = _day_bounds()
    try:
        with _connect() as con:
            row = con.execute(
                "SELECT COALESCE(SUM(estimated_usd),0) AS s FROM api_spend_events "
                "WHERE ts >= ? AND ts < ? AND status NOT IN ('rejected','smoke')",
                (start, end),
            ).fetchone()
            today = float(row["s"] or 0.0)
    except Exception as exc:
        raise SpendPolicyError(f"OpenAI spend ledger unavailable: {type(exc).__name__}") from exc
    projected = today + estimate
    if projected > hard:
        raise SpendPolicyError(
            f"OpenAI daily hard cap would be exceeded: ${projected:.4f} > ${hard:.2f}"
        )
    return {
        "ok": True,
        "caller": caller,
        "operation": operation,
        "model": model,
        "estimated_call_usd": estimate,
        "today_before_usd": today,
        "today_projected_usd": projected,
        "hard_usd": hard,
        "metadata": dict(metadata or {}),
    }


def record(
    operation: str,
    model: str,
    estimated_usd: float,
    *,
    input_tokens: int = 0,
    cached_tokens: int = 0,
    output_tokens: int = 0,
    estimated: bool,
    status: str = "recorded",
    metadata: Mapping[str, Any] | None = None,
) -> int:
    """Record one attributed media API call; failure is surfaced to the caller."""
    _require_operation(operation)
    caller = _caller()
    now = datetime.now(_TZ)
    try:
        with _connect() as con:
            cur = con.execute(
                """
                INSERT INTO api_spend_events
                (ts, iso_time, channel, provider, model, caller, operation,
                 input_tokens, cached_tokens, output_tokens, estimated_usd,
                 pricing_checked_at, pricing_source, estimated, status, metadata_json)
                VALUES (?, ?, 'api', 'openai', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(), now.isoformat(timespec="seconds"), model, caller, operation,
                    max(0, int(input_tokens)), max(0, int(cached_tokens)), max(0, int(output_tokens)),
                    max(0.0, float(estimated_usd)), _PRICING_CHECKED_AT, _PRICING_SOURCE,
                    1 if estimated else 0, status, json.dumps(dict(metadata or {}), sort_keys=True),
                ),
            )
            row_id = cur.lastrowid
            if row_id is None:
                raise SpendPolicyError("OpenAI spend record returned no row id")
            return int(row_id)
    except Exception as exc:
        raise SpendPolicyError(f"OpenAI spend record failed: {type(exc).__name__}") from exc


def audio_duration_seconds(file_path: str) -> float:
    """Return media duration via ffprobe; fail closed if duration cannot be read."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", file_path],
            capture_output=True, text=True, timeout=10, check=True,
        )
        duration = float(proc.stdout.strip())
    except Exception as exc:
        raise SpendPolicyError(f"Cannot determine audio duration for spend gate: {type(exc).__name__}") from exc
    if duration <= 0:
        raise SpendPolicyError("Cannot determine positive audio duration for spend gate")
    return duration


def transcription_preflight_usd(duration_seconds: float) -> float:
    # Official minute-equivalent price is $0.006; 25% headroom covers token variance.
    return max(0.0001, float(duration_seconds) / 60.0 * 0.006 * 1.25)


def image_preflight_usd(quality: str, size: str, source_count: int) -> float:
    output = _IMAGE_OUTPUT_FALLBACK_USD.get((quality, size))
    if output is None:
        raise SpendPolicyError(f"Unknown gpt-image-2 quality/size pricing: {quality}/{size}")
    # Conservative allowance: prompt text + up to ~8k image tokens per reference.
    return output + 0.001 + max(0, int(source_count)) * 0.065


def _as_mapping(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        return obj
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        value = dump()
        if isinstance(value, Mapping):
            return value
    return {}


def usage_mapping(response: Any) -> Mapping[str, Any]:
    return _as_mapping(getattr(response, "usage", None))


def transcription_cost(response: Any, duration_seconds: float) -> tuple[float, dict[str, int], bool]:
    usage = usage_mapping(response)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    if input_tokens or output_tokens:
        cost = (input_tokens * _STT_INPUT_PER_M + output_tokens * _STT_OUTPUT_PER_M) / 1_000_000
        return cost, {"input_tokens": input_tokens, "cached_tokens": 0, "output_tokens": output_tokens}, False
    cost = max(0.0001, float(duration_seconds) / 60.0 * 0.006)
    return cost, {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}, True


def image_cost(response: Any, quality: str, size: str, source_count: int, prompt: str) -> tuple[float, dict[str, int], bool, dict[str, Any]]:
    usage = usage_mapping(response)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    input_details = _as_mapping(usage.get("input_tokens_details"))
    text_tokens = int(input_details.get("text_tokens", 0) or 0)
    image_tokens = int(input_details.get("image_tokens", 0) or 0)
    cached_tokens = int(input_details.get("cached_tokens", 0) or 0)
    if input_tokens or output_tokens:
        unknown_input = max(0, input_tokens - text_tokens - image_tokens)
        fresh_image = max(0, image_tokens - cached_tokens)
        unknown_input_rate = _IMAGE_INPUT_PER_M if source_count else _IMAGE_TEXT_INPUT_PER_M
        cost = (
            text_tokens * _IMAGE_TEXT_INPUT_PER_M
            + unknown_input * unknown_input_rate
            + fresh_image * _IMAGE_INPUT_PER_M
            + cached_tokens * _IMAGE_CACHED_INPUT_PER_M
            + output_tokens * _IMAGE_OUTPUT_PER_M
        ) / 1_000_000
        details = {"text_tokens": text_tokens, "image_tokens": image_tokens, "unknown_input_tokens": unknown_input}
        return cost, {"input_tokens": input_tokens, "cached_tokens": cached_tokens, "output_tokens": output_tokens}, False, details
    fallback = image_preflight_usd(quality, size, source_count)
    details = {"fallback": "official output calculator plus conservative input allowance", "prompt_chars": len(prompt)}
    return fallback, {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}, True, details
