"""Shared state in SQLite, so the CLI, the agent and the dashboard all see the same picture."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from .config import HOME, Config, Model, Provider, routing_order

DB_PATH = HOME / "data" / "state.db"
STALE_AFTER = 600

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_status (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    until REAL,
    last_error TEXT,
    updated REAL NOT NULL,
    PRIMARY KEY (provider, model)
);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    prompt TEXT NOT NULL,
    session TEXT,
    status TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    activity TEXT,
    error TEXT,
    started REAL NOT NULL,
    updated REAL NOT NULL,
    ended REAL
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    outcome TEXT NOT NULL,
    latency REAL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    error TEXT,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER,
    level TEXT NOT NULL,
    message TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS discovered (
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    position INTEGER NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (provider, model)
);
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    root TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    plan TEXT,
    notes TEXT,
    auto_run INTEGER NOT NULL DEFAULT 0,
    provider TEXT,
    model TEXT,
    error TEXT,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS task_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    data TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS task_files (
    task_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    existed INTEGER NOT NULL,
    original BLOB,
    ts REAL NOT NULL,
    PRIMARY KEY (task_id, path)
);
CREATE INDEX IF NOT EXISTS task_messages_by_task ON task_messages (task_id, id);
CREATE INDEX IF NOT EXISTS attempts_by_model ON attempts (provider, model, ts);
CREATE INDEX IF NOT EXISTS events_by_ts ON events (ts);
"""

_schema_ready = False


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    global _schema_ready
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        if not _schema_ready:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            _schema_ready = True
        yield conn
        conn.commit()
    finally:
        conn.close()


# Requests: one user task ("ask" or one agent turn), possibly spanning several model attempts.

def start_request(kind: str, prompt: str, session: str | None = None) -> int:
    now = time.time()
    with connect() as c:
        cur = c.execute(
            "INSERT INTO requests (kind, prompt, session, status, activity, started, updated)"
            " VALUES (?, ?, ?, 'running', 'Starting', ?, ?)",
            (kind, prompt[:2000], session, now, now),
        )
        return int(cur.lastrowid)


def set_activity(request_id: int, activity: str, provider: str | None = None, model: str | None = None) -> None:
    with connect() as c:
        c.execute(
            "UPDATE requests SET activity = ?, provider = COALESCE(?, provider),"
            " model = COALESCE(?, model), updated = ? WHERE id = ?",
            (activity, provider, model, time.time(), request_id),
        )


def finish_request(request_id: int, ok: bool, error: str | None = None) -> None:
    now = time.time()
    with connect() as c:
        c.execute(
            "UPDATE requests SET status = ?, error = ?, activity = ?, ended = ?, updated = ? WHERE id = ?",
            ("ok" if ok else "failed", error, "Done" if ok else "Failed", now, now, request_id),
        )


def add_event(request_id: int | None, message: str, level: str = "info") -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO events (request_id, level, message, ts) VALUES (?, ?, ?, ?)",
            (request_id, level, message, time.time()),
        )


def record_attempt(
    request_id: int | None,
    provider: str,
    model: str,
    outcome: str,
    latency: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    error: str | None = None,
) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO attempts (request_id, provider, model, outcome, latency, prompt_tokens,"
            " completion_tokens, error, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (request_id, provider, model, outcome, latency, prompt_tokens, completion_tokens, error, time.time()),
        )


# Model availability.

def _effective(row: sqlite3.Row | None, now: float) -> tuple[str, float | None, str | None]:
    if row is None:
        return "ready", None, None
    status, until = row["status"], row["until"]
    if status != "invalid" and until is not None and until <= now:
        return "ready", None, None
    return status, until, row["last_error"]


def model_state(provider: str, model: str) -> tuple[str, float | None, str | None]:
    with connect() as c:
        row = c.execute(
            "SELECT * FROM model_status WHERE provider = ? AND model = ?", (provider, model)
        ).fetchone()
    return _effective(row, time.time())


def set_model_status(
    provider: str, models: list[str], status: str, until: float | None = None, error: str | None = None
) -> None:
    now = time.time()
    with connect() as c:
        c.executemany(
            "INSERT INTO model_status (provider, model, status, until, last_error, updated)"
            " VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (provider, model) DO UPDATE SET"
            " status = excluded.status, until = excluded.until,"
            " last_error = excluded.last_error, updated = excluded.updated",
            [(provider, m, status, until, error, now) for m in models],
        )


def mark_ready(provider: str, model: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM model_status WHERE provider = ? AND model = ?", (provider, model))


def clear_status(provider: str | None = None, model: str | None = None) -> int:
    query, args = "DELETE FROM model_status", []
    if provider:
        query += " WHERE provider = ?"
        args.append(provider)
        if model:
            query += " AND model = ?"
            args.append(model)
    with connect() as c:
        return c.execute(query, args).rowcount


# Sticky sessions keep one conversation on the same model while it has quota.

def get_session(session: str) -> tuple[str, str] | None:
    with connect() as c:
        row = c.execute("SELECT provider, model FROM sessions WHERE session = ?", (session,)).fetchone()
    return (row["provider"], row["model"]) if row else None


def set_session(session: str, provider: str, model: str) -> None:
    with connect() as c:
        c.execute(
            "INSERT INTO sessions (session, provider, model, updated) VALUES (?, ?, ?, ?)"
            " ON CONFLICT (session) DO UPDATE SET provider = excluded.provider,"
            " model = excluded.model, updated = excluded.updated",
            (session, provider, model, time.time()),
        )


# Models found by `fr sync` for providers marked `discover:` in providers.yaml.

def save_discovered(provider: str, model_ids: list[str]) -> None:
    now = time.time()
    with connect() as c:
        c.execute("DELETE FROM discovered WHERE provider = ?", (provider,))
        c.executemany(
            "INSERT INTO discovered (provider, model, position, ts) VALUES (?, ?, ?, ?)",
            [(provider, m, i, now) for i, m in enumerate(model_ids)],
        )


def last_used() -> dict[tuple[str, str], float]:
    with connect() as c:
        rows = c.execute("SELECT provider, model, MAX(ts) AS ts FROM attempts GROUP BY provider, model").fetchall()
    return {(r["provider"], r["model"]): r["ts"] for r in rows}


def with_discovered(providers: list[Provider]) -> list[Provider]:
    with connect() as c:
        rows = c.execute("SELECT provider, model FROM discovered ORDER BY provider, position").fetchall()
    found: dict[str, list[str]] = {}
    for r in rows:
        found.setdefault(r["provider"], []).append(r["model"])
    for p in providers:
        if not p.discover:
            continue
        listed = {m.id for m in p.models}
        p.models.extend(Model(id=m, discovered=True) for m in found.get(p.name, []) if m not in listed)
    return providers


# Dashboard / `fr status` snapshot.

def snapshot(config: Config) -> dict:
    now = time.time()
    local_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    providers = with_discovered(config.providers)
    with connect() as c:
        statuses = {(r["provider"], r["model"]): r for r in c.execute("SELECT * FROM model_status")}
        up_next = []
        for p, m in routing_order([p for p in providers if p.available], config.routing, last_used()):
            if _effective(statuses.get((p.name, m.id)), now)[0] == "ready":
                up_next.append({"provider": p.name, "model": m.id, "rank": m.rank, "tools": m.tools})
                if len(up_next) == 8:
                    break
        out = []
        for p in providers:
            usage = {
                r["model"]: r
                for r in c.execute(
                    "SELECT model, COUNT(*) AS attempts, SUM(outcome = 'ok') AS ok,"
                    " SUM(outcome = 'rate_limited') AS limited,"
                    " COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0) AS tokens,"
                    " MAX(ts) AS last_used FROM attempts WHERE provider = ? AND ts >= ? GROUP BY model",
                    (p.name, p.last_reset().timestamp()),
                )
            }
            models = []
            for m in p.models:
                status, until, error = _effective(statuses.get((p.name, m.id)), now)
                if not p.available:
                    status = "no_key"
                u = usage.get(m.id)
                models.append(
                    {
                        "id": m.id,
                        "rank": m.rank,
                        "tools": m.tools,
                        "discovered": m.discovered,
                        "status": status,
                        "until": until,
                        "error": error,
                        "attempts": u["attempts"] if u else 0,
                        "ok": u["ok"] if u else 0,
                        "limited": u["limited"] if u else 0,
                        "tokens": u["tokens"] if u else 0,
                        "last_used": u["last_used"] if u else None,
                    }
                )
            out.append(
                {
                    "name": p.name,
                    "label": p.label,
                    "note": p.note,
                    "has_key": p.available,
                    "key_env": p.key_env,
                    "signup": p.signup,
                    "keyless": p.keyless,
                    "trust": p.trust,
                    "discover": bool(p.discover),
                    "shared_limits": p.shared_limits,
                    "next_reset": p.next_reset().timestamp(),
                    "models": models,
                }
            )

        active = []
        for r in c.execute("SELECT * FROM requests WHERE status = 'running' ORDER BY started DESC"):
            row = dict(r)
            row["stale"] = now - row["updated"] > STALE_AFTER
            active.append(row)
        recent = [
            dict(r)
            for r in c.execute("SELECT * FROM requests WHERE status != 'running' ORDER BY ended DESC LIMIT 25")
        ]
        events = [dict(r) for r in c.execute("SELECT * FROM events ORDER BY id DESC LIMIT 100")]
        totals = dict(
            c.execute(
                "SELECT COUNT(*) AS attempts, COALESCE(SUM(outcome = 'ok'), 0) AS ok,"
                " COALESCE(SUM(outcome = 'rate_limited'), 0) AS limited,"
                " COALESCE(SUM(prompt_tokens), 0) + COALESCE(SUM(completion_tokens), 0) AS tokens"
                " FROM attempts WHERE ts >= ?",
                (local_midnight,),
            ).fetchone()
        )
        totals["requests"] = c.execute(
            "SELECT COUNT(*) FROM requests WHERE started >= ?", (local_midnight,)
        ).fetchone()[0]

    return {
        "now": now,
        "routing": config.routing,
        "providers": out,
        "up_next": up_next,
        "active": active,
        "recent": recent,
        "events": events,
        "totals": totals,
    }
