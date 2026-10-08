from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import openai
from openai import OpenAI

from . import state
from .config import Config, Model, Provider, base_model, load_config, routing_order

COOLDOWN_SECONDS = 60
ERROR_COOLDOWN_SECONDS = 120
AUTH_COOLDOWN_SECONDS = 600
FORBIDDEN_COOLDOWN_SECONDS = 3600
PAYMENT_COOLDOWN_SECONDS = 6 * 3600
LONG_RETRY_SECONDS = 300
MAX_PARALLEL = 3

DAILY_HINTS = ("per day", "perday", "per-day", "daily", "rpd", "tokens per day", "monthly", "per month")
MINUTE_HINTS = ("per minute", "perminute", "per-min", "rpm", "tpm")
ACCOUNT_HINTS = (
    "paused", "suspended", "billing", "purchase", "insufficient", "no credits", "out of credits",
    "card_required", "requires_payment", "card verification",
)
REASONING_TAGS = re.compile(r"<(think|thinking|thought)>.*?</\1>", re.S | re.I)


class NoModelAvailable(RuntimeError):
    pass


@dataclass
class Result:
    response: Any
    provider: Provider
    model: Model

    @property
    def message(self) -> Any:
        return self.response.choices[0].message

    @property
    def text(self) -> str:
        return REASONING_TAGS.sub("", self.message.content or "").strip()

    @property
    def label(self) -> str:
        return f"{self.provider.name}/{self.model.id}"


def _retry_after(err: openai.APIStatusError) -> float | None:
    value = err.response.headers.get("retry-after") if err.response is not None else None
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _short(err: Exception) -> str:
    text = getattr(err, "message", None) or str(err)
    return " ".join(text.split())[:300]


def _clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%I:%M %p").lstrip("0")


class Router:
    def __init__(self, config: Config | None = None):
        self.config = config or load_config()
        self.providers = state.with_discovered(self.config.providers)
        self._clients: dict[str, OpenAI] = {}

    def client(self, provider: Provider) -> OpenAI:
        if provider.name not in self._clients:
            headers: dict = dict(provider.headers)
            if provider.keyless:
                # Anonymous endpoints like OVH reject any Authorization header, even a dummy one.
                headers["Authorization"] = openai.Omit()
            self._clients[provider.name] = OpenAI(
                base_url=provider.base_url,
                api_key=provider.api_key,
                default_headers=headers or None,
                max_retries=0,
                timeout=provider.timeout,
            )
        return self._clients[provider.name]

    def candidates(
        self, session: str | None = None, force: str | None = None, needs_tools: bool = False
    ) -> list[tuple[Provider, Model]]:
        pairs = routing_order([p for p in self.providers if p.available], self.config.routing, state.last_used())
        if needs_tools:
            pairs = [(p, m) for p, m in pairs if m.tools]
        if force:
            exact = [(p, m) for p, m in pairs if force in (p.name, m.id, f"{p.name}/{m.id}")]
            pairs = exact or [(p, m) for p, m in pairs if base_model(force) == base_model(m.id)]
            if not pairs:
                raise NoModelAvailable(f"No configured model with a key matches '{force}'")
            return pairs
        sticky = state.get_session(session) if session else None
        if sticky:
            pairs.sort(key=lambda pm: (pm[0].name, pm[1].id) != sticky)
        return pairs

    def chat(
        self,
        messages: list[dict] | None = None,
        *,
        request_id: int,
        session: str | None = None,
        force: str | None = None,
        prepare: Callable[[Provider, Model], list[dict] | None] | None = None,
        **params: Any,
    ) -> Result:
        """Send one chat completion, falling through the chain until a model answers.

        `prepare` builds the messages for each candidate (e.g. trimmed to its context size);
        returning None skips that model. When the newest attempt has not answered after
        `hedge_after` seconds, the next candidate from another provider starts alongside it
        and the first answer wins; the slower attempts finish in the background.
        """
        remaining = self.candidates(session, force, needs_tools=bool(params.get("tools")))
        finished: queue.Queue = queue.Queue()
        winner = threading.Lock()
        in_flight: dict[str, float] = {}  # provider name -> start time; one attempt per provider at a time
        dead_providers: set[str] = set()
        tried = 0
        too_small = 0
        hedge = self.config.hedge_after

        def start_next() -> bool:
            nonlocal messages, tried, too_small
            i = 0
            while i < len(remaining):
                provider, model = remaining[i]
                if provider.name in in_flight:
                    i += 1
                    continue
                del remaining[i]
                if provider.name in dead_providers:
                    continue
                status, _, _ = state.model_state(provider.name, model.id)
                if status != "ready" and not force:
                    continue
                if prepare:
                    messages = prepare(provider, model)
                    if messages is None:
                        too_small += 1
                        continue

                tried += 1
                label = f"{provider.name}/{model.id}"
                state.set_activity(request_id, f"Waiting for {label}", provider.name, model.id)
                state.add_event(request_id, f"Trying {label}")
                in_flight[provider.name] = time.monotonic()
                threading.Thread(
                    target=self._attempt,
                    args=(finished, winner, request_id, session, provider, model, messages, params),
                    daemon=True,
                ).start()
                return True
            return False

        while in_flight or start_next():
            timeout = None
            if hedge and len(in_flight) < MAX_PARALLEL:
                timeout = max(0.0, max(in_flight.values()) + hedge - time.monotonic())
            try:
                provider, result, err, dead = finished.get(timeout=timeout)
            except queue.Empty:
                if start_next():
                    state.add_event(request_id, f"No answer after {hedge:.0f}s, trying another provider alongside")
                    continue
                # Nothing else can start until a running attempt finishes.
                provider, result, err, dead = finished.get()
            del in_flight[provider.name]
            if result:
                return result
            if err:
                raise err
            if dead:
                dead_providers.add(provider.name)

        if tried == 0 and too_small:
            raise NoModelAvailable(
                f"The conversation is too long for all {too_small} models that are available right now."
            )
        if tried == 0:
            raise NoModelAvailable(
                "Every configured model is used up, cooling down, or missing an API key. Run `fr status`."
            )
        raise NoModelAvailable(f"All {tried} models that were available failed. Run `fr status` for details.")

    def _attempt(
        self,
        finished: queue.Queue,
        winner: threading.Lock,
        request_id: int,
        session: str | None,
        provider: Provider,
        model: Model,
        messages: list[dict] | None,
        params: dict,
    ) -> None:
        """One request in a background thread. Reports (provider, result, unexpected error, provider dead)."""
        started = time.monotonic()
        try:
            call = {**params, "extra_headers": {"Authorization": openai.Omit()}} if provider.keyless else params
            resp = self.client(provider).chat.completions.create(model=model.id, messages=messages, **call)
            if not getattr(resp, "choices", None):
                raise openai.APIError("Response had no choices", request=None, body=None)
        except openai.APIError as err:
            finished.put((provider, None, None, self._on_error(request_id, provider, model, started, err)))
            return
        except Exception as err:
            finished.put((provider, None, err, False))
            return

        label = f"{provider.name}/{model.id}"
        latency = time.monotonic() - started
        usage = getattr(resp, "usage", None)
        state.record_attempt(
            request_id, provider.name, model.id, "ok", latency,
            getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
        )
        state.mark_ready(provider.name, model.id)
        if not winner.acquire(blocking=False):
            state.add_event(request_id, f"{label} answered in {latency:.1f}s, after another model had already answered")
            return
        if session:
            state.set_session(session, provider.name, model.id)
        state.add_event(request_id, f"{label} answered in {latency:.1f}s", "success")
        state.set_activity(request_id, f"Got response from {label}", provider.name, model.id)
        finished.put((provider, Result(resp, provider, model), None, False))

    def _on_error(
        self, request_id: int, provider: Provider, model: Model, started: float, err: openai.APIError
    ) -> bool:
        """Record a failed attempt. Returns True when the whole provider is unusable for now."""
        if isinstance(err, openai.RateLimitError):
            if any(h in str(err).lower() for h in ACCOUNT_HINTS):
                self._on_failure(
                    request_id, provider, model, started, err, "account_blocked",
                    status="exhausted", cooldown=PAYMENT_COOLDOWN_SECONDS, whole_provider=True,
                )
                return True
            self._on_rate_limit(request_id, provider, model, err, started)
            return False
        if isinstance(err, openai.NotFoundError):
            self._on_failure(request_id, provider, model, started, err, "not_found", status="invalid")
            return False
        if isinstance(err, openai.AuthenticationError):
            self._on_failure(
                request_id, provider, model, started, err, "auth_error",
                status="error", cooldown=AUTH_COOLDOWN_SECONDS, whole_provider=True,
            )
            return True
        if isinstance(err, openai.PermissionDeniedError):
            if any(h in str(err).lower() for h in ACCOUNT_HINTS):
                self._on_failure(
                    request_id, provider, model, started, err, "account_blocked",
                    status="exhausted", cooldown=PAYMENT_COOLDOWN_SECONDS, whole_provider=True,
                )
                return True
            # 403 is usually one model being off-limits for this key, not a bad key.
            self._on_failure(
                request_id, provider, model, started, err, "forbidden",
                status="error", cooldown=FORBIDDEN_COOLDOWN_SECONDS,
            )
            return False
        if isinstance(err, openai.APIStatusError):
            if err.status_code == 402:
                self._on_failure(
                    request_id, provider, model, started, err, "payment_required",
                    status="exhausted", cooldown=PAYMENT_COOLDOWN_SECONDS,
                    whole_provider=provider.shared_limits,
                )
                return False
            if 400 <= err.status_code < 500:
                self._on_failure(request_id, provider, model, started, err, "bad_request")
                return False
        self._on_failure(
            request_id, provider, model, started, err, "error",
            status="cooldown", cooldown=ERROR_COOLDOWN_SECONDS,
        )
        return False

    def _on_rate_limit(
        self, request_id: int, provider: Provider, model: Model, err: openai.RateLimitError, started: float
    ) -> None:
        text = _short(err)
        lower = text.lower()
        retry_after = _retry_after(err)
        per_minute = any(h in lower for h in MINUTE_HINTS)
        daily = any(h in lower for h in DAILY_HINTS) and not per_minute
        now = time.time()

        if daily or (retry_after is not None and retry_after > LONG_RETRY_SECONDS):
            status = "exhausted"
            until = now + retry_after if retry_after else provider.next_reset().timestamp()
            summary = f"used up until {_clock(until)}"
        else:
            status = "cooldown"
            until = now + (retry_after or COOLDOWN_SECONDS)
            summary = f"rate limited, cooling down for {int(until - now)}s"

        # Upstream-capacity 429s affect one model; account-level limits hit every model.
        shared = provider.shared_limits and (daily or per_minute)
        targets = [m.id for m in provider.models] if shared else [model.id]
        state.set_model_status(provider.name, targets, status, until, text)
        state.record_attempt(request_id, provider.name, model.id, "rate_limited", time.monotonic() - started, error=text)
        scope = f"all {provider.name} models" if shared else f"{provider.name}/{model.id}"
        state.add_event(request_id, f"{scope} {summary}", "warning")

    def _on_failure(
        self,
        request_id: int,
        provider: Provider,
        model: Model,
        started: float,
        err: Exception,
        outcome: str,
        status: str | None = None,
        cooldown: float | None = None,
        whole_provider: bool = False,
    ) -> None:
        text = _short(err)
        if status:
            until = time.time() + cooldown if cooldown else None
            targets = [m.id for m in provider.models] if whole_provider else [model.id]
            state.set_model_status(provider.name, targets, status, until, text)
        state.record_attempt(request_id, provider.name, model.id, outcome, time.monotonic() - started, error=text)
        state.add_event(request_id, f"{provider.name}/{model.id} failed ({outcome}): {text[:140]}", "error")
