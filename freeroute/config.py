from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

HOME = Path(os.environ.get("FREEROUTE_HOME", Path(__file__).resolve().parent.parent))
load_dotenv(HOME / ".env")

RESET_ZONES = {"pacific": "America/Los_Angeles", "utc": "UTC"}
DEFAULT_RANK = 150
DEFAULT_CONTEXT = 32000


@dataclass
class Model:
    id: str
    rank: int = DEFAULT_RANK
    tools: bool = True
    discovered: bool = False
    context: int | None = None


@dataclass
class Provider:
    name: str
    label: str
    base_url_template: str
    key_env: str
    signup: str = ""
    reset: str = "utc"
    shared_limits: bool = False
    timeout: float = 120
    context: int = DEFAULT_CONTEXT
    headers: dict[str, str] = field(default_factory=dict)
    note: str = ""
    keyless: bool = False
    trust: str = ""
    discover: dict | None = None
    models: list[Model] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return os.path.expandvars(self.base_url_template)

    @property
    def api_key(self) -> str | None:
        value = os.environ.get(self.key_env, "").strip()
        if self.keyless:
            # Keyless providers are opt-in: KEY_ENV=1 enables them, the key itself is a placeholder.
            return "anonymous" if value and value not in ("0", "false", "no") else None
        return value or None

    @property
    def available(self) -> bool:
        # An unresolved ${VAR} in the URL means a required value is missing.
        return self.api_key is not None and "$" not in self.base_url

    def next_reset(self) -> datetime:
        tz = ZoneInfo(RESET_ZONES.get(self.reset, "UTC"))
        now = datetime.now(tz)
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def last_reset(self) -> datetime:
        return self.next_reset() - timedelta(days=1)

    def context_of(self, model: Model) -> int:
        return model.context or self.context


@dataclass
class Config:
    routing: str
    providers: list[Provider]
    hedge_after: float = 0


def _model(entry: str | dict) -> Model:
    if isinstance(entry, dict):
        return Model(
            id=str(entry["id"]),
            rank=int(entry.get("rank", DEFAULT_RANK)),
            tools=bool(entry.get("tools", True)),
            context=int(entry["context"]) if entry.get("context") else None,
        )
    return Model(id=str(entry))


def load_config(path: Path | None = None) -> Config:
    load_dotenv(HOME / ".env")  # pick up keys added since startup
    raw = yaml.safe_load((path or HOME / "providers.yaml").read_text()) or {}
    providers = []
    for entry in raw.get("providers", []):
        if entry.get("enabled", True) is False:
            continue
        providers.append(
            Provider(
                name=entry["name"],
                label=entry.get("label", entry["name"]),
                base_url_template=entry["base_url"],
                key_env=entry["key_env"],
                signup=entry.get("signup", ""),
                reset=entry.get("reset", "utc"),
                shared_limits=bool(entry.get("shared_limits", False)),
                timeout=float(entry.get("timeout", 120)),
                context=int(entry.get("context", DEFAULT_CONTEXT)),
                headers=entry.get("headers") or {},
                note=entry.get("note", ""),
                keyless=bool(entry.get("keyless", False)),
                trust=entry.get("trust", ""),
                discover=entry.get("discover"),
                models=[_model(m) for m in entry.get("models") or []],
            )
        )
    return Config(
        routing=raw.get("routing", "rank"),
        providers=providers,
        hedge_after=float(raw.get("hedge_after") or 0),
    )


def load_providers(path: Path | None = None) -> list[Provider]:
    return load_config(path).providers


def base_model(model_id: str) -> str:
    """The same model across providers: 'z-ai/glm-5.3', 'glm-5.3:free' and 'glm-5.3' all give 'glm-5.3'."""
    name = model_id.lower().rsplit("/", 1)[-1]
    return name.removesuffix(":free").replace(":", "-")


def routing_order(
    providers: list[Provider], routing: str, last_used: dict[tuple[str, str], float] | None = None
) -> list[tuple[Provider, Model]]:
    """Every (provider, model) pair in the order the router will try them."""
    pairs = [(p, m) for p in providers for m in p.models]
    if routing == "rank":
        # Equal ranks take turns (least recently used first) so every provider's quota gets spent.
        used = last_used or {}
        pairs.sort(key=lambda pm: (pm[1].rank, used.get((pm[0].name, pm[1].id), 0.0)))
    return pairs
