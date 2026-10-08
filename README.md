# freeroute

One local router for many free LLM APIs. You add whatever API keys you have; `freeroute` sends each
request to the best model that still has free quota, and moves on to the next one when a model is
rate-limited, used up, slow, or broken.

It comes with:

- `fr ask` — ask a question, answered by the first model that works.
- `fr code` — a coding agent that edits files in a folder and keeps going when a model runs out of
  quota (the next model continues from the saved plan and notes).
- `fr dashboard` — a live web dashboard showing every provider, model, quota and request.

There are 32 providers configured in `providers.yaml`. You do not need all of them; any provider
without a key is skipped.

---

## Setup (about 5 minutes)

**Requirements:** Python 3.11 or newer, git, and [uv](https://docs.astral.sh/uv/getting-started/installation/)
(recommended). Install uv on macOS or Linux with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**1. Get the code and install it**

```bash
git clone https://github.com/kunaltomar597/freellm.git
cd freellm
uv sync
```

`uv sync` creates a `.venv` folder in the project and installs everything into it. That folder is
specific to each machine and is never committed.

Without uv:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Keep the `-e` (editable install). freeroute looks for `.env`, `providers.yaml` and `data/` in the
project folder, and an editable install is what points it there.

**2. Add your API keys**

```bash
cp .env.example .env
```

Open `.env` and paste in the keys you have. Every provider has its signup link above its line.
Leave the rest empty. Two providers need no key and are switched on with a `1`:

```
KILO_ENABLED=1
OVH_ENABLED=1
```

Cloudflare needs both `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.

A good starting set (all free, quick signup): `GEMINI_API_KEY`, `GROQ_API_KEY`, `NVIDIA_API_KEY`,
`OPENROUTER_API_KEY`, `CEREBRAS_API_KEY`.

**3. Check it works**

```bash
uv run fr keys                         # which providers have a key
uv run fr ask "Reply with exactly: ok" # should print "ok" and which model answered
```

That's it. If you used pip instead of uv, replace `uv run fr` with `.venv/bin/fr` everywhere.

To have `fr` available in any terminal without `uv run`, install it as a tool from the project folder:

```bash
uv tool install -e .
```

---

## Usage

```bash
uv run fr ask "Explain Python decorators in 3 lines"
uv run fr ask -m groq "Same question, but only Groq"          # force a provider, model, or provider/model
git diff | uv run fr ask "Review this diff"                    # piped input is appended to the prompt

uv run fr code -C ~/projects/myapp "Add a /health endpoint and a test for it"
uv run fr code -C ~/projects/myapp -y "..."                    # -y: run shell commands without asking
uv run fr code --task 3                                        # continue an earlier task

uv run fr dashboard --open                                     # http://127.0.0.1:4000 (--port to change)
```

Other commands:

| Command | What it does |
|---|---|
| `fr status` | Every model with a key: ready, cooling down, or used up until when; today's usage |
| `fr status --all` | Same, including providers without a key |
| `fr keys` | Every provider, its signup link, and whether its key is set |
| `fr models [provider]` | Model IDs each key can actually reach (useful when editing `providers.yaml`) |
| `fr sync [provider]` | Refresh model lists for providers marked `discover:` |
| `fr reset [provider[/model]]` | Clear "used up" / "cooling down" marks |

---

## Configuration

Everything lives in the project folder:

| File | Purpose | In git? |
|---|---|---|
| `.env` | Your API keys | **No** — never commit it |
| `.env.example` | Template with every key name and signup link | Yes |
| `providers.yaml` | Providers, models, their ranking and routing settings | Yes |
| `data/state.db` | Local usage history, cooldowns, coding-agent tasks (created on first run) | No |

Settings at the top of `providers.yaml`:

- `routing: rank` tries the smartest available model first across all providers; `routing: order`
  tries providers top to bottom.
- `hedge_after: 20` — if a model has not answered after 20 seconds, the next provider is tried
  alongside it (up to 3 at once) and the first answer wins. Set to `0` to try one model at a time.

Per provider you can set `timeout`, `context` (input budget in tokens for the coding agent), and per
model `rank` (lower is smarter) and `tools: false` (the coding agent skips models that cannot call
tools). The comments at the top of `providers.yaml` describe every field.

To keep config and data somewhere else, set `FREEROUTE_HOME=/path/to/folder`; that folder then needs
its own `.env` and `providers.yaml`.

Free-tier limits are per key. Running the same keys on two machines shares one daily quota.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `0 of 32 providers ready` | `.env` is missing or empty. It must sit next to `providers.yaml`. |
| `Every configured model is used up, cooling down, or missing an API key` | Run `fr status` to see when quotas reset, add more keys, or `fr reset` if a mark is stale. |
| A provider always fails with `auth_error` | The key is wrong or revoked; replace it in `.env`. |
| A model shows `not usable` | The provider removed it. Run `fr models <provider>` and update `providers.yaml`. |
| `fr: command not found` | Use `uv run fr` (or `.venv/bin/fr`), or install it with `uv tool install -e .` |
| Port 4000 in use | `fr dashboard --port 4010` |

---

## Setup instructions for coding agents

If you are an AI coding agent (Cursor, Claude Code, Codex, ...) asked to set this project up, follow
these steps in order from the repository root and stop to report if one fails.

1. Check `python3 --version` is 3.11 or newer. Check `uv --version`; if uv is missing, install it with
   `curl -LsSf https://astral.sh/uv/install.sh | sh` (ask the user first if you need permission to
   install software).
2. Run `uv sync`. Do not install with plain `pip install .` (non-editable); freeroute must resolve its
   config from this folder.
3. If `.env` does not exist, run `cp .env.example .env`. Never overwrite an existing `.env`.
4. Ask the user for their API keys, or tell them to paste them into `.env` themselves. Write each key
   only on its existing `NAME=` line. Never print key values in chat output, logs, or commits, and
   never commit `.env` (it is already in `.gitignore`).
5. Run `uv run fr keys` and report how many providers are ready.
6. Verify end to end with `uv run fr ask "Reply with exactly: ok"`. Success means the output contains
   `ok` and a line `[answered by <provider>/<model>]`. If it fails, run `uv run fr status` and report
   the providers' states.
7. Optionally start the dashboard in the background with `uv run fr dashboard` and confirm
   `http://127.0.0.1:4000` returns HTTP 200.

Do not edit `providers.yaml` during setup unless the user asks; it works as shipped.
