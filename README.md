# AI Reporting Agent

A local AI agent that lets you query, visualise, and extend your data using natural language. Built on Claude, LangGraph, Cube.js, and a semantic config library.

---

## What it does

- **Ask for charts** — "show me revenue by country as a bar chart" → agent queries Cube, renders Chart.js
- **Add dimensions & measures** — "add a bi-monthly dimension" → agent opens an interactive form, pre-fills a best-guess SQL expression, lets you test against live data before committing
- **Schema validation** — changes are validated before hitting the DB; Cube compile errors surface immediately in the UI
- **Conversation memory** — history is summarised automatically to keep token usage low
- **Session recovery** — corrupted thread state (e.g. page refresh mid-tool-call) is detected and auto-cleared

---

## Architecture

```
Browser (Chat UI)
      │  SSE  │  HTTP
      ▼
  FastAPI (agent/ui.py)
      │
  LangGraph ReAct agent  ←──  Claude (one model, default Haiku — see CLAUDE_MODEL)
      │
  ┌───┴─────────────────────┐
  │   MCP Tool Servers       │
  │  cube-mcp  (port 5001)  │  ← Cube.js queries + schema reload
  │  library-mcp (port 5002)│  ← Cube config CRUD + staging
  └───────────────────────┬─┘
                          │
               ┌──────────┴──────────┐
               │                     │
           Cube.js               Library API
          (port 4000)            (port 3001)
               │                     │
           Postgres              Postgres
         (data, 5432)         (configs, 5433)
```

---

## Quick start

**Prerequisites:** Docker, Python 3.11+, an Anthropic API key.

```bash
# 1. Clone
git clone https://github.com/rohitjn26/ai-reporting-agent.git
cd ai-reporting-agent

# 2. Install Python dependencies
make install

# 3. Add your API key
cp agent/.env.example agent/.env
# edit agent/.env and set ANTHROPIC_API_KEY=...

# 4. Start the stack (Postgres, Cube, Library, MCP servers) and seed data
make up

# 5. Run the UI
make ui
# Open http://localhost:8501
```

---

## Project structure

```
agent/          Python agent — FastAPI UI, LangGraph ReAct loop, chart renderer
  graph/        LangGraph agent, config editor tool, summarisation
  chart/        Chart.js renderer + local HTTP server
cube/           Cube.js container — dynamic schema loaded from library
  data-models/  Schema builder (JS) that compiles library configs into Cube definitions
data-db/        Postgres init SQL — sample e-commerce schema + seed data
library/        FastAPI semantic config store — CRUD API for cube definitions
mcp/            FastMCP servers exposing Cube and library tools to the agent
docker-compose.yml
Makefile
```

---

## Key make targets

| Command | What it does |
|---|---|
| `make up` | Build and start all Docker services, seed configs |
| `make ui` | Launch the chat UI at http://localhost:8501 |
| `make down` | Stop all services and remove volumes |
| `make logs` | Tail all service logs |
| `make ps` | Show service status and ports |
| `make test` | Run the full test suite (unit + e2e) |
| `make test-unit` | Run fast unit tests only (no Docker) |
| `make test-e2e` | Run the end-to-end test (disposable Postgres via Docker) |

---

## Testing

Test dependencies live in `requirements-dev.txt`; install them into the same venv `make install` created:

```bash
make install-dev          # pytest, pytest-asyncio, pg8000, testcontainers
make test-unit            # fast, no Docker
make test                 # unit + e2e (e2e needs a running Docker daemon)
```

- **Unit tests** (`tests/unit/`) cover pure logic with no external services: the Chart.js/HTML
  renderer, agent model routing and prompt merging, Cube SQL parameter inlining, cube-config
  field validation, and the interactive config-editor's form parsing (network + `interrupt`
  are mocked).
- **End-to-end test** (`tests/e2e/`) is self-contained: it spins up a throwaway Postgres via
  [testcontainers](https://testcontainers.com/), runs the FastAPI library app against it, points
  the library MCP tools at that live server, and drives the full
  create → preview → commit → delete config lifecycle — no mocks, no dev-stack dependency. It
  skips automatically if Docker is unavailable.

---

## Performance notes

The agent is tuned so Anthropic prompt caching holds across the conversation:

- **One model for the whole loop.** Every turn — data queries and config edits
  alike — runs on a single model (`CLAUDE_MODEL`, default Haiku). Prompt caches
  are model-scoped, so mixing models on one thread would cold-start the cache on
  every switch. Config edits are safe on Haiku because `edit_cube_config` only
  pre-fills a form the user reviews before anything commits. Set `CLAUDE_MODEL`
  to run the whole loop on a stronger model.
- **Frozen system prompt.** The conversation summary is stored as a history
  message, not concatenated into the system prompt, so the cached prefix stays
  byte-identical turn to turn.
- **Cached tools+system prefix.** The system prompt is sent as a `cache_control`
  block; since tools render before it, that one breakpoint caches the tool
  definitions + system prompt together (~6.2K tokens, over Haiku's 4096-token
  minimum). It's served from cache on every tool round-trip and every turn.
- **Reused clients.** LLM clients are memoized so the HTTP connection pool stays
  warm instead of re-doing a TLS handshake per tool call.

## How the config edit flow works

1. User asks to add a measure or dimension (or agent detects a missing one)
2. Agent calls `edit_cube_config` — form opens in the chat UI, pre-filled with a SQL suggestion
3. User clicks **Test SQL** to validate the expression against live Postgres data
4. User clicks **Apply change** — agent shows a preview card (what's being added, no raw JSON)
5. User confirms → committed to library DB → Cube restarts → change is live
