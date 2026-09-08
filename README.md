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
  LangGraph ReAct agent  ←──  Claude (claude-sonnet-4-6)
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

---

## How the config edit flow works

1. User asks to add a measure or dimension (or agent detects a missing one)
2. Agent calls `edit_cube_config` — form opens in the chat UI, pre-filled with a SQL suggestion
3. User clicks **Test SQL** to validate the expression against live Postgres data
4. User clicks **Apply change** — agent shows a preview card (what's being added, no raw JSON)
5. User confirms → committed to library DB → Cube restarts → change is live
