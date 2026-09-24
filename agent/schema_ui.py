"""Standalone launcher for the Schema-discovery tab — no agent, no Docker.

Reuses the same page and endpoints as ui.py, but skips the agent lifespan so
you can try schema discovery without the full stack running.

    cd agent && source .venv/bin/activate && python3 schema_ui.py

Open http://localhost:8502 and use the **Schema** tab. (The Chat tab is inert
here — run ui.py with the stack up for that.)
"""
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

import ui  # module import only; does not build the agent or start Docker

app = FastAPI()

app.get("/", response_class=HTMLResponse)(lambda: ui._HTML)
app.get("/discovery/list")(ui.discovery_list)
app.get("/discovery/browse")(ui.discovery_browse)
app.post("/discovery/run")(ui.discovery_run)
app.post("/discovery/semantic")(ui.discovery_semantic)


if __name__ == "__main__":
    print("\n  Schema discovery → http://localhost:8502  (open the Schema tab)\n")
    uvicorn.run(app, host="127.0.0.1", port=8502)
