"""
Generate architecture diagrams as SVG (rendered to PDF/PNG via rsvg-convert).
Three diagrams: the agent system, the eval harness, and the config-edit
human-in-the-loop flow. LLM calls are red; human-in-the-loop pauses are amber.

Run:  python docs/make_diagrams.py
Then: rsvg-convert -f pdf -o docs/architecture_agent.pdf docs/architecture_agent.svg
"""
from __future__ import annotations
import html

# ── palette ───────────────────────────────────────────────────────────────────
STYLES = {
    "llm":     ("#fee2e2", "#ef4444"),   # LLM call — red
    "service": ("#dbeafe", "#2563eb"),   # service/process — blue
    "data":    ("#e5e7eb", "#4b5563"),   # datastore — gray
    "tool":    ("#dcfce7", "#16a34a"),   # tool/code — green
    "app":     ("#ede9fe", "#7c3aed"),   # app/entry — purple
    "human":   ("#fef3c7", "#d97706"),   # human-in-the-loop pause — amber
    "lane":    ("#f8fafc", "#cbd5e1"),   # background lane
}


class SVG:
    def __init__(self, w, h, title):
        self.w, self.h = w, h
        self.parts = []
        self.boxes = {}
        self.parts.append(
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="white"/>'
        )
        self.text(w / 2, 34, title, size=21, weight="bold", anchor="middle", color="#0f172a")

    def esc(self, s):
        return html.escape(str(s))

    def text(self, x, y, s, size=13, weight="normal", anchor="middle", color="#0f172a", italic=False):
        style = f'font-family="system-ui, sans-serif" font-size="{size}" font-weight="{weight}"'
        if italic:
            style += ' font-style="italic"'
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" {style}>{self.esc(s)}</text>'
        )

    def lane(self, x, y, w, h, label):
        fill, stroke = STYLES["lane"]
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5" stroke-dasharray="6 4"/>'
        )
        self.text(x + 14, y + 24, label, size=14, weight="bold", anchor="start", color="#475569")

    def box(self, name, x, y, w, h, lines, kind="service", dashed=False, badge=None):
        fill, stroke = STYLES[kind]
        dash = ' stroke-dasharray="7 4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="2"{dash}/>'
        )
        if isinstance(lines, str):
            lines = [lines]
        # vertically center the block of lines
        lh = 17
        total = lh * len(lines)
        start = y + h / 2 - total / 2 + 13
        for i, ln in enumerate(lines):
            weight = "bold" if i == 0 else "normal"
            size = 13 if i == 0 else 11.5
            color = "#0f172a" if i == 0 else "#334155"
            self.text(x + w / 2, start + i * lh, ln, size=size, weight=weight, color=color)
        if badge:
            self.parts.append(
                f'<rect x="{x + w - 44}" y="{y + 6}" width="38" height="16" rx="8" '
                f'fill="#ef4444"/>'
            )
            self.text(x + w - 25, y + 18, badge, size=10, weight="bold", color="white")
        self.boxes[name] = (x, y, w, h)

    def anchor(self, name, side):
        x, y, w, h = self.boxes[name]
        return {
            "top": (x + w / 2, y), "bottom": (x + w / 2, y + h),
            "left": (x, y + h / 2), "right": (x + w, y + h / 2),
        }[side]

    def arrow(self, p1, p2, label=None, dashed=False, color="#334155", label_dy=-5):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(
            f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="{color}" '
            f'stroke-width="1.8" marker-end="url(#arw)"{dash}/>'
        )
        if label:
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            if p1[0] == p2[0]:  # vertical: label beside the line, not across it
                self.text(mx + 6, my + 4, label, size=10.5, color="#64748b", anchor="start")
            else:
                self.text(mx, my + label_dy, label, size=10.5, color="#64748b")

    def polyline(self, pts, label=None, dashed=False, color="#334155"):
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        d = " ".join(f"{x},{y}" for x, y in pts)
        self.parts.append(
            f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.8" '
            f'marker-end="url(#arw)"{dash}/>'
        )
        if label:
            mx, my = pts[0]
            self.text(mx + 6, my - 6, label, size=10.5, color="#64748b", anchor="start")

    def legend(self, x, y, items):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="196" height="{28 + 22*len(items)}" rx="8" '
            f'fill="white" stroke="#cbd5e1"/>'
        )
        self.text(x + 12, y + 20, "Legend", size=12, weight="bold", anchor="start")
        for i, (kind, lbl) in enumerate(items):
            fill, stroke = STYLES[kind]
            yy = y + 34 + i * 22
            self.parts.append(
                f'<rect x="{x+12}" y="{yy}" width="16" height="14" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
            self.text(x + 36, yy + 12, lbl, size=11, anchor="start", color="#334155")

    def sep(self, y, label, x0=110, x1=770):
        """A dashed boundary line with a centered label — used for SSE stream breaks."""
        self.parts.append(
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#94a3b8" '
            f'stroke-width="1.5" stroke-dasharray="4 4"/>'
        )
        cx = (x0 + x1) / 2
        wpx = 8.6 * len(label)
        self.parts.append(
            f'<rect x="{cx - wpx/2 - 8}" y="{y - 11}" width="{wpx + 16}" height="22" rx="6" '
            f'fill="white" stroke="#f59e0b"/>'
        )
        self.text(cx, y + 4, label, size=11, weight="bold", color="#b45309", italic=True)

    def render(self):
        defs = (
            '<defs><marker id="arw" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/></marker></defs>'
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}">{defs}' + "".join(self.parts) + "</svg>"
        )


# ── Diagram 1: agent system ──────────────────────────────────────────────────
def agent_diagram():
    s = SVG(1470, 880, "AI Reporting Agent — System Architecture")
    s.legend(1230, 60, [
        ("llm", "LLM call (red badge)"),
        ("app", "entry / UI"),
        ("tool", "tool / code"),
        ("service", "service"),
        ("data", "datastore"),
    ])

    s.box("user", 510, 56, 200, 44, ["User (browser)"], "app")
    s.box("ui", 50, 130, 1120, 160, [""], "app")
    s.text(610, 152, "Agent UI — ui.py  (host :8501, FastAPI + SSE)", size=13, weight="bold", color="#5b21b6")
    s.box("schema", 70, 180, 230, 90, ["Schema tab", "/discovery/* endpoints", "browse · run · describe"], "app")
    s.box("agent", 330, 180, 320, 90,
          ["ReAct agent (LangGraph)", "one model: CLAUDE_MODEL (default Haiku)",
           "/chat · /resume · /feedback"], "llm", badge="LLM")
    s.box("summ", 680, 180, 200, 90, ["History summariser", "compress old turns"], "llm", badge="LLM")
    s.box("ckpt", 910, 180, 240, 90,
          ["Checkpointer", "Postgres (CHECKPOINT_DB_URL)", "or in-memory"], "data")
    s.box("langfuse", 1230, 220, 210, 60, ["Langfuse (cloud)", "traces, optional"], "service")

    s.box("local", 50, 400, 250, 100,
          ["Local tools", "create_chart → Chart.js", "edit_cube_config (form interrupt)"], "tool")
    s.box("bq", 340, 400, 250, 100,
          ["build_query", "route to one view → structured query", "validate vs schema · repair once"],
          "llm", badge="LLM")
    s.box("cubemcp", 630, 400, 250, 100,
          ["cube-mcp (:5001, SSE)", "get_cube_metadata · reload_schema",
           "query_cube + guard", "(only fields from build_query)"], "tool")
    s.box("libmcp", 920, 400, 250, 100,
          ["library-mcp (:5002, SSE)", "config CRUD · versions · rollback",
           "save_graph · dashboards", "commit → reload → auto-rollback"], "tool")

    s.box("chart", 50, 570, 250, 88,
          ["Chart preview", "/chart · /graph · /dashboard", "replay.py (live re-query)"], "tool")
    s.box("fb", 340, 570, 250, 88,
          ["Feedback store", "ratings + corrected queries", "Postgres or SQLite"], "data")
    s.box("cube", 630, 570, 250, 88, ["Cube.js (:4000)", "semantic layer (views)", "builds model from Library"], "service")
    s.box("libapi", 920, 570, 250, 88,
          ["Library API (:3001)", "versioned cube configs", "graphs · dashboards"], "service")
    s.box("disc", 1230, 570, 210, 88,
          ["Schema discovery", "profile → grain → joins", "+ AI descriptions"], "tool", badge="LLM")

    s.box("datadb", 630, 730, 250, 72, ["Postgres data-db (:5432)", "reporting data (per-dataset schemas)"], "data")
    s.box("libdb", 920, 730, 250, 72, ["Postgres library-db (:5433)", "configs · graphs · dashboards"], "data")

    s.arrow(s.anchor("user", "bottom"), s.anchor("ui", "top"), "prompt (SSE)")
    for i, name in enumerate(("local", "bq", "cubemcp", "libmcp")):
        s.arrow((s.anchor(name, "top")[0], 290), s.anchor(name, "top"), "tool call" if i == 0 else None)
    s.arrow((1170, 250), s.anchor("langfuse", "left"), "traces", dashed=True)
    s.arrow(s.anchor("local", "bottom"), s.anchor("chart", "top"))
    s.arrow(s.anchor("fb", "top"), s.anchor("bq", "bottom"), "few-shot examples", dashed=True)
    s.arrow((560, 500), (660, 570))
    s.text(622, 528, "/meta", size=10.5, color="#64748b", anchor="start")
    s.arrow(s.anchor("cubemcp", "bottom"), s.anchor("cube", "top"))
    s.arrow(s.anchor("libmcp", "bottom"), s.anchor("libapi", "top"))
    s.arrow(s.anchor("cube", "bottom"), s.anchor("datadb", "top"), "SQL")
    s.arrow(s.anchor("libapi", "bottom"), s.anchor("libdb", "top"), "SQL")
    s.arrow(s.anchor("cube", "right"), s.anchor("libapi", "left"), "reads", dashed=True)
    s.polyline([(175, 658), (175, 700), (680, 700), (680, 658)], dashed=True)
    s.text(185, 694, "live re-query", size=10.5, color="#64748b", anchor="start")
    s.arrow(s.anchor("disc", "left"), s.anchor("libapi", "right"), "seed.py", dashed=True)
    s.polyline([(1335, 658), (1335, 840), (690, 840), (690, 802)], dashed=True)
    s.text(1325, 834, "tables (publish.py)", size=10.5, color="#64748b", anchor="end")
    return s.render()


# ── Diagram 2: eval harness ──────────────────────────────────────────────────
def eval_diagram():
    s = SVG(1300, 760, "Evals — Architecture  (build once · run often)")
    s.legend(1080, 74, [
        ("llm", "LLM call (red badge)"),
        ("tool", "code (no LLM)"),
        ("service", "service"),
        ("data", "cache / datastore"),
    ])

    s.lane(40, 74, 1010, 200, "BUILD TIME — generate the dataset  (rare: schema change / more paraphrases)")
    s.lane(40, 310, 1220, 420, "RUN TIME — score the agent  (frequent: every prompt tweak / model swap)")

    # build lane
    s.box("meta1", 70, 146, 150, 60, ["Cube /meta", "live schema"], "service")
    s.box("gen", 260, 140, 180, 72, ["generate.py", "back-translate", "title · synonym · filter/sort"], "tool")
    s.box("para", 480, 140, 180, 72, ["paraphrase.py", "reword prompt", "expected = COPIED"], "llm", badge="LLM")
    s.box("verify", 700, 140, 180, 72, ["verify.py", "independent re-derive", "drop drift"], "llm", badge="LLM")
    s.box("cache", 920, 130, 110, 92, ["cases", ".jsonl", "CACHE"], "data")

    s.arrow(s.anchor("meta1", "right"), s.anchor("gen", "left"))
    s.arrow(s.anchor("gen", "right"), s.anchor("para", "left"), "opt-in")
    s.arrow(s.anchor("para", "right"), s.anchor("verify", "left"), "opt-in")
    s.arrow(s.anchor("verify", "right"), s.anchor("cache", "left"))
    s.text(570, 252, "Bucket A: dataset-building LLM calls — paid ONCE, frozen in cache",
           size=11, color="#b91c1c")

    # run lane — main suite
    s.box("runpy", 70, 370, 160, 84, ["run.py", "load cases (no LLM)"], "tool")
    s.box("agent", 280, 370, 220, 84,
          ["Agent under test", "ReAct agent + real tools", "Bucket B — every run"], "llm", badge="LLM")
    s.box("capture", 550, 370, 190, 84, ["capture tool calls", "query_cube · create_chart", "save_graph"], "tool")
    s.box("grading", 790, 370, 210, 84,
          ["grading.py (no LLM)", "grade_query · members_exist", "chart_type · mapping"], "tool")
    s.box("report", 1050, 370, 180, 84, ["Report", "pass-rate per aspect", "× template / source"], "tool")
    s.box("meta2", 820, 500, 150, 56, ["Cube /meta", "live schema"], "service")

    s.polyline([(975, 222), (975, 292), (54, 292), (54, 412), (70, 412)])
    s.text(560, 287, "read from cache (no LLM)", size=10.5, color="#64748b")
    s.arrow(s.anchor("runpy", "right"), s.anchor("agent", "left"), "per case")
    s.arrow(s.anchor("agent", "right"), s.anchor("capture", "left"))
    s.arrow(s.anchor("capture", "right"), s.anchor("grading", "left"))
    s.arrow(s.anchor("grading", "right"), s.anchor("report", "left"))
    s.arrow(s.anchor("meta2", "top"), s.anchor("grading", "bottom"), "members_exist")
    s.text(390, 476, "Bucket B: agent LLM calls — every run (this IS the eval)",
           size=11, color="#b91c1c")

    # run lane — targeted suites
    s.text(70, 596, "Targeted suites — own case files, run deliberately (live LLM)",
           size=13, weight="bold", anchor="start", color="#475569")
    s.box("routing", 70, 615, 280, 84,
          ["routing.py", "select_view step only", "right view? refuses spans?"], "llm", badge="LLM")
    s.box("refusal", 390, 615, 280, 84,
          ["refusal.py", "build_query on unanswerable asks", "says no vs force-fits"], "llm", badge="LLM")
    s.box("adv", 710, 615, 320, 84,
          ["adversarial.py", "prompt injection · few-shot poisoning",
           "write tools stubbed (sandbox)"], "llm", badge="LLM")
    return s.render()


# ── Diagram 3: config-edit human-in-the-loop ─────────────────────────────────
def config_edit_diagram():
    s = SVG(1150, 940, "Config-Edit — Human-in-the-Loop Flow")
    s.legend(920, 66, [
        ("llm", "LLM call (red badge)"),
        ("human", "human decision (pause)"),
        ("app", "UI"),
        ("service", "service / tool"),
        ("data", "mechanism / store"),
    ])

    MX, MW = 120, 340   # main spine x, width
    R = 560             # right column x

    s.box("user", MX, 62, MW, 44, ["User: “add a measure for avg basket size”"], "app")
    s.box("agent1", MX, 128, MW, 52, ["Agent (Sonnet) — calls edit_cube_config"], "llm", badge="LLM")
    s.box("pause", MX, 200, MW, 56,
          ["interrupt() → graph PAUSES", "state saved in MemorySaver (by thread_id)"], "data")
    s.sep(288, "SSE: emit {interrupt} → /chat stream ENDS")
    s.box("form", MX, 306, MW, 62, ["① Config-edit form (pre-filled)", "human reviews / adjusts"], "human")
    s.box("testsql", R, 300, 250, 40, ["Test SQL → /test-sql"], "service")
    s.box("pgdata", R, 356, 250, 40, ["Postgres data-db (:5432)"], "data")

    s.sep(410, "▶ /resume?answer=…  (NEW connection)")
    s.box("agent2", MX, 430, MW, 56,
          ["Agent RESUMES — Command(resume)", "→ preview_cube_config_update"], "llm", badge="LLM")
    s.box("stage", R, 436, 250, 44, ["library-mcp → Library API", "(stage only — not saved)"], "service")
    s.box("diff", MX, 506, MW, 46, ["② UI diff card: current vs proposed"], "app")
    s.box("confirm", MX, 574, MW, 56,
          ["Agent asks “commit?” (text)", "③ human replies “yes”"], "human")

    s.sep(662, "▶ new /chat turn: “yes”")
    s.box("agent3", MX, 682, MW, 52, ["Agent — commit_cube_config_update"], "llm", badge="LLM")
    s.box("commit", R, 684, 250, 44, ["library-mcp → Library API"], "service")
    s.box("libdb", R, 744, 250, 40, ["Postgres library-db (:5433)"], "data")
    s.box("reload", MX, 756, MW, 52, ["Agent — reload_cube_schema"], "llm", badge="LLM")
    s.box("cube", R, 804, 250, 56, ["Cube.js restarts, rebuilds model", "from Library API /v1/CUBE_CONFIG"], "service")
    s.box("verify", MX, 830, MW, 50, ["④ get_cube_config_detail → show result"], "app")

    s.arrow(s.anchor("user", "bottom"), s.anchor("agent1", "top"))
    s.arrow(s.anchor("agent1", "bottom"), s.anchor("pause", "top"))
    s.arrow(s.anchor("pause", "bottom"), s.anchor("form", "top"))
    s.arrow(s.anchor("form", "right"), s.anchor("testsql", "left"))
    s.arrow(s.anchor("testsql", "bottom"), s.anchor("pgdata", "top"))
    s.arrow(s.anchor("pgdata", "left"), (s.anchor("form", "right")[0], 376), "sample rows", dashed=True)
    s.arrow(s.anchor("form", "bottom"), s.anchor("agent2", "top"))
    s.arrow(s.anchor("agent2", "right"), s.anchor("stage", "left"))
    s.arrow(s.anchor("agent2", "bottom"), s.anchor("diff", "top"))
    s.arrow(s.anchor("diff", "bottom"), s.anchor("confirm", "top"))
    s.arrow(s.anchor("confirm", "bottom"), s.anchor("agent3", "top"))
    s.arrow(s.anchor("agent3", "right"), s.anchor("commit", "left"))
    s.arrow(s.anchor("commit", "bottom"), s.anchor("libdb", "top"))
    s.arrow(s.anchor("agent3", "bottom"), s.anchor("reload", "top"))
    s.arrow(s.anchor("reload", "right"), s.anchor("cube", "left"))
    s.arrow(s.anchor("reload", "bottom"), s.anchor("verify", "top"))
    return s.render()


if __name__ == "__main__":
    import pathlib
    here = pathlib.Path(__file__).parent
    (here / "architecture_agent.svg").write_text(agent_diagram())
    (here / "architecture_evals.svg").write_text(eval_diagram())
    (here / "architecture_config_edit.svg").write_text(config_edit_diagram())
    print("Wrote architecture_agent.svg, architecture_evals.svg, architecture_config_edit.svg")
