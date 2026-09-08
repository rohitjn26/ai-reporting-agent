"""
LangGraph ReAct agent wired to MCP tool servers + local chart/config tools.
"""
import json, os
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from chart.renderer import render_chart
from chart.server import serve_chart
from graph.config_editor import edit_cube_config

CUBE_MCP_URL    = os.environ.get("CUBE_MCP_URL",    "http://localhost:5001/sse")
LIBRARY_MCP_URL = os.environ.get("LIBRARY_MCP_URL", "http://localhost:5002/sse")
MODEL           = os.environ.get("CLAUDE_MODEL",    "claude-sonnet-4-6")

SYSTEM_PROMPT = """\
You are a data reporting agent. When the user asks for a chart or data insight:

1. Call get_cube_metadata to discover available cubes, measures, and dimensions.
2. Choose the right cube and identify the correct measures and dimensions from the metadata.
3. Call query_cube. Member names must be fully qualified: "cube_name.member_name".
4. Call create_chart with the results to render the visualization.
5. Return the chart URL with a brief description.

Always call create_chart at the end — the user expects a visual result.

IMPORTANT query rules:
- If query_cube fails, read the error carefully. Do NOT retry the same query.
  Fix the member names or filters based on the error, then try once more.
  If it still fails, tell the user what went wrong instead of looping.
- NEVER call reload_cube_schema when answering a data/chart question.
  reload_cube_schema is only for after committing a config change.

When the user asks for a measure or dimension that does NOT appear in get_cube_metadata:
1. Tell the user it does not exist yet, and ask: "Should I add it to the [cube_name] cube?"
2. If the user says yes, call edit_cube_config with your best-guess values:
   - suggested_field_type: "measure" or "dimension" based on what they asked for
   - suggested_key: a snake_case name derived from the user's request
   - suggested_sql: a SQL expression inferred from the cube's base table columns
     (e.g. for "bi-month": CASE WHEN EXTRACT(MONTH FROM created_at) IN (1,2) THEN 'Jan-Feb' ...)
   - suggested_type: the most likely type (e.g. "string" for grouping dims, "sum" for revenue measures)
   - suggested_title: a clean human-readable label
   The form will open pre-filled — the user can review and adjust before applying.

When the user asks to MODIFY an existing cube config, use this exact flow — never skip steps:
1. Call edit_cube_config(cube_name, intent). This will ask the user clarifying questions
   (measure vs dimension, add vs replace, name, SQL, type, title) via the chat UI.
   Wait for it to return the full change spec — do NOT call preview before it finishes.
2. Call preview_cube_config_update using the config_id, updated_measures, and
   updated_dimensions returned by edit_cube_config. This stages the change without saving.
3. Summarise what changed and ask: "Should I commit this to the database?"
4. Only after the user confirms, call commit_cube_config_update.
5. Call reload_cube_schema so the change is live in Cube.js immediately.
6. Call get_cube_config_detail to show the saved result for verification.

When the user asks to CREATE a new cube config, use create_cube_config directly.
"""

# Single shared checkpointer — persists interrupt state across SSE reconnections.
_checkpointer = MemorySaver()

# Summarise when message count exceeds this; keep the last KEEP_RECENT messages as-is.
SUMMARISE_AFTER = 10
KEEP_RECENT     = 4
_SUMMARY_MODEL  = os.environ.get("SUMMARY_MODEL", "claude-haiku-4-5-20251001")


def _extract_text(content) -> str:
    """Pull plain text from a message content field (str or list-of-blocks)."""
    if isinstance(content, str):
        return content[:400]
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )[:400]
    return str(content)[:400]


async def maybe_summarise(agent, thread_id: str) -> bool:
    """
    If the stored message history for this thread is longer than SUMMARISE_AFTER,
    ask a fast model to compress the old messages into one SystemMessage summary,
    then remove the originals from the graph state.

    Returns True if summarisation happened, False otherwise.
    Never runs when an interrupt is pending.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state  = agent.get_state(config)

    # Don't touch state if a task is still pending (e.g. mid-interrupt).
    if state.next:
        return False

    messages = state.values.get("messages", [])
    if len(messages) <= SUMMARISE_AFTER:
        return False

    to_summarise = list(messages[:-KEEP_RECENT])
    keep         = list(messages[-KEEP_RECENT:])

    # Ensure 'keep' starts at a clean HumanMessage boundary so we never
    # leave an orphaned ToolMessage whose tool_use was summarised away.
    while keep and not isinstance(keep[0], HumanMessage):
        to_summarise.append(keep.pop(0))

    if not keep or not to_summarise:
        return False

    conversation = "\n".join(
        f"{m.type.upper()}: {_extract_text(m.content)}"
        for m in to_summarise
    )

    prompt = (
        "Summarise this agent conversation in under 120 words. "
        "Capture: what the user asked for, which cubes/measures/dimensions were used, "
        "any config changes made, and the current state of things.\n\n"
        + conversation
    )

    summariser = ChatAnthropic(model=_SUMMARY_MODEL)
    response   = await summariser.ainvoke([HumanMessage(content=prompt)])
    summary    = response.content if isinstance(response.content, str) else _extract_text(response.content)

    remove_ops   = [RemoveMessage(id=m.id) for m in to_summarise]
    summary_msg  = SystemMessage(content=f"[Conversation summary]\n{summary}")
    agent.update_state(config, {"messages": remove_ops + [summary_msg]})
    return True


@tool
def create_chart(
    chart_type: str,
    labels: list[str],
    datasets: list[dict],
    title: str = "",
) -> str:
    """
    Render a Chart.js chart and serve it at localhost.

    Args:
        chart_type: "bar" | "line" | "pie" | "doughnut"
        labels:     list of label strings (x-axis for bar/line, segments for pie)
        datasets:   list of dicts, each with "label" (str) and "data" (list of numbers)
        title:      optional chart title
    """
    html = render_chart(chart_type, labels, datasets, title)
    url = serve_chart(html, open_browser=True)
    return f"Chart ready at: {url}"


def _build_state_modifier(state) -> list:
    """
    Merge any SystemMessage summaries stored in the message history into the
    main SYSTEM_PROMPT so the model always sees exactly ONE system message.

    Without this, `create_react_agent` prepends SYSTEM_PROMPT as a SystemMessage
    and our injected summary SystemMessage results in two system messages, which
    the Anthropic API rejects with "multiple non-consecutive system messages".
    """
    messages = state["messages"] if isinstance(state, dict) else state.messages
    summaries  = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]

    system_content = SYSTEM_PROMPT
    if summaries:
        summary_text = "\n\n".join(m.content for m in summaries)
        system_content = SYSTEM_PROMPT + "\n\n" + summary_text

    return [SystemMessage(content=system_content)] + non_system


async def build_agent():
    """
    Connect to both MCP servers, collect their tools, add local tools,
    and return a compiled LangGraph ReAct agent with interrupt support.
    """
    mcp_client = MultiServerMCPClient({
        "cube":    {"url": CUBE_MCP_URL,    "transport": "sse"},
        "library": {"url": LIBRARY_MCP_URL, "transport": "sse"},
    })

    mcp_tools = await mcp_client.get_tools()
    all_tools  = mcp_tools + [create_chart, edit_cube_config]

    model = ChatAnthropic(model=MODEL)
    agent = create_react_agent(
        model,
        all_tools,
        state_modifier=_build_state_modifier,
        checkpointer=_checkpointer,
    )
    return agent
