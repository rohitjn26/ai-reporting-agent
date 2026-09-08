"""
Reporting Agent — interactive CLI.

Usage:
  cd agent
  python main.py

Environment (set in .env or export):
  ANTHROPIC_API_KEY   required
  CUBE_MCP_URL        default: http://localhost:5001/sse
  LIBRARY_MCP_URL     default: http://localhost:5002/sse
  CHART_PORT          default: 8080
"""
import asyncio, os, sys
from pathlib import Path

# Allow imports from agent/ when running main.py directly
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent.parent / ".env")

from langchain_core.messages import HumanMessage
from graph.agent import build_agent

BANNER = """
╔══════════════════════════════════════════════════╗
║          Reporting Agent  (LangGraph + MCP)      ║
║  Stack: Cube.js → Library → MCP → Claude         ║
╚══════════════════════════════════════════════════╝
Type a request to generate a chart. Examples:
  • bar chart of total revenue by country
  • line chart of monthly order count in 2024
  • pie chart of orders by status
  • show avg order value by country as a bar chart
  • list available cubes

Type 'exit' or Ctrl-C to quit.
"""


async def run_agent():
    print(BANNER)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[error] ANTHROPIC_API_KEY is not set. Add it to agent/.env")
        sys.exit(1)

    print("Connecting to MCP servers...", end=" ", flush=True)
    try:
        agent = await build_agent()
        print("OK\n")
    except Exception as e:
        print(f"\n[error] Could not connect to MCP servers: {e}")
        print("Make sure the stack is running: make up && make seed")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        print("Agent: ", end="", flush=True)
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=user_input)]}
            )
            last = result["messages"][-1]
            content = last.content if hasattr(last, "content") else str(last)
            # content can be a list of blocks (Claude's format)
            if isinstance(content, list):
                text_parts = [
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                ]
                content = "\n".join(text_parts)
            print(content)
        except Exception as e:
            print(f"[error] {e}")
        print()


if __name__ == "__main__":
    asyncio.run(run_agent())
