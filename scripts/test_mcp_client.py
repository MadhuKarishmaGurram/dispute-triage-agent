"""
test_mcp_client.py
Connects to one of our MCP servers the same way an AI agent would:
  1. starts the server
  2. asks what tools it offers
  3. calls some test tools and prints the results

Usage:
  uv run python scripts/test_mcp_client.py complaints
  uv run python scripts/test_mcp_client.py regulations
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# For each server: which tool calls to test, and with what inputs.
TEST_CALLS = {
    "complaints": [
        ("search_similar_complaints",
         {"query": "A collection account I paid is still reported as unpaid", "limit": 2}),
        ("get_category_stats", {}),
    ],
    "regulations": [
        ("list_regulation_sections", {}),
        ("search_regulations",
         {"query": "How many days does a credit bureau have to investigate a dispute?",
          "limit": 2}),
    ],
}


async def main(server_name: str) -> None:
    # How to start the chosen server.
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", f"triage.mcp_servers.{server_name}_server"],
        env=dict(os.environ),
    )

    # STEP 1: Start the server and open a session with it.
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"Connected to the {server_name} server.\n")

            # STEP 2: Ask the server what tools it has (the "menu").
            tools = await session.list_tools()
            print("Tools the server offers:")
            for tool in tools.tools:
                first_line = (tool.description or "").strip().splitlines()[0]
                print(f"  - {tool.name}: {first_line}")

            # STEP 3: Call each test tool and print what comes back.
            for tool_name, arguments in TEST_CALLS[server_name]:
                print(f"\n--- Calling {tool_name} ---")
                result = await session.call_tool(tool_name, arguments)
                for item in result.content:
                    print(item.text)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "complaints"
    if name not in TEST_CALLS:
        print(f"Unknown server '{name}'. Choose one of: {', '.join(TEST_CALLS)}")
        sys.exit(1)
    asyncio.run(main(name))