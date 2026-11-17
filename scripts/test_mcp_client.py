"""
test_mcp_client.py
Connects to our complaints MCP server the same way an AI agent would:
  1. starts the server
  2. asks what tools it offers
  3. calls each tool and prints the result
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# How to start the server: run the complaints_server module with the
# same Python this script is using, and pass along our environment
# (which includes the Google Cloud login location).
SERVER = StdioServerParameters(
    command=sys.executable,
    args=["-m", "triage.mcp_servers.complaints_server"],
    env=dict(os.environ),
)


async def main() -> None:
    # STEP 1: Start the server and open a session with it.
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected to the complaints server.\n")

            # STEP 2: Ask the server what tools it has (the "menu").
            tools = await session.list_tools()
            print("Tools the server offers:")
            for tool in tools.tools:
                first_line = (tool.description or "").strip().splitlines()[0]
                print(f"  - {tool.name}: {first_line}")

            # STEP 3: Call the search tool.
            print("\n--- Calling search_similar_complaints ---")
            result = await session.call_tool(
                "search_similar_complaints",
                {"query": "A collection account I paid is still reported as unpaid", "limit": 2},
            )
            for item in result.content:
                print(item.text)

            # STEP 4: Call the stats tool.
            print("\n--- Calling get_category_stats ---")
            result = await session.call_tool("get_category_stats", {})
            for item in result.content:
                print(item.text)


if __name__ == "__main__":
    asyncio.run(main())