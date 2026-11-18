"""
run_agent.py
Runs the triage agent on one complaint and prints what it did.

Usage:
  uv run python scripts/run_agent.py
  uv run python scripts/run_agent.py "your complaint text here"
"""

import asyncio
import sys
import time

from dotenv import load_dotenv

load_dotenv()
from triage.agent.graph import triage_complaint  # after load_dotenv so it sees .env

DEFAULT_COMPLAINT = (
    "I paid off a collection account with Midwest Recovery in full in March and I have the "
    "receipt. My credit report still shows it as unpaid with a balance, and my score dropped. "
    "I disputed it six weeks ago and nobody has fixed it or told me anything."
)


async def main() -> None:
    complaint = " ".join(sys.argv[1:]) or DEFAULT_COMPLAINT
    print("COMPLAINT:\n" + complaint + "\n")
    print("Running the agent (this takes about 30-60 seconds)...\n")

    start = time.time()
    result = await triage_complaint(complaint)

    print("=" * 70)
    print("TRACE (what the agent did, step by step):")
    for number, line in enumerate(result.get("trace", []), start=1):
        print(f"  {number:>2}. {line}")

    print("\nCATEGORY: ", result.get("category"))
    print("KEY FACTS:")
    for fact in result.get("key_facts", []):
        print("   -", fact)

    print("\nDRAFT RESPONSE:")
    print(result.get("draft_response", "(none)"))
    print("\nCITATIONS:", result.get("citations"))
    print(f"\nSTATUS: {result.get('status')}   |   DRAFT ID: {result.get('draft_id')}")
    print(f"Time taken: {time.time() - start:.0f} seconds")


if __name__ == "__main__":
    asyncio.run(main())