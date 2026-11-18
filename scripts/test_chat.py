"""
test_chat.py
Asks Gemini one question through LangChain, to check that the chat model,
the Google login, and the model name in .env all work together.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()
from triage.gcp import make_chat_model, message_text  # after load_dotenv

model_name = os.getenv("GEMINI_MODEL")
print(f"Asking {model_name} a question...\n")

model = make_chat_model()
start = time.time()
reply = model.invoke("In one sentence, what does the U.S. Fair Credit Reporting Act regulate?")

print("Gemini says:")
print(message_text(reply))
print(f"\n(answered in {time.time() - start:.1f} seconds)")