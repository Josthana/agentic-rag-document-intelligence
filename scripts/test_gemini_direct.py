import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from .env")


print("1. API key loaded successfully.", flush=True)

client = genai.Client(
    api_key=api_key
)

print("2. Gemini client created.", flush=True)

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents="Explain RAG in two sentences.",
    config=types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    ),
)

print("3. Response received.", flush=True)
print()
print(response.text)