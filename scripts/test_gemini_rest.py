import os

import requests
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

print("1. API key loaded.", flush=True)

url = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/models/gemini-3.7-flash:generateContent"
)

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": api_key,
}

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Explain RAG in two sentences."
                }
            ]
        }
    ]
}

print("2. Sending REST request...", flush=True)

try:
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30,
    )

    print("3. HTTP status:", response.status_code)
    print()

    if response.ok:
        data = response.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]

        print("Gemini response:")
        print(text)

    else:
        print("Gemini error:")
        print(response.text)

except requests.exceptions.Timeout:
    print("ERROR: Request timed out after 30 seconds.")

except requests.exceptions.ConnectionError as error:
    print("ERROR: Could not connect to Gemini.")
    print(error)