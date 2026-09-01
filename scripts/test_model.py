print("1. Starting test")

from app.models import llm

print("2. Model imported successfully")

response = llm.invoke("Reply with exactly: Hello")

print("3. Model responded")
print(response)