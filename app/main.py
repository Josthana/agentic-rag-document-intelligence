from fastapi import FastAPI

app = FastAPI(
    title="Agentic RAG Document Intelligence",
    description="API for document ingestion, retrieval, and agentic RAG",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Agentic RAG Document Intelligence API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }