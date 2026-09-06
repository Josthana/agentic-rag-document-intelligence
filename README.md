# Agentic RAG Document Intelligence System

An end-to-end **Agentic Retrieval-Augmented Generation (RAG)** system for answering questions from PDF documents using hybrid retrieval, local embeddings, a local LLM, LangGraph, FastAPI, Docker, and Kubernetes.

The project combines:

- PDF ingestion and chunking
- Semantic search with ChromaDB
- Keyword search with BM25
- Hybrid retrieval
- Local LLM generation with Ollama
- Agentic query rewriting with LangGraph
- FastAPI REST API
- Automated evaluation
- Pytest unit tests
- Docker containerization
- Kubernetes deployment

---

## Project Overview

Traditional RAG systems usually follow a fixed flow:

```text
Question
   ↓
Retrieve documents
   ↓
Generate answer
```

This project extends that architecture with an **agentic retrieval loop**.

If the first retrieval attempt produces weak context, the system can rewrite the search query and retrieve again before deciding whether enough evidence exists to answer.

```text
User Question
     ↓
Hybrid Retrieval
     ↓
Evaluate Retrieval Quality
     ↓
 ┌───────────────┐
 │ Enough context?│
 └───────┬───────┘
         │
     ┌───┴───┐
     │       │
    Yes      No
     │       │
     ↓       ↓
 Generate   Rewrite Query
 Answer       │
     │        ↓
     │    Retrieve Again
     │        │
     └──────→ Decision
              │
              ↓
             END
```

This makes the application more robust than a simple one-pass RAG pipeline.

---

# Architecture

```text
                        ┌──────────────────┐
                        │   PDF Documents  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ PDF Ingestion    │
                        │ PyPDFLoader      │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Text Chunking    │
                        │ Recursive Split  │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
           ┌─────────────────┐       ┌─────────────────┐
           │ ChromaDB        │       │ BM25            │
           │ Vector Search   │       │ Keyword Search  │
           └────────┬────────┘       └────────┬────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Hybrid Retriever │
                        │ Weighted Fusion  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ LangGraph Agent  │
                        │                  │
                        │ Retrieve         │
                        │ Evaluate         │
                        │ Rewrite / Retry  │
                        │ Generate         │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Ollama           │
                        │ llama3.2:3b      │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ Grounded Answer  │
                        │ + Sources        │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │ FastAPI REST API │
                        └──────────────────┘
```

---

# Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| PDF loading | LangChain `PyPDFLoader` |
| Chunking | RecursiveCharacterTextSplitter |
| Local embeddings | SentenceTransformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector database | ChromaDB |
| Keyword retrieval | BM25 |
| Hybrid retrieval | Weighted vector + BM25 fusion |
| Agent orchestration | LangGraph |
| Local LLM | Ollama |
| LLM model | `llama3.2:3b` |
| API | FastAPI |
| API server | Uvicorn |
| Testing | Pytest |
| Evaluation | Custom RAG evaluation pipeline |
| Containerization | Docker |
| Orchestration | Kubernetes |

---

# Repository Structure

```text
agentic-rag-document-intelligence/
│
├── app/
│   ├── __init__.py
│   ├── api.py
│   ├── bm25_store.py
│   ├── config.py
│   ├── graph.py
│   ├── ingestion.py
│   ├── models.py
│   ├── rag.py
│   ├── retriever.py
│   └── vector_store.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── questions.json
│   └── results/
│
├── scripts/
│   ├── check_setup.py
│   ├── ingest_documents.py
│   ├── test_bm25_search.py
│   ├── test_graph.py
│   ├── test_hybrid_search.py
│   ├── test_ingestion.py
│   ├── test_model.py
│   ├── test_rag.py
│   └── test_vector_search.py
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_ingestion.py
│   ├── test_models.py
│   └── test_retriever.py
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# Core Components

## 1. Document Ingestion

`app/ingestion.py`

The ingestion layer:

1. Finds PDFs in `data/raw`
2. Loads PDFs page by page
3. Removes pages without useful text
4. Adds document metadata
5. Splits text into overlapping chunks
6. Assigns unique chunk IDs

Example metadata:

```json
{
  "source": "employee_handbook.pdf",
  "page_number": 2,
  "chunk_id": "chunk-00003"
}
```

---

## 2. Semantic Vector Retrieval

`app/vector_store.py`

Semantic retrieval uses:

```text
SentenceTransformer
        ↓
all-MiniLM-L6-v2
        ↓
384-dimensional embeddings
        ↓
ChromaDB
```

The system embeds both document chunks and user questions and retrieves semantically similar chunks.

---

## 3. BM25 Keyword Retrieval

`app/bm25_store.py`

BM25 complements vector search by finding exact or near-exact keyword matches.

This is helpful when queries contain:

- Policy names
- Technical terms
- Employee terminology
- Exact business phrases
- Specific document vocabulary

The BM25 index is persisted locally.

---

## 4. Hybrid Retrieval

`app/retriever.py`

The final retriever combines semantic and keyword search:

```text
User Query
   │
   ├───────────────┐
   ▼               ▼
Vector Search     BM25 Search
   │               │
   └───────┬───────┘
           ▼
    Score Normalization
           ▼
      Weighted Fusion
           ▼
     Ranked Results
```

Default weights:

```text
Vector weight = 0.6
BM25 weight   = 0.4
```

The combined ranking improves retrieval robustness compared with relying on only one retrieval method.

---

## 5. RAG Generation

`app/rag.py`

The RAG service:

1. Retrieves relevant chunks
2. Builds a document context
3. Sends the question and context to Ollama
4. Instructs the LLM to answer only from supplied evidence
5. Returns the generated answer with retrieved sources

The generation model is:

```text
llama3.2:3b
```

served locally through Ollama.

This allows the RAG generation layer to operate without sending document context to an external LLM service.

---

## 6. LangGraph Agent

`app/graph.py`

The LangGraph workflow adds decision-making to the RAG pipeline.

```text
retrieve
   ↓
evaluate context
   ↓
 ┌──────────────┐
 │ strong?      │
 └──────┬───────┘
        │
   ┌────┴────┐
   │         │
  yes        no
   │         │
   ▼         ▼
generate   rewrite
             │
             ▼
          retrieve
             │
             ▼
          evaluate
```

The agent can:

- Retrieve evidence
- Measure retrieval quality
- Rewrite weak queries
- Retry retrieval
- Stop after a maximum number of attempts
- Generate a grounded answer
- Refuse to answer when sufficient context cannot be found

---

# Installation

## Prerequisites

Install:

- Python 3.13+
- Ollama
- Docker Desktop
- Git

For Kubernetes deployment, enable Kubernetes inside Docker Desktop.

---

## Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd agentic-rag-document-intelligence
```

---

## Create a virtual environment

```bash
python3 -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

---

## Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# Environment Variables

Create:

```text
.env
```

Example configuration:

```env
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384

CHROMA_DIR=state/chroma
CHROMA_COLLECTION=documents
VECTOR_TOP_K=5

BM25_TOP_K=5
BM25_INDEX_FILE=state/bm25_index.json

FINAL_TOP_K=5
VECTOR_WEIGHT=0.6
BM25_WEIGHT=0.4

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT=120
```

Never commit `.env` files containing credentials or secrets.

---

# Ollama Setup

Install Ollama and download the model:

```bash
ollama pull llama3.2:3b
```

Test it:

```bash
ollama run llama3.2:3b "Reply only with: Ollama works"
```

Expected:

```text
Ollama works
```

---

# Add Documents

Place PDF documents inside:

```text
data/raw/
```

Example:

```text
data/raw/
├── employee_handbook.pdf
├── expense_policy.pdf
├── leave_policy.pdf
├── remote_work_policy.pdf
└── travel_policy.pdf
```

---

# Run the Application Locally

Start FastAPI:

```bash
python -m uvicorn app.api:app --reload
```

The API is available at:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# API Endpoints

## Root

```http
GET /
```

Example:

```bash
curl http://127.0.0.1:8000/
```

---

## Health Check

```http
GET /health
```

Example:

```bash
curl http://127.0.0.1:8000/health
```

Example response:

```json
{
  "status": "healthy",
  "vector_chunks": 36,
  "bm25_chunks": 36,
  "model": "llama3.2:3b"
}
```

---

## Ask a Question

```http
POST /query
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the requirements for remote work eligibility?"
  }'
```

Example response structure:

```json
{
  "query": "What are the requirements for remote work eligibility?",
  "answer": "Remote work eligibility depends on role requirements, performance, security requirements and manager approval...",
  "used_query": "What are the requirements for remote work eligibility?",
  "sources": [
    {
      "chunk_id": "chunk-00003",
      "source": "employee_handbook.pdf",
      "page": 2,
      "score": 0.72,
      "retriever": "hybrid"
    }
  ]
}
```

---

# Automated Tests

The project contains automated unit tests for:

- Data models
- Document cleaning
- Document chunking
- Hybrid retrieval
- API behavior

Run:

```bash
pytest -v
```

Current result:

```text
9 passed
```

The unit tests use lightweight test doubles where appropriate so that the entire Ollama and embedding stack does not need to run for every test.

---

# RAG Evaluation

The project includes a custom evaluation framework in:

```text
evaluation/evaluate.py
```

Run:

```bash
python -m evaluation.evaluate
```

The evaluator measures:

- Source hit rate
- Expected keyword recall
- Citation presence
- Retrieval score
- End-to-end latency
- Overall evaluation score

## Evaluation Results

Evaluation set:

```text
5 questions
```

Results:

| Metric | Result |
|---|---:|
| Questions evaluated | 5 |
| Average overall score | **0.90** |
| Source hit rate | **1.00** |
| Average keyword recall | **1.00** |
| Citation rate | **0.60** |
| Average retrieval score | **0.455** |
| Average latency | **10.318 sec** |

### Interpretation

The system retrieved an expected source document for all evaluation questions:

```text
Source Hit Rate = 100%
```

Expected keyword coverage was also:

```text
Keyword Recall = 100%
```

The primary improvement area identified by evaluation is citation formatting:

```text
Citation Rate = 60%
```

This can be improved with stronger structured-output or citation-validation logic.

The retrieval score is a custom hybrid fusion score and should not be interpreted as a percentage accuracy metric.

---

# Docker Deployment

Build the image:

```bash
docker build \
  -t agentic-rag-document-intelligence:v2 .
```

Ollama runs on the host machine.

Containers access it through:

```text
http://host.docker.internal:11434
```

Run:

```bash
docker run --rm \
  --name agentic-rag-api \
  -p 8000:8000 \
  --env-file .env \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  agentic-rag-document-intelligence:v2
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the requirements for remote work eligibility?"
  }'
```

---

# Kubernetes Deployment

This project includes:

```text
k8s/
├── deployment.yaml
└── service.yaml
```

The application was tested using a local Docker Desktop Kubernetes cluster.

Verify the cluster:

```bash
kubectl get nodes
```

Apply resources:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Check deployment:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

Expected pod state:

```text
READY   STATUS
1/1     Running
```

Check logs:

```bash
kubectl logs deployment/agentic-rag-api
```

Port-forward the API:

```bash
kubectl port-forward \
  service/agentic-rag-api-service \
  8002:8000
```

Test:

```bash
curl http://127.0.0.1:8002/health
```

Then:

```bash
curl -X POST http://127.0.0.1:8002/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the requirements for remote work eligibility?"
  }'
```

---

# Development Milestones

The project was built incrementally:

```text
Phase 1  — PDF ingestion                ✅
Phase 2  — Document chunking            ✅
Phase 3  — Chroma semantic retrieval    ✅
Phase 4  — BM25 keyword retrieval       ✅
Phase 5  — Hybrid retrieval             ✅
Phase 6  — RAG + Ollama                 ✅
Phase 7  — LangGraph agent              ✅
Phase 8  — FastAPI                      ✅
Phase 9  — Evaluation                   ✅
Phase 10 — Automated tests              ✅
Phase 11 — Docker + Kubernetes          ✅
Phase 12 — Documentation / final demo   ✅
```

---

# Key Engineering Decisions

## Why Hybrid Retrieval?

Semantic search is strong at meaning-based retrieval but may miss exact terminology.

BM25 is strong at keyword matching but does not understand semantic similarity.

Combining them gives:

```text
Semantic relevance
        +
Exact keyword relevance
        ↓
Better retrieval robustness
```

---

## Why Local Embeddings?

`all-MiniLM-L6-v2` provides:

- Fast inference
- Small model size
- 384-dimensional embeddings
- No embedding API cost
- Local document processing

---

## Why Ollama?

Ollama allows the generation model to run locally.

Benefits include:

- Reduced dependency on external APIs
- Better control over document data
- No per-request LLM API cost during development
- Easier offline experimentation

---

## Why LangGraph?

LangGraph allows retrieval to behave as a workflow instead of a fixed function call.

The agent can decide:

```text
Is retrieval good enough?
        │
       Yes → Generate
        │
       No
        ↓
Rewrite query
        ↓
Retrieve again
```

This creates a true agentic retrieval loop.

---

# Current Limitations

The current implementation has several areas for future improvement:

1. **Citation formatting**

   Evaluation showed a 60% citation rate.

2. **Latency**

   Local generation averaged approximately 10 seconds during evaluation.

3. **Persistence in Kubernetes**

   The current local Kubernetes deployment uses temporary local runtime storage. A production deployment should use persistent volumes.

4. **Authentication**

   The API currently does not implement user authentication or authorization.

5. **Observability**

   Production deployments should add structured logging, tracing and metrics.

6. **Evaluation dataset size**

   The current evaluation set contains five questions and should be expanded for production-quality benchmarking.

---

# Future Improvements

Potential extensions include:

- Cross-encoder reranking
- Reciprocal Rank Fusion
- Metadata filtering
- Multi-document query decomposition
- LLM-based retrieval grading
- Structured citation validation
- Streaming responses
- Conversation memory
- Redis caching
- Persistent Kubernetes volumes
- Prometheus metrics
- Grafana dashboards
- OpenTelemetry tracing
- CI/CD with GitHub Actions
- Authentication and rate limiting
- Cloud deployment
- Larger RAG evaluation datasets

---

# Final Demo Flow

For a portfolio or interview demonstration:

### 1. Show system architecture

Explain:

```text
PDF
↓
Chunking
↓
Chroma + BM25
↓
Hybrid Retriever
↓
LangGraph
↓
Ollama
↓
FastAPI
```

### 2. Show the healthy Kubernetes pod

```bash
kubectl get pods
```

Expected:

```text
1/1 Running
```

### 3. Start port forwarding

```bash
kubectl port-forward \
  service/agentic-rag-api-service \
  8002:8000
```

### 4. Show health

```bash
curl http://127.0.0.1:8002/health
```

### 5. Ask a document question

```bash
curl -X POST http://127.0.0.1:8002/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the requirements for remote work eligibility?"
  }'
```

### 6. Explain the response

Highlight:

- Grounded answer
- Source document
- Page number
- Hybrid retrieval score
- LangGraph decision flow

### 7. Show automated tests

```bash
pytest -v
```

Result:

```text
9 passed
```

### 8. Show evaluation

```bash
python -m evaluation.evaluate
```

Highlight:

```text
Overall evaluation score: 90%
Source hit rate: 100%
Keyword recall: 100%
```

---

# What This Project Demonstrates

This project demonstrates practical experience with:

- Retrieval-Augmented Generation
- Vector databases
- Semantic embeddings
- Information retrieval
- BM25
- Hybrid search
- Local LLM inference
- Agentic AI workflows
- LangGraph
- FastAPI
- API design
- RAG evaluation
- Automated testing
- Docker
- Kubernetes
- Production-oriented AI system architecture

---

## Project Status

**Complete — end-to-end Agentic RAG system successfully tested locally, in Docker, and on Kubernetes.**