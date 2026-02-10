# AI-Powered Document Intelligence Platform

A production-grade document intelligence platform featuring RAG-based Q&A, document classification, field extraction, and hybrid search. Built with FastAPI, PostgreSQL + pgvector, Celery, and Gemini 2.5 Flash.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Frontend   │───▶│  FastAPI API  │───▶│ PostgreSQL  │
│  (Next.js)   │    │   Backend    │    │ + pgvector  │
└─────────────┘    └──────┬───────┘    └─────────────┘
                          │
                   ┌──────┴───────┐
                   │              │
              ┌────▼────┐   ┌────▼────┐
              │  Celery  │   │  MinIO  │
              │ Workers  │   │  (S3)   │
              └────┬─────┘   └─────────┘
                   │
              ┌────▼─────┐
              │  Redis    │
              │ (Broker)  │
              └──────────┘
```

## Features

- **Document Ingestion**: Upload PDF, DOCX, TXT, CSV/Excel files with async processing
- **Text Extraction**: Native PDF extraction (pdfplumber) with OCR fallback (Tesseract)
- **Smart Chunking**: Token-based chunking with configurable overlap and metadata enrichment
- **Document Classification**: Zero-shot classification via Gemini 2.5 Flash with confidence scoring
- **Field Extraction**: Structured field extraction per document type (Invoice, Contract, Resume, etc.)
- **Hybrid Search**: Semantic search (pgvector) + keyword search (tsvector) with Reciprocal Rank Fusion
- **Re-ranking**: Cross-encoder re-ranking for improved precision
- **RAG Q&A**: Conversational Q&A with source citations, streaming SSE, and hallucination guardrails
- **Human-in-the-Loop**: Manual correction endpoints for classification and extraction
- **QLoRA Fine-Tuning**: Fine-tune Mistral 7B with 4-bit quantization on synthetic document data
- **Model Routing & A/B Testing**: Route between API and local models with cost tracking
- **Agentic Workflows**: ReAct-style agents for contract comparison, invoice anomaly detection, summarization, and cross-document research

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Celery, Redis, Pydantic v2 |
| Database | PostgreSQL 16 + pgvector + tsvector |
| Storage | MinIO (S3-compatible) |
| LLM | Google Gemini 2.5 Flash |
| Fine-Tuning | QLoRA (PEFT + bitsandbytes) on Mistral 7B |
| Model Serving | Ollama (local) + Gemini API (cloud) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L6-v2 |
| Agents | ReAct-style with LangGraph |
| Extraction | pdfplumber, Tesseract OCR, python-docx |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A Gemini API key (get one at https://aistudio.google.com)

### 1. Clone and configure

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 2. Start all services

```bash
docker compose up --build
```

This starts 5 services:
- **PostgreSQL** (pgvector) on port 5432
- **Redis** on port 6379
- **MinIO** on port 9000 (console at 9001)
- **FastAPI backend** on port 8000
- **Celery worker** for async processing

### 3. Run database migrations

```bash
docker compose exec backend alembic upgrade head
```

### 4. Verify

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/api/v1/health
- MinIO console: http://localhost:9001 (minioadmin / minioadmin123)

## API Endpoints

### Documents
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload a document |
| POST | `/api/v1/documents/upload/batch` | Upload multiple documents |
| GET | `/api/v1/documents/` | List documents |
| GET | `/api/v1/documents/{id}` | Get document details |
| GET | `/api/v1/documents/{id}/status` | Check processing status |
| GET | `/api/v1/documents/{id}/chunks` | Get document chunks |
| DELETE | `/api/v1/documents/{id}` | Delete a document |

### Classification & Extraction
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/documents/{id}/classification` | Get classification result |
| POST | `/api/v1/documents/{id}/classify` | Trigger classification |
| GET | `/api/v1/documents/{id}/extracted-fields` | Get extracted fields |
| PATCH | `/api/v1/documents/{id}/fields` | Correct extracted fields |

### Search & Chat
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/search` | Hybrid search across documents |
| POST | `/api/v1/chat` | Ask a question (non-streaming) |
| POST | `/api/v1/chat/stream` | Ask a question (SSE streaming) |
| POST | `/api/v1/conversations` | Create a conversation |
| GET | `/api/v1/conversations/{id}/messages` | Get conversation history |

### Agents
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/agents/workflows` | List available agent workflows |
| POST | `/api/v1/agents/run` | Run an agent workflow |
| POST | `/api/v1/agents/run/stream` | Run workflow with SSE streaming |
| GET | `/api/v1/agents/runs/{run_id}` | Get a previous run result |

## Usage Example

### Upload a document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@invoice.pdf"
```

### Check processing status

```bash
curl http://localhost:8000/api/v1/documents/{document_id}/status
```

### Search across documents

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the total invoice amount?", "top_k": 5}'
```

### Ask a question (RAG)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the key terms of the contract"}'
```

### Run an agent workflow

```bash
# List available workflows
curl http://localhost:8000/api/v1/agents/workflows

# Compare two contracts
curl -X POST http://localhost:8000/api/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"workflow": "contract_comparison", "params": {"doc_id_1": "...", "doc_id_2": "..."}}'

# Summarize a document
curl -X POST http://localhost:8000/api/v1/agents/run \
  -H "Content-Type: application/json" \
  -d '{"workflow": "summarization", "params": {"document_id": "..."}}'

# Research across all documents (streaming)
curl -X POST http://localhost:8000/api/v1/agents/run/stream \
  -H "Content-Type: application/json" \
  -d '{"workflow": "research_assistant", "params": {"research_question": "What are the payment terms across all contracts?"}}'
```

## Processing Pipeline

```
Upload → Store in MinIO → Extract Text → Chunk → Classify → Extract Fields → Generate Embeddings
```

Each step runs as a Celery task with automatic retry (3 attempts, exponential backoff). Document status transitions: `pending → processing → completed | failed`.

## Project Structure

```
doc-intelli/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   ├── models/          # SQLAlchemy models
│   │   ├── schemas/         # Pydantic request/response models
│   │   ├── services/        # Business logic
│   │   │   ├── extraction/  # PDF, DOCX, TXT, CSV extractors
│   │   │   ├── chunking/    # Text chunking strategies
│   │   │   ├── model_router.py  # A/B testing + cost tracking
│   │   │   ├── rag_service.py
│   │   │   └── search_service.py
│   │   ├── agents/          # Agentic workflows (ReAct)
│   │   │   ├── base.py      # ReAct agent + AgentState
│   │   │   ├── tools.py     # Tool registry
│   │   │   └── workflows.py # Specialized agents
│   │   ├── tasks/           # Celery async tasks
│   │   └── core/            # Database, Redis, logging
│   ├── alembic/             # Database migrations
│   └── tests/               # Test suite
├── ml/
│   ├── classification/      # DeBERTa classifier training
│   ├── fine_tuning/         # QLoRA fine-tuning pipeline
│   │   ├── generate_dataset.py  # Synthetic data generation
│   │   ├── train_qlora.py       # QLoRA training
│   │   ├── merge_adapter.py     # LoRA adapter merging
│   │   └── evaluate_finetuned.py
│   └── evaluation/          # RAG evaluation
├── scripts/                 # Utility scripts
├── docker-compose.yml
└── .env.example
```

## ML Training (Optional)

### DeBERTa Document Classifier

```bash
cd ml/classification
python data/prepare_dataset.py --num_samples 100
python train_classifier.py --data_dir ./data/dataset
python evaluate_classifier.py --model_dir ./output/model
```

### QLoRA Fine-Tuning (Mistral 7B)

```bash
cd ml/fine_tuning

# Generate synthetic instruction-tuning data
python generate_dataset.py --num_samples 500

# Train with QLoRA (4-bit quantization)
python train_qlora.py --config config.yaml

# Merge LoRA adapter into base model
python merge_adapter.py --adapter_dir ./output/adapter --output_dir ./output/merged

# Evaluate on held-out test set
python evaluate_finetuned.py --model_dir ./output/merged --test_file ./data/test.jsonl
```

## Development

### Running locally (without Docker)

```bash
cd backend
pip install -e ".[dev]"
uvicorn app.main:app --reload --port 8000
```

Start infrastructure separately:
```bash
docker compose up postgres redis minio
```

### Running tests

```bash
cd backend
pytest tests/ -v
```

## Author

**Ali Haider** - [alihaider.live](https://alihaider.live)
#   D o c - i n t e l l i  
 