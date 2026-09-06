# Enterprise AI Knowledge & Decision Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked with mypy](https://img.shields.io/badge/type_checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

> A production-oriented enterprise AI platform engineered to intelligently unify unstructured document knowledge retrieval, structured SQL business analytics, and agentic multi-tool reasoning behind a secure, observable API.

---

## 1. System Vision & Architecture

Standard enterprise RAG applications often fail because they treat every question as a vector similarity search problem. In real enterprise environments:
- **Policy & Procedure Questions** ("What is our refund policy?") require **Hybrid Retrieval** (dense vectors + sparse BM25) and **Reranking** with exact citation attribution.
- **Quantitative Business Questions** ("Which product had the highest complaints last month?") cannot be answered by cosine similarity on PDF chunks; they require **Safe, Deterministic SQL Generation** over structured data warehouses.
- **Complex Multi-Step Questions** ("Which product had the most complaints and what does our documentation say about its known defects?") require **Agentic Multi-Tool Reasoning** with state graphs.
- **Direct Queries** ("Summarize this text") require **Zero-Retrieval LLM Direct Generation** to save latency and vector database compute.

### Overall Target Architecture

```mermaid
flowchart TD
    User([Enterprise User / Client]) -->|HTTP REST / Streaming| API[FastAPI Gateway]
    
    subgraph Security & Governance
        API --> Guardrails[Guardrails & Injection Filter]
        Guardrails --> Router{Query Router / Intent Classifier}
    end

    subgraph Execution Paths
        Router -->|Direct Generation| LLMDirect[Direct LLM Provider]
        Router -->|Unstructured Docs| RAGPipeline[Hybrid RAG Engine]
        Router -->|Structured Data| SQLAgent[Safe Read-Only SQL Engine]
        Router -->|Multi-Step Query| MultiAgent[LangGraph Agentic Orchestrator]
    end

    subgraph Hybrid Retrieval System
        RAGPipeline --> DenseRetriever[Qdrant Vector DB]
        RAGPipeline --> SparseRetriever[Elasticsearch BM25]
        DenseRetriever --> Reranker[Cross-Encoder Reranker]
        SparseRetriever --> Reranker
        Reranker --> CitationEngine[Citation & Grounding Engine]
        CitationEngine --> LLMDirect
    end

    subgraph Structured Database System
        SQLAgent --> SchemaGuard[Read-Only Schema Validator]
        SchemaGuard --> Postgres[(PostgreSQL Business DB)]
    end

    subgraph Observability & Evaluation
        API -.-> Tracing[Langfuse / OpenTelemetry Tracing]
        API -.-> EvalMetrics[Ragas / Hit-Rate Evaluation]
    end
```

---

## 2. Technology Stack & Rationale

| Component | Selected Technology | Why Chosen Over Alternatives |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.11+ | Optimal balance of cutting-edge typing features and enterprise AI library support. |
| **API Framework** | FastAPI + Uvicorn | Native async I/O, auto-generated OpenAPI documentation, Pydantic v2 data validation, and first-class dependency injection. |
| **Configuration** | Pydantic Settings | 12-factor configuration; strictly typed, validated at startup, and secret-safe (`SecretStr`). |
| **LLM Gateway** | Custom Provider Abstraction | Avoids tight framework lock-in. Seamlessly switches between OpenAI, Groq, Ollama (local), vLLM, and a deterministic offline Mock provider. |
| **Vector Database** | Qdrant *(Phase 5)* | Rust-powered high-concurrency vector store with robust payload filtering and snapshot management. |
| **Keyword Search** | Elasticsearch *(Phase 8)* | Enterprise BM25 sparse search with proven reliability for exact keyword, SKU, and code matching. |
| **Code Quality** | Ruff + Mypy + Pytest | Sub-second linting, strict static typing, and high-coverage automated unit/integration test suites. |

---

## 3. Project Structure

```text
d:/PROJECT/Enterprise AI Agent/
├── .env.example                     # Reference environment variables
├── .env                             # Local configuration (defaults to mock mode)
├── .gitignore                       # Git ignore rules
├── pyproject.toml                   # Project metadata, dependencies, Ruff & Mypy configs
├── requirements.txt                 # Production dependencies
├── requirements-dev.txt             # Development and test dependencies
├── README.md                        # Documentation and architecture guide
├── src/
│   └── enterprise_agent/
│       ├── __init__.py
│       ├── config/                  # 12-factor configuration
│       │   ├── __init__.py
│       │   └── settings.py          # Pydantic Settings model
│       ├── core/                    # Core foundation
│       │   ├── __init__.py
│       │   ├── exceptions.py        # Domain exception hierarchy
│       │   └── logging.py           # Structured logger configuration
│       ├── embeddings/              # Dense vector embedding providers & batching
│       │   ├── __init__.py
│       │   ├── base.py              # EmbeddingProvider ABC & EmbeddingResult
│       │   ├── factory.py           # Provider and service factory
│       │   ├── gemini.py            # Gemini text-embedding-004 provider
│       │   ├── mock.py              # Deterministic L2-normalized mock provider
│       │   ├── openai.py            # OpenAI text-embedding-3-small provider
│       │   └── service.py           # EmbeddingsService batch coordinator
│       ├── ingestion/               # Document ingestion, parsing, chunking
│       │   ├── __init__.py
│       │   ├── models.py            # Document, DocumentChunk, IngestionResult models
│       │   ├── sanitization.py      # Untrusted text defense & delimiter neutralizing
│       │   ├── service.py           # IngestionService coordinator & deduplication
│       │   ├── chunking/            # Chunking strategies
│       │   │   ├── __init__.py
│       │   │   ├── base.py          # BaseChunker ABC
│       │   │   └── recursive.py     # RecursiveCharacterChunker with page mapping
│       │   └── parsers/             # Multi-format parsers
│       │       ├── __init__.py
│       │       ├── base.py          # BaseDocumentParser ABC & ParsedDocument
│       │       ├── factory.py       # Parser resolution factory
│       │       ├── markdown.py      # Markdown parser with H1 & section tracking
│       │       ├── pdf.py           # PDF parser with page tracking (pypdf)
│       │       └── text.py          # Plain text parser with encoding fallback
│       ├── llm/                     # LLM Provider Abstraction
│       │   ├── __init__.py
│       │   ├── base.py              # LLMProvider interface & data contracts
│       │   ├── gemini_client.py     # Google Gemini native provider (google-genai SDK)
│       │   ├── openai_client.py     # OpenAI-compatible asynchronous implementation
│       │   ├── mock_client.py       # Deterministic mock provider for tests & offline dev
│       │   └── factory.py           # Factory resolving provider via settings
│       ├── schemas/                 # Request & Response contracts
│       │   ├── __init__.py
│       │   ├── chat.py              # ChatRequest, ChatResponse, TokenUsage
│       │   ├── documents.py         # IngestTextRequest, IngestResponse, DocumentChunkResponse
│       │   ├── embeddings.py        # EmbedTextsRequest, EmbedTextsResponse, EmbedQueryRequest
│       │   ├── rag.py               # RAGQueryRequest, RAGQueryResponse, RetrievedSourceChunk
│       │   └── search.py            # SemanticSearchRequest, SemanticSearchResponse, IndexDocumentResponse
│       ├── rag/                     # Retrieval-Augmented Generation subsystem
│       │   ├── __init__.py
│       │   ├── context.py           # RAGContextAssembler with dynamic token budgeting
│       │   ├── prompts.py           # RAGPromptBuilder & enterprise anti-hallucination guardrails
│       │   └── service.py           # RAGService coordinating vector search & LLM answer synthesis
│       ├── vectorstore/             # Vector database storage & ANN semantic retrieval
│       │   ├── __init__.py
│       │   ├── base.py              # VectorStore ABC & SearchResult domain models
│       │   ├── factory.py           # AsyncQdrantClient & VectorStore resolution factory
│       │   ├── qdrant.py            # QdrantVectorStore adapter with Cosine distance & payload filtering
│       │   └── service.py           # VectorSearchService indexing & semantic query coordinator
│       ├── api/                     # Presentation layer
│       │   ├── __init__.py
│       │   ├── deps.py              # Dependency injection providers
│       │   └── v1/
│       │       ├── __init__.py
│       │       ├── api.py           # v1 router aggregator
│       │       ├── health.py        # GET /api/v1/health & upstream probe
│       │       ├── chat.py          # POST /api/v1/chat & POST /api/v1/chat/stream
│       │       ├── documents.py     # POST /documents/ingest/text, /file, GET /documents, /index, DELETE
│       │       ├── embeddings.py    # POST /embeddings, POST /embeddings/query
│       │       ├── rag.py           # POST /rag/query, POST /rag/stream
│       │       └── search.py        # POST /search/semantic
│       └── main.py                  # FastAPI app factory, lifespan, & error handlers
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Pytest fixtures and mock client setup
    ├── unit/
    │   ├── __init__.py
    │   ├── test_chunking.py         # Recursive chunker & deduplication unit tests
    │   ├── test_config.py           # Settings validation unit tests
    │   ├── test_embeddings.py       # Embedding providers & batching unit tests
    │   ├── test_gemini_service.py   # Gemini message conversion & error mapping unit tests
    │   ├── test_llm_service.py      # LLM abstraction & error mapping unit tests
    │   ├── test_parsers.py          # Text, Markdown, and PDF parser unit tests
    │   ├── test_rag.py              # RAG prompt, context assembler, & service unit tests
    │   ├── test_sanitization.py     # Document text security sanitization tests
    │   └── test_vectorstore.py      # Qdrant vector store & ANN search unit tests
    └── integration/
        ├── __init__.py
        ├── test_chat_api.py         # Direct & SSE streaming endpoints integration tests
        ├── test_documents_api.py    # Document ingestion & chunk inspection API tests
        ├── test_embeddings_api.py   # Dense embeddings batch & query API tests
        ├── test_rag_api.py          # Grounded RAG query & token streaming integration tests
        └── test_search_api.py       # Semantic vector search & indexing API tests
```

---

## 4. Getting Started

### Prerequisites
- Python 3.11+
- Git

### 1. Setup Virtual Environment
```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements-dev.txt
pip install -e .
```

### 3. Configure Environment
Copy `.env.example` to `.env`. By default, `LLM_PROVIDER="mock"` is enabled so you can run the entire system offline immediately without any API keys or billing.
```bash
cp .env.example .env
```

To switch to **Google Gemini** (recommended):
```env
LLM_PROVIDER="gemini"
LLM_MODEL="gemini-2.5-flash"
GEMINI_API_KEY="AIzaSyYourGeminiKeyHere..."
```

To switch to **OpenAI / Groq / Ollama**:
```env
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4o-mini"
LLM_API_KEY="sk-your-actual-api-key"
LLM_BASE_URL="https://api.openai.com/v1"
```

### 4. Run the API Server
```bash
# Using uvicorn
uvicorn enterprise_agent.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI documentation is available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 5. API Verification & Usage

### Check Health Status
```bash
curl -X GET http://localhost:8000/api/v1/health
```
**Response:**
```json
{
  "status": "healthy",
  "app_name": "Enterprise AI Knowledge & Decision Agent",
  "version": "0.1.0",
  "environment": "development",
  "llm_healthy": true
}
```

### Submit Direct Chat Query (`POST /api/v1/chat`)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our company vacation policy?",
    "system_prompt": "You are a professional HR assistant.",
    "temperature": 0.2
  }'
```
**Response:**
```json
{
  "answer": "Standard vacation allowance is 20 days annually...",
  "model": "gemini-2.5-flash",
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 40,
    "total_tokens": 52
  },
  "latency_ms": 312.45,
  "status": "success"
}
```

### Stream Live Tokens over Server-Sent Events (`POST /api/v1/chat/stream`)
```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain how the quarterly bonus calculation works."
  }'
```

### Ingest Plain Text or Markdown (`POST /api/v1/documents/ingest/text`)
```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Corporate Refund Policy",
    "content": "# Customer Refund Policy\n\nFull refunds are issued within 30 calendar days of delivery.",
    "source": "handbook_2026.md"
  }'
```
**Response:**
```json
{
  "document_id": "709befc4-f87f-44b4-a94c-60bf50e7779d",
  "title": "Corporate Refund Policy",
  "source": "handbook_2026.md",
  "num_chunks": 1,
  "total_characters": 83,
  "checksum": "8b7793bc511d48d19bab46ba16a6a307c0c5d8d49efa4639f0b34a36f7ae3e97",
  "status": "success",
  "message": "Successfully ingested and produced 1 chunks."
}
```

### Upload and Ingest Document File (`POST /api/v1/documents/ingest/file`)
```bash
curl -X POST http://localhost:8000/api/v1/documents/ingest/file \
  -F "file=@./docs/company_policy.pdf"
```

### List Ingested Documents (`GET /api/v1/documents`)
```bash
curl -X GET http://localhost:8000/api/v1/documents
```

### Inspect Document Chunks (`GET /api/v1/documents/{document_id}/chunks`)
```bash
curl -X GET http://localhost:8000/api/v1/documents/709befc4-f87f-44b4-a94c-60bf50e7779d/chunks
```

### Generate Dense Embeddings (`POST /api/v1/embeddings`)
```bash
curl -X POST http://localhost:8000/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Enterprise cloud architecture", "High availability vector search"],
    "batch_size": 32
  }'
```

### Semantic Vector Search (`POST /api/v1/search/semantic`)
```bash
curl -X POST http://localhost:8000/api/v1/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the policy for hotel expenses and flight booking?",
    "top_k": 5,
    "min_score": 0.0,
    "filters": {"department": "Operations"}
  }'
```

### Grounded RAG Query (`POST /api/v1/rag/query`)
```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the monthly internet allowance and initial equipment reimbursement?",
    "top_k": 3,
    "min_score": 0.0,
    "filters": {"department": "People & Culture"}
  }'
```

### Real-Time Streaming RAG Query (`POST /api/v1/rag/stream`)
```bash
curl -N -X POST http://localhost:8000/api/v1/rag/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the rules for travel reimbursement?",
    "top_k": 3
  }'
```

### Explicitly Index Ingested Document (`POST /api/v1/documents/{id}/index`)
```bash
curl -X POST http://localhost:8000/api/v1/documents/709befc4-f87f-44b4-a94c-60bf50e7779d/index
```

### Delete Document & Purge Vectors (`DELETE /api/v1/documents/{id}`)
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/709befc4-f87f-44b4-a94c-60bf50e7779d
```

---

## 6. Running Tests & Code Quality

The project enforces strict typing (`mypy`), modern linting (`ruff`), and automated testing (`pytest`).

```bash
# 1. Run unit and integration tests with coverage
pytest --cov=enterprise_agent tests/

# 2. Run Ruff linter and formatter checks
ruff check .
ruff format --check .

# 3. Run strict static type checking
mypy src tests
```

---

## 7. Roadmap & Milestones

- [x] **Milestone 1: Project Foundation & Core LLM Abstraction**
  - Clean architecture directory layout
  - 12-factor configuration via Pydantic Settings
  - Decoupled `LLMProvider` interface with OpenAI-compatible & Mock implementations
  - Base `/api/v1/chat` and `/api/v1/health` endpoints
  - High-coverage unit and integration test suite
- [x] **Milestone 2: Google Gemini Integration & Real-Time Streaming**
  - First-class Google Gemini integration via official `google-genai` SDK (`GeminiLLMProvider`)
  - Server-Sent Events (SSE) `/api/v1/chat/stream` real-time token streaming
  - Transient failure resilience with exponential backoff (`tenacity`)
  - Support for `gemini-2.5-flash`, `gemini-1.5-flash`, and `gemini-1.5-pro`
- [x] **Milestone 3: Enterprise Document Ingestion & Recursive Chunking**
  - Multi-format document parsers (PDF with `pypdf`, Markdown with H1/section extraction, plain text)
  - Security sanitization pipeline (zero-width steganography stripping, null bytes removal, prompt injection delimiter defanging)
  - Recursive character chunker with overlap and exact page range mapping
  - Document ingestion API (`POST /api/v1/documents/ingest/text`, `POST /api/v1/documents/ingest/file`, `GET /api/v1/documents`, `GET /api/v1/documents/{id}/chunks`)
  - Content-addressed SHA-256 deduplication
- [x] **Milestone 4: Vector Embeddings Pipeline & Batch Vectorization**
  - Multi-provider dense embedding abstraction (`EmbeddingProvider` ABC)
  - Google Gemini `text-embedding-004` (768 dims) with asymmetric task conditioning (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`)
  - OpenAI `text-embedding-3-small` / `text-embedding-3-large` (1536 dims)
  - Deterministic L2 unit-normalized offline mock provider (`MockEmbeddingProvider`)
  - Batch slicing coordinator (`EmbeddingsService`) with token tracking & rate limit protection
  - REST API endpoints (`POST /api/v1/embeddings`, `POST /api/v1/embeddings/query`)
- [x] **Milestone 5: Qdrant Vector Store Integration & Semantic Search**
  - High-concurrency vector store abstraction (`VectorStore` ABC)
  - Qdrant integration (`QdrantVectorStore`) with native dual mode: zero-docker in-memory (`:memory:`) & remote server/cloud
  - HNSW index support with Cosine distance metric and deterministic UUIDv5 point mapping
  - Pre-filtered single-stage metadata payload indexing (`document_id`, `source`, department tags)
  - End-to-end `VectorSearchService` coordinating embedding generation, indexing, and ANN retrieval
  - REST API endpoint (`POST /api/v1/search/semantic`, `POST /documents/{id}/index`, `DELETE /documents/{id}`)
  - Auto-indexing integration on document ingestion (`auto_index=true`)
- [x] **Milestone 6: Baseline RAG Pipeline & Streaming Synthesis**
  - Grounded prompt engineering with strict anti-hallucination enterprise guardrails
  - Dynamic context assembler (`RAGContextAssembler`) with greedy relevance sorting & token budgeting
  - RAG orchestration engine (`RAGService`) connecting Qdrant vector retrieval and LLM generation
  - Zero-match short-circuit optimization saving latency and LLM token costs
  - Synchronous endpoint (`POST /api/v1/rag/query`) returning answer, cited source chunks, and telemetry
  - Real-time token streaming (`POST /api/v1/rag/stream`) with upfront SSE source citations
- [x] **Milestone 7: Exact Document Citations & Grounding Verification**
  - Deterministic sentence boundary segmenter (`SentenceSplitter`) with abbreviation awareness
  - Regex citation marker extractor (`CitationExtractor`) supporting `[Source N]`, `[Doc N]`, and multi-source tags
  - Claim-to-chunk factual grounding verifier (`GroundingVerifier`) combining token recall, Jaccard similarity, numeric entity checks, and bigram overlap
  - Extractive quote locator isolating the exact supporting chunk passage (`quote_snippet`)
  - Faithfulness score calculation (0.0 to 1.0) and status classification (`verified`, `partially_grounded`, `unverified`, `insufficient_context`)
- [x] **Milestone 8: Hybrid Search with Elasticsearch (BM25 + Dense Qdrant)**
  - Sparse lexical store abstraction (`SparseStore` ABC)
  - Native in-memory BM25 store (`InMemoryBM25Store`) with Lucene positive-IDF formula and code-preserving tokenizer
  - Production cluster adapter (`ElasticsearchStore`) with text analyzer mappings and metadata bool filters
  - Rank fusion algorithms: Reciprocal Rank Fusion (`reciprocal_rank_fusion` with $k=60$) and normalized linear score combination (`linear_score_fusion`)
  - End-to-end `HybridSearchService` running dense and sparse retrieval concurrently via `asyncio.gather`
  - Dual-store indexing and purge synchronization on document ingestion and deletion
  - REST API endpoints: `POST /api/v1/search/hybrid` and `POST /api/v1/search/sparse`
- [x] **Milestone 9: Cross-Encoder Reranking & Two-Stage Retrieval**
  - Standardized asynchronous cross-encoder interface contract (`Reranker` ABC)
  - Ultra-fast CPU ONNX model execution (`FlashRankReranker`) using `ms-marco-TinyBERT-L-2-v2` with `asyncio.to_thread` non-blocking execution
  - Deterministic testing implementation (`MockReranker`) with token overlap heuristics and forced score overrides
  - Cached singleton factory (`create_reranker`) driven by Pydantic Settings
  - `TwoStageRetrievalService` orchestrating Stage 1 coarse recall ($K_1$) across hybrid/dense/sparse and Stage 2 fine reranking ($K_2$)
  - Enriched search schemas with `rerank_score`, `initial_rank`, `final_rank`, and `initial_score` telemetry
  - REST API endpoints: `POST /api/v1/search/rerank` and `POST /api/v1/search/rerank/direct`
- [x] **Milestone 10: Query Transformation (HyDE, Multi-Query, Step-Back)**
  - Abstract transformation contract (`QueryTransformer` ABC)
  - Hypothetical Document Embeddings (`HyDETransformer`) generating synthetic answers to bridge question-answer vector space asymmetry
  - Query expansion & perspective decomposition (`MultiQueryTransformer`) with automated bullet/numbering stripping and deduplication
  - Problem abstraction (`StepBackTransformer`) deriving broader foundational/architectural questions with conceptual rationales
  - Orchestration service (`QueryTransformationService`) unifying transformation strategies with concurrent retrieval, deduplication, and cross-encoder reranking
  - REST API endpoints: `POST /api/v1/transform/hyde`, `POST /api/v1/transform/multi-query`, `POST /api/v1/transform/step-back`, `POST /api/v1/transform/search`
- [x] **Milestone 11: Tool-Augmented Agent Engine**
  - ReAct (Thought-Action-Observation) loop orchestrator (`AgentService`) with stateful scratchpad
  - Extensible tool abstraction (`BaseTool` ABC & `ToolResult` model) and centralized `ToolRegistry`
  - Safe mathematical evaluation tool (`CalculatorTool`) using AST whitelisting (zero `eval()` injection risk)
  - UTC telemetry and calendar tool (`CurrentTimeTool`)
  - Two-stage hybrid search tool (`KnowledgeSearchTool`) linking agent directly into enterprise RAG pipeline
  - Resilient agent output parser (`parse_agent_output`) handling markdown fences, thoughts, actions, and fallbacks
  - Synchronous execution (`POST /api/v1/agent/chat`) and real-time SSE streaming (`POST /api/v1/agent/stream`)
  - Tool inspection API (`GET /api/v1/agent/tools`) exposing JSON Schemas
- [x] **Milestone 12: Safe Read-Only SQL Database Tool**
  - AST-based SQL security validator (`SQLValidator`) powered by `sqlparse` enforcing single-statement `SELECT`/`WITH` queries
  - Multi-tier mutation and DDL defense rejecting `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `PRAGMA`, and file I/O functions
  - Driver-level defense-in-depth via SQLite URI read-only connection pooling (`file:...mode=ro`)
  - Synthetic enterprise database seeder (`seed_enterprise_db`) generating 4 relational tables (`departments`, `employees`, `products`, `sales_orders`)
  - Row truncation detection and limit clamping (`sql_max_rows`) preventing context window overflow
  - Schema inspection tool (`SQLSchemaTool`) and query execution tool (`SQLQueryTool`) registered in agent `ToolRegistry`
  - REST API endpoints: `GET /api/v1/sql/schema` and `POST /api/v1/sql/query`
  - Multi-step agent integration tests verifying schema introspection, SQL execution, and reasoning
- [x] **Milestone 13: Semantic Query Router**
  - Intent classification across 4 specialized enterprise routing targets: `direct_chat`, `rag_search`, `sql_database`, and `autonomous_agent`
  - High-speed heuristic regex router (`HeuristicRouter`) bypassing LLM overhead for greetings, simple chitchat, compound tasks, and explicit policy lookups
  - Dense semantic cosine embedding router (`SemanticEmbeddingRouter`) classifying queries against curated exemplars via vector similarity thresholding
  - Zero-shot LLM intent classifier (`LLMRouter`) with structured JSON reasoning for ambiguous, multi-domain, or complex queries
  - Cascade orchestration engine (`QueryRouterService`) executing a 3-tier cascade (Heuristics -> Semantic Embedding -> Zero-Shot LLM) with strategy overrides
  - Automatic subsystem dispatch seamlessly executing and formatting payloads from Direct Chat, `RAGService`, `SQLDatabaseService`, or `AgentService`
  - REST API endpoints: `POST /api/v1/router/classify` and `POST /api/v1/router/dispatch`
- [x] **Milestone 14: Enterprise Guardrails & Prompt-Injection Defense**
  - Bidirectional security guardrails pipeline inspecting untrusted user inputs and generated model outputs
  - High-accuracy regex & Luhn-validated PII detection and redaction engine (`PIIRedactor`) masking SSNs, credit cards, emails, phone numbers, and API keys (`sk-*`, `ghp_*`, `AKIA*`)
  - Multi-pattern prompt injection & jailbreak scanner (`PromptInjectionDetector`) flagging directive overrides, DAN modes, roleplay hijacks, base64 obfuscated payloads, and delimiter breakouts
  - Dynamic canary token injection and prompt leakage detection (`CanaryTokenManager`) preventing confidential system instruction exfiltration
  - Centralized orchestration service (`GuardrailsService`) enforcing automated blocking, sanitization, and violation telemetry
- [x] **Milestone 15: RAG Evaluation with Ragas & Custom Metrics**
  - Quantitative RAG evaluation framework implementing the 4 foundational metrics: Faithfulness, Answer Relevance, Context Precision, and Context Recall
  - Claim-to-context factual entailment scoring (`calculate_faithfulness`) detecting hallucinations and unsupported assertions
  - Semantic and lexical answer relevance computation (`calculate_answer_relevance`) measuring question alignment and intent coverage
  - Context Precision metric (`calculate_context_precision`) calculating Mean Average Precision (MAP@K) of retrieved passages
  - Context Recall metric (`calculate_context_recall`) measuring ground-truth statement coverage across retrieved contexts
  - Curated 10-query enterprise golden benchmark dataset (`ENTERPRISE_BENCHMARK_DATASET`) covering HR stipends, travel limits, SOC2 controls, and security policies
  - Evaluation orchestration engine (`RAGEvaluationService`) computing sample-level scores and dataset-level statistical summaries (mean, min, max, threshold verification)
  - REST API endpoints: `POST /api/v1/evaluation/run` and `GET /api/v1/evaluation/dataset`
- [x] **Milestone 16: Experiment Tracking & Benchmarking**
  - Persistent SQLite experiment telemetry repository (`ExperimentRunRepository`) storing run metadata, hyperparameters, evaluation metrics, tags, and execution latency
  - In-memory SQLite connection pooling and multi-process transactional persistence with foreign key support and automatic schema migration
  - Run comparison engine (`RunComparisonEngine`) calculating absolute/percentage metric deltas, polarity-aware improvement detection (quality higher=better, latency lower=better), and benchmark leaderboards
  - Automated grid sweep coordinator (`ExperimentTrackingService.run_grid_sweep`) testing retrieval strategy (dense vs hybrid) and top-k variations against golden benchmark datasets
  - REST API endpoints (`POST /api/v1/experiments/runs`, `GET /api/v1/experiments/runs/{id}`, `GET /api/v1/experiments/runs`, `PUT /api/v1/experiments/runs/{id}`, `POST /api/v1/experiments/compare`, and `POST /api/v1/experiments/grid`)
  - Integration with `RAGEvaluationService` for automated metric evaluation and run logging
- [x] **Milestone 17: Observability (Langfuse / OpenTelemetry Distributed Tracing)**
  - Distributed tracing engine (`Tracer`) with async and sync context managers (`tracer.async_span`, `tracer.span`) and function decorators (`@tracer.trace`)
  - OpenTelemetry GenAI semantic conventions integration tracking model IDs, prompt/completion tokens, execution latency, and operation status
  - W3C `traceparent` context extraction, generation, and cross-service propagation (`00-{trace_id}-{span_id}-01`)
  - Thread-safe bounded in-memory trace buffer (`InMemorySpanExporter`) with FIFO ejection, span nesting aggregation, and query filters
  - OpenTelemetry HTTP JSON payload exporter (`OTLPSpanExporter`) compatible with OpenTelemetry Collector, Langfuse, Jaeger, and Datadog
  - Automatic instrumentation across RAG queries (`rag.query`), vector retrieval (`rag.retrieval`), and LLM synthesis (`rag.synthesis`)
  - REST API endpoints (`GET /api/v1/observability/traces`, `GET /api/v1/observability/traces/{trace_id}`, `DELETE /api/v1/observability/traces`, `GET /api/v1/observability/stats`)
- [x] **Milestone 18: API Surface Polish & Rate Limiting**
  - Thread-safe sliding-window rate limiter (`SlidingWindowRateLimiter`) with burst protection and per-client tracking via IP / `X-API-Key`
  - Standard HTTP 429 response formatting with `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers
  - Distributed request correlation ID middleware (`CorrelationIdMiddleware`) managing `X-Correlation-ID` context propagation
  - OWASP defensive HTTP security headers middleware (`SecurityHeadersMiddleware`) enforcing `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, and scoped `Content-Security-Policy`
  - Polished OpenAPI metadata with 14 comprehensive domain tags, enterprise contact info, and licensing specifications
- [ ] **Milestone 19: End-to-End Testing & Mock Fixtures**
- [ ] **Milestone 20: Docker & Docker Compose Infrastructure**
- [ ] **Milestone 21: Synthetic Enterprise Knowledge Base**
- [ ] **Milestone 22: Interactive Demonstration Scenarios**
- [ ] **Milestone 23: Security & Credential Hardening**
- [ ] **Milestone 24: Latency & Throughput Performance Optimization**
- [ ] **Milestone 25: Resume-Quality Documentation & Architecture Showcase**
