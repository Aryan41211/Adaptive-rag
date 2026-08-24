# Adaptive RAG - Agentic AI Chatbot

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5.4-orange.svg)](https://python.langchain.com/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-purple.svg)](https://qdrant.tech/)

## 📋 Overview

**Adaptive RAG** is an intelligent, end-to-end Retrieval-Augmented Generation (RAG) system powered by agentic AI architecture. It combines dynamic query routing, intelligent document retrieval, and advanced LLM capabilities to provide accurate, context-aware answers to user queries.

The system intelligently adapts its retrieval strategy based on query type, utilizing indexed documents, general knowledge, or real-time web search to generate comprehensive responses. Built with a modular architecture using LangGraph for workflow orchestration and multiple storage backends for scalability.

---

## 🎯 Key Features

### 🧠 Intelligent Query Routing
- **Adaptive Classification**: Automatically routes queries to the most appropriate processing pipeline
- **Three Query Types**:
  - **Index**: Queries answerable from uploaded documents
  - **General**: Queries answerable with general knowledge
  - **Search**: Queries requiring real-time web search

### 📚 Advanced RAG Pipeline
- **Document Processing**: Intelligent chunking and embedding of documents
- **Vector Search**: Fast similarity-based retrieval using Qdrant
- **Relevance Grading**: Automatic evaluation of retrieved documents
- **Query Rewriting**: Optimizes queries for better retrieval results

### 🤖 Agentic AI Architecture
- **Multi-Agent System**: Specialized agents for different tasks
- **ReAct Framework**: Reasoning and Acting pattern for intelligent decision-making
- **Tool Integration**: Seamless integration with retrieval tools and web search

### 💾 State Management
- **MongoDB Backend**: Persistent chat history and session management
- **Session Tracking**: Individual conversation contexts per user
- **Memory Management**: Full conversation context retention

### 🎨 User Interface
- **Streamlit Web App**: Interactive chat interface with document upload
- **File Support**: PDF and TXT document uploads
- **Real-time Feedback**: Live chat with instant responses

### ⚡ API-First Architecture
- **FastAPI Backend**: High-performance REST API
- **Async Operations**: Non-blocking database and API calls
- **RESTful Endpoints**: Well-defined API contracts

---

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                          │
│  ┌──────────────��───────────────────────────────────────��───┐  │
│  │  Streamlit Web Application                               │  │
│  │  • Chat Interface                                        │  │
│  │  • Document Upload (PDF, TXT)                            │  │
│  │  • Session Management                                    │  │
│  └──────────────────────────────────────────────────────────��  │
└───────────────────────────────────────────��─────────────────────┘
                            ↓
┌────────────────────────────────────────────────��────────────────┐
│                       FastAPI Backend                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  REST API Endpoints                                      │  │
│  │  • POST /rag/query                                       │  │
│  │  • POST /rag/documents/upload                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Orchestration                      │
│  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌──────────┐         │
│  │ Query   │→ │ Classify │→ │ Router  │→ │ Pipeline │         │
│  │ Analyze │  │ Query    │  │ Output  │  │ Exec     │         │
│  └─────────┘  └──────────┘  └───��─────┘  └──────────┘         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────��──────────┬────────────────��─┬────────────────┐
        ↓                  ↓                  ↓                ↓
   ┌─────────┐       ┌──────────┐      ┌────────────┐   ┌──────────┐
   │ Retriever│      │ General  │      │ Web Search │   │ Response │
   │ (Index)  │      │ LLM      │      │ (Tavily)   │   │ Generator│
   └─────────┘       └──────────┘      └────────────┘   └──────────┘
        ↓                  ↓                  ↓                ↓
        └──────────────────┬──────────────────┬────────────────┘
                           ↓
            ┌─────────────────────────────────┐
            │   Response to User               │
            └─────────────────────────────────┘
```

### Graph Nodes

1. **query_analysis**: Analyzes and classifies incoming queries
2. **retriever**: Retrieves relevant documents from vector store
3. **grade**: Evaluates relevance of retrieved documents
4. **rewrite**: Optimizes query for better retrieval results
5. **generate**: Generates final response from context
6. **web_search**: Performs real-time web search when needed
7. **general_llm**: Provides general knowledge answers

---

## 📦 Project Structure

```
Adaptive-Rag/
├── src/
│   ├── main.py                       # FastAPI app: lifespan, middleware, error handlers, health
│   ├── api/
│   │   ├── routes.py                 # /rag/* endpoints (authenticated)
│   │   ├── auth_routes.py            # /auth/register, /auth/login
│   │   └── deps.py                   # Bearer-token dependency -> CurrentUser
│   ├── config/
│   │   ├── settings.py               # YAML prompt loader
│   │   └── prompts.yaml              # LLM prompts
│   ├── core/
│   │   ├── config.py                 # Validated environment settings (fail-fast)
│   │   ├── logger.py                 # Logging setup + request-id correlation
│   │   ├── security.py               # bcrypt hashing, JWT issue/verify
│   │   └── exceptions.py             # Domain errors carrying HTTP status
│   ├── db/
│   │   ├── mongo_client.py           # Lazy, optional Motor client
│   │   └── users.py                  # User store (Mongo or in-memory)
│   ├── llms/
│   │   └── openai.py                 # Chat and embedding model factories
│   ├── memory/
│   │   └── chat_history_mongo.py     # (user, session)-scoped history with trimming
│   ├── models/
│   │   ├── state.py                  # Graph state (incl. loop counters)
│   │   ├── query_request.py          # Request/response schemas
│   │   ├── grade.py                  # Relevance grade
│   │   ├── route_identifier.py       # Route classification
│   │   └── verification_result.py    # Answer faithfulness
│   ├── rag/
│   │   ├── graph_builder.py          # LangGraph nodes and wiring
│   │   ├── vector_store.py           # Backend-agnostic per-user document API
│   │   ├── backends/
│   │   │   ├── base.py               # Backend interface
│   │   │   ├── qdrant_backend.py     # Persistent, shared, multi-worker
│   │   │   └── faiss_backend.py      # In-process development fallback
│   │   ├── document_upload.py        # Validation, parsing, chunking, indexing
│   │   └── reAct_agent.py            # Per-user agent, cache keyed on index version
│   └── tools/
│       ├── common_tools.py           # Description enhancement
│       └── graph_tools.py            # Conditional edges, bounded loops
│
├── streamlit_app/
│   ├── home.py                       # Sign-in / registration
│   ├── pages/chat.py                 # Chat and document upload
│   └── utils/api_client.py           # Typed API client with timeouts
│
├── tests/                            # 233 tests (pytest)
│   ├── conftest.py                   # Fixtures, fakes, state reset
│   ├── test_config.py                # Settings validation
│   ├── test_security.py              # Hashing and JWT
│   ├── test_auth_api.py              # Auth endpoints and route protection
│   ├── test_api_query.py             # Query endpoint, validation, errors
│   ├── test_upload.py                # Upload validation and indexing
│   ├── test_upload_api.py            # Upload endpoint end to end
│   ├── test_vector_store.py          # Per-user isolation, run against both backends
│   ├── test_qdrant_backend.py        # Durability and cross-worker visibility
│   ├── test_chat_history.py          # Ownership scoping and trimming
│   ├── test_graph_tools.py           # Routing and loop bounds
│   ├── test_graph_nodes.py           # Node behaviour and degradation
│   └── test_frontend.py              # Page structure and API client
│
├── evals/                            # Answer-quality evaluation harness
│   ├── data/golden.yaml              # Golden dataset: documents + cases
│   ├── dataset.py                    # Loading and validation
│   ├── metrics.py                    # Deterministic scoring
│   ├── runner.py                     # Runs cases through the real pipeline
│   └── report.py                     # Text and JSON reports
│
├── deploy/Caddyfile                  # TLS reverse proxy configuration
├── .env.example                      # Documented configuration template
├── requirements.txt                  # Runtime dependencies (LangChain pinned)
├── requirements-dev.txt              # Test dependencies
├── requirements.lock.txt             # Fully pinned, reproducible install
├── pytest.ini                        # Test configuration
├── README.md                         # This file
├── CODE_STYLE_GUIDE.md               # Code formatting standards
├── QUICK_REFERENCE.md                # Quick reference guide
└── DOCUMENTATION_INDEX.md            # Documentation navigation index
```

---

## 🔌 API Endpoints

### Base URL
```
http://localhost:8000
```

Interactive documentation: `http://localhost:8000/docs`

**All `/rag/*` endpoints require a bearer token.** Obtain one from
`/auth/register` or `/auth/login` and send it as
`Authorization: Bearer <access_token>`.

---

### 1. Register

```http
POST /auth/register
Content-Type: application/json

{ "username": "alice", "password": "a-good-password" }
```

**Response `201`:**
```json
{
  "access_token": "eyJhbGciOi...",
  "token_type": "bearer",
  "expires_in": 3600,
  "username": "alice"
}
```

- `username`: 3-64 chars, letters/digits/`.`/`_`/`-`
- `password`: 8-72 chars (bcrypt's limit), hashed with bcrypt before storage

**Status codes:** `201` created · `409` username taken · `422` invalid input

---

### 2. Login

```http
POST /auth/login
Content-Type: application/json

{ "username": "alice", "password": "a-good-password" }
```

Returns the same token payload as `/auth/register`.

**Status codes:** `200` ok · `401` bad credentials · `422` invalid input

---

### 3. Query

```http
POST /rag/query
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "query": "What is the main topic of the document?",
  "session_id": "user_session_123"
}
```

**Response `200`:**
```json
{
  "answer": "Based on the document, the main topic is...",
  "session_id": "user_session_123",
  "citations": [
    { "source": "resume.pdf", "page": 2, "snippet": "..." }
  ],
  "usage": {
    "calls": 4, "input_tokens": 5310, "output_tokens": 412,
    "total_tokens": 5722, "cost_usd": 0.0173
  }
}
```

`citations` lists the passages the answer was grounded in, and is empty for
general-knowledge and web-search answers. `usage` covers every model call the
turn made.

- `query`: 1-4000 characters, must not be blank
- `session_id`: 1-128 characters, `[A-Za-z0-9._:-]` only

The session is scoped to the authenticated user: two users sending the same
`session_id` get two separate, private conversations.

**Status codes:** `200` ok · `401` missing/invalid token · `422` invalid input
· `502` model provider unavailable · `500` internal error

---

### 4. Document upload

```http
POST /rag/documents/upload
Authorization: Bearer <access_token>
X-Description: Brief description of the document

Form data:
  file: <PDF or TXT file>
```

**Response `200`:**
```json
{
  "filename": "resume.pdf",
  "chunks_indexed": 18,
  "total_chunks": 18,
  "description": "Answers questions about the uploaded resume."
}
```

Documents are indexed into a knowledge base **private to the uploading user**.
Uploading a second document adds to the first rather than replacing it.

Validation applied:

| Check | Failure status |
|---|---|
| Extension is `.pdf` or `.txt` | `415` |
| Contents match the extension (PDF magic bytes / UTF-8 text) | `415` |
| Size within `MAX_UPLOAD_BYTES` (default 10 MB) | `413` |
| File is non-empty and contains extractable text | `422` |
| `X-Description` present, 1-300 chars | `422` |
| Embedding service reachable | `502` |

---

### 5. Query (streaming)

```http
POST /rag/query/stream
Authorization: Bearer <access_token>
Content-Type: application/json

{ "query": "What is the main topic?", "session_id": "user_session_123" }
```

Returns `text/event-stream`. Each frame is `data: {json}` followed by a blank
line. Same pipeline, same quota as `/rag/query`.

| Event | Meaning |
|---|---|
| `token` | A fragment of the answer: `{"type":"token","text":"..."}` |
| `restart` | Discard the answer so far and start again. Emitted when verification rejects an answer and it is regenerated |
| `citations` | The sources the answer was grounded in |
| `usage` | Token counts and estimated cost for the turn |
| `error` | The turn failed; the response has already begun, so this cannot be a status code |
| `done` | Complete, carrying the full answer, citations and usage |

A turn makes several model calls, so the wait before the first token is
noticeable; streaming makes it visible progress rather than silence. Only the
answer-producing nodes stream — the classifier and grader also call models,
but their output is internal routing data.

Failed streams are not written to the conversation history.

---

### 6. List indexed documents

```http
GET /rag/documents
Authorization: Bearer <access_token>
```

**Response `200`:**
```json
{
  "documents": [{ "filename": "resume.pdf", "chunks": 18 }],
  "total_chunks": 18
}
```

---

### 7. Delete a document

```http
DELETE /rag/documents/{filename}
Authorization: Bearer <access_token>
```

Removes one source document and every chunk it produced. `DELETE
/rag/documents` (no filename) clears them all.

**Status codes:** `200` deleted · `401` missing/invalid token · `404` no such
document

---

### 8. Delete your account

```http
DELETE /auth/me
Authorization: Bearer <access_token>
```

Removes the indexed documents, the conversation history and the account
record, then revokes the calling token.

**Status codes:** `204` deleted · `401` missing/invalid token

---

### 9. Sign out

```http
POST /auth/logout
Authorization: Bearer <access_token>
```

Denylists the token until its natural expiry, on every worker.

---

### 10. Usage and cost

```http
GET /metrics
Authorization: Bearer <access_token>
```

**Response `200`:**
```json
{
  "version": "1.0.0",
  "vector_backend": "qdrant",
  "usage": {
    "requests": 42, "calls": 137,
    "input_tokens": 191204, "output_tokens": 18330,
    "total_tokens": 209534, "cost_usd": 0.661
  }
}
```

Counters are per-process; with several workers, scrape each one. Every query
response also carries a `usage` object for that turn.

---

### 11. Clear a conversation

```http
DELETE /rag/sessions/{session_id}
Authorization: Bearer <access_token>
```

**Status codes:** `204` deleted · `401` missing/invalid token

---

### 12. Health probes

| Endpoint | Purpose |
|---|---|
| `GET /` | Service identity and version |
| `GET /healthz` | Liveness: the process is serving |
| `GET /readyz` | Readiness plus dependency status (persistence, web search) |

Every response carries an `X-Request-ID` header (echoed from the request when
supplied), which also appears in the matching log lines.

---

## 📖 Usage Guide

### 1. Prerequisites

| Requirement | Needed? | Notes |
|---|---|---|
| Python 3.10+ | **Required** | Developed and tested on 3.12 |
| OpenAI API key | **Required** | Chat completions and embeddings |
| Qdrant | Recommended | Persistent, shared document storage. Without it an in-process FAISS index is used: documents are lost on restart and the API must run a single worker |
| MongoDB | Recommended | Durable users and chat history. Without it both are kept in memory, lost on restart, and not shared between workers |
| Tavily API key | Optional | Enables the web-search route; without it those queries fall back to general knowledge |

Start the optional services locally with:

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
docker run -d -p 27017:27017 --name mongo mongo:7
```

Then set `QDRANT_URL=http://localhost:6333` and
`MONGODB_URL=mongodb://localhost:27017` in `.env`.

### 2. Installation

```bash
git clone https://github.com/dhruvsinghal09/Adaptive-Rag.git
cd Adaptive-Rag

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
# For an exact, reproducible environment instead:
# pip install -r requirements.lock.txt
```

### 3. Environment Configuration

```bash
cp .env.example .env
```

Then edit `.env`. Two values are **required** and the app refuses to start
without them:

```env
OPENAI_API_KEY=sk-...
JWT_SECRET_KEY=<generate one, see below>
```

Generate a signing secret:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Everything else is optional and documented inline in `.env.example`.
Configuration is validated at startup, so a missing or placeholder value
fails immediately with a clear message rather than surfacing later as an
opaque provider error.

### 4. Running the Application

**Start FastAPI Backend:**
```bash
# Terminal 1: Run FastAPI server
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

**Start Streamlit Frontend:**
```bash
# Terminal 2: Run Streamlit app
streamlit run streamlit_app/home.py
```

**Access the Application:**
- Web Interface: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- ReDoc Documentation: http://localhost:8000/redoc

### 5. Example Usage

**Using the Web Interface:**
1. Navigate to http://localhost:8501
2. Create account or login
3. Upload documents in the sidebar
4. Start chatting in the main chat area

**Using cURL:**
```bash
# Upload a document
curl -X POST http://localhost:8000/rag/documents/upload \
  -H "X-Description: Sample document about Python" \
  -F "file=@document.pdf"

# Query the RAG system
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about Python",
    "session_id": "user_123"
  }'
```

**Using Python:**
```python
import requests

# Query endpoint
response = requests.post(
    "http://localhost:8000/rag/query",
    json={
        "query": "What is Python?",
        "session_id": "user_123"
    }
)
print(response.json())
```

---

## 🔧 Configuration

All settings are environment variables, validated by `src/core/config.py`.

### Required

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key for chat and embeddings |
| `JWT_SECRET_KEY` | Signs access tokens. Minimum 32 characters; placeholder values are rejected |

### Optional

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | *(empty)* | Persistent, shared vector store. In-process FAISS fallback when unset |
| `QDRANT_API_KEY` | *(empty)* | Required by Qdrant Cloud, not by a local container |
| `QDRANT_COLLECTION` | `adaptive_rag_documents` | Collection name; created automatically |
| `TAVILY_API_KEY` | *(empty)* | Enables the web-search route |
| `MONGODB_URL` | *(empty)* | Durable storage; in-memory fallback when unset |
| `MONGODB_DB_NAME` | `adaptive_rag` | Database name |
| `OPENAI_MODEL` | `gpt-4o` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MAX_HISTORY_MESSAGES` | `20` | Conversation turns sent to the model |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size cap (10 MB) |
| `MAX_QUERY_LENGTH` | `4000` | Maximum question length |
| `MAX_REWRITE_ATTEMPTS` | `2` | Bounds the retrieve/rewrite retry loop |
| `MAX_VERIFY_ATTEMPTS` | `1` | Bounds answer-faithfulness regeneration |
| `AGENT_MAX_ITERATIONS` | `5` | ReAct agent step limit |
| `RETRIEVER_TOP_K` | `4` | Chunks retrieved per query |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Token lifetime |
| `RATE_LIMIT_ENABLED` | `true` | Master switch for request quotas |
| `RATE_LIMIT_QUERY_PER_MINUTE` | `20` | Per-user query quota |
| `RATE_LIMIT_UPLOAD_PER_HOUR` | `20` | Per-user upload quota |
| `RATE_LIMIT_AUTH_PER_MINUTE` | `10` | Per-address quota on login/registration |
| `CORS_ALLOW_ORIGINS` | *(empty)* | Comma-separated browser origins; empty means no cross-origin access |
| `ALLOWED_HOSTS` | `*` | Comma-separated hostnames the API answers to |
| `LOG_LEVEL` | `INFO` | Root log level |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Backend URL used by the Streamlit UI |

### `src/config/prompts.yaml`

- **system_prompt** — ReAct agent scaffold (`{tools}`, `{tool_names}`, `{input}`, `{agent_scratchpad}`)
- **classify_prompt** — query routing
- **grading_prompt** — retrieved-context relevance
- **rewrite_prompt** — query reformulation
- **generate_prompt** — final answer
- **verify_prompt** — answer faithfulness

### Query Routing Logic

The system routes queries based on classification:

```
Query Classification
├── "index" → Use retriever (indexed documents)
├── "general" → Use general LLM (common knowledge)
└── "search" → Use web search (real-time information)
```

---

## 🧪 Testing the API

### Using FastAPI Interactive Documentation

1. Navigate to http://localhost:8000/docs
2. Expand endpoint sections
3. Click "Try it out"
4. Enter test data
5. Click "Execute"

### Example Test Cases

**Test 1: Simple Query**
```json
{
  "query": "Hello, how are you?",
  "session_id": "test_user_1"
}
```

**Test 2: Document-Based Query**
```json
{
  "query": "What topics are covered in the uploaded document?",
  "session_id": "test_user_1"
}
```

**Test 3: General Knowledge Query**
```json
{
  "query": "What is machine learning?",
  "session_id": "test_user_1"
}
```

---

## 🔐 Security

### Implemented

- **Authentication** — bearer JWT (HS256) on every `/rag/*` endpoint; passwords hashed with bcrypt
- **Per-user data isolation** — each user has a private document index; conversations are keyed by `(user_id, session_id)`, so guessing another user's `session_id` reveals nothing
- **Input validation** — length and character-set constraints on queries, session ids, usernames and descriptions
- **Upload hardening** — size cap, extension allowlist, content sniffing (PDF magic bytes / UTF-8 decode), filename reduced to its basename
- **Error containment** — internal exceptions are logged in full and returned as a generic message; stack traces and internal detail never reach the client
- **Startup secret validation** — placeholder or short `JWT_SECRET_KEY` values are refused
- **Timing-neutral login** — the password hash is verified even for unknown usernames

- **Rate limiting** — per-user quotas on queries and uploads, and a per-address
  quota on credential endpoints. Counters are shared through MongoDB, so the
  limit applies to the deployment rather than to each worker. Exceeding a quota
  returns `429` with `Retry-After`
- **Token revocation** — `POST /auth/logout` denylists the token's identifier
  until its natural expiry, honoured by every worker
- **CORS and host allowlisting** — off by default (correct for the
  server-rendered UI), configurable via `CORS_ALLOW_ORIGINS` and `ALLOWED_HOSTS`
- **Non-root container** — the image runs as uid 10001 with a healthcheck

- **TLS** — the `tls` compose profile terminates HTTPS with automatic
  certificate renewal and sets HSTS and the standard hardening headers
- **Data deletion** — a user can list and remove individual documents, clear
  them all, or delete their account and everything attached to it

### Still required before public exposure

- **MongoDB credentials/TLS** — supply an authenticated connection string in
  production; the compose stack runs it unauthenticated on a private network
- **Secret management** — `.env` is fine for local use; use a secret manager in
  deployment
- **Backups** — nothing backs up the Qdrant or MongoDB volumes

---

## 🚀 Deployment

### Local development

```bash
# Terminal 1 - API
uvicorn src.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 - UI
streamlit run streamlit_app/home.py
```

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

### Running tests

```bash
pytest                 # full suite
pytest --cov=src       # with coverage
```

### Workers

How many workers you can run depends on where state lives.

**With Qdrant and MongoDB configured — multiple workers:**

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Documents live in Qdrant and users and conversations live in MongoDB, so any
worker can serve any request. The agent cache is invalidated by a value read
back from Qdrant rather than remembered in process, so a worker that did not
handle an upload still picks up the new documents.

**Without them — one worker only:**

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The FAISS index and the in-memory user and history stores are private to each
process. With a second worker an upload lands in one process while the next
query is served by another, which answers *"no documents uploaded"*, and a
user registered on one worker cannot log in on another.

`GET /readyz` reports which backends are active.

### Health probes

| Probe | Endpoint |
|---|---|
| Liveness | `GET /healthz` |
| Readiness | `GET /readyz` |

### Containerisation

```bash
cp .env.example .env          # set OPENAI_API_KEY and JWT_SECRET_KEY
docker compose up --build
```

Brings up the API (`:8000`), the Streamlit UI (`:8501`), Qdrant (`:6333`) and
MongoDB (`:27017`). The API waits for Qdrant and MongoDB to report healthy
before starting, so it never boots into its non-durable fallbacks by accident.

The image is multi-stage, runs as an unprivileged user, and carries a
healthcheck against `/healthz`.

To build the API image alone:

```bash
docker build -t adaptive-rag .
docker run -p 8000:8000 -e OPENAI_API_KEY=... -e JWT_SECRET_KEY=... adaptive-rag
```

### TLS

```bash
DOMAIN=rag.example.com ACME_EMAIL=you@example.com   docker compose --profile tls up -d
```

Adds a Caddy reverse proxy on :80 and :443 that obtains and renews a Let's
Encrypt certificate automatically, redirects HTTP to HTTPS, and sets HSTS,
`X-Content-Type-Options`, `X-Frame-Options` and `Referrer-Policy`. It forwards
the real client address, which the rate limiter keys on.

Use `DOMAIN=localhost` to try it locally; Caddy then issues an internal
certificate rather than contacting Let's Encrypt.

### Evaluating answer quality

```bash
python -m evals                        # run the golden dataset
python -m evals --json report.json     # also write machine-readable output
python -m evals --fail-under 0.8       # non-zero exit below that pass rate
```

Indexes a set of synthetic documents into a scratch user, runs every case
through the real pipeline, and scores four things:

| Metric | What it measures |
|---|---|
| Routing accuracy | Was the question sent to documents, general knowledge or search correctly? |
| Retrieval accuracy | Did the answer cite the document that actually contains the fact? |
| Fact accuracy | Does the answer contain the facts a correct one must? |
| Fabrications | Did it invent an answer to a question the documents cannot answer? |

The documents are synthetic and self-contained, so a case can only pass by
genuinely retrieving them rather than recalling training data. The dataset
includes deliberately unanswerable questions, because confident invention is
the failure mode that matters most.

Scoring is deterministic — there is no LLM judge, which would make the score
depend on a second model and cost money on every run. The trade-off is that it
measures whether the right facts are present, not whether the prose is good.

**This calls the model provider and costs money**, so it is not part of the
test suite or CI. The harness itself is covered by `tests/test_evals.py`.

Add cases in `evals/data/golden.yaml`.

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | What it does |
|---|---|
| Lint | `ruff check` and `ruff format --check` |
| Tests | Full suite on Python 3.11 and 3.12, with coverage |
| Docker | Builds the image and smoke-tests `/healthz` and `/readyz` |

---

## ⚠️ Known limitations

| Limitation | Impact | Status |
|---|---|---|
| Rate limit counters are per-process without MongoDB | The effective limit is multiplied by the worker count | Configure `MONGODB_URL` |
| `/metrics` counters are per-process | Scrape each worker separately | By design |
| No backups of the data volumes | Loss of the host loses the data | Operator concern |
| Model pricing is a static table | Cost estimates drift when providers reprice | Update `src/core/usage.py` |
| FAISS fallback is not durable | Applies only when `QDRANT_URL` is unset | By design; use Qdrant |
| In-memory user/history fallback | Applies only when `MONGODB_URL` is unset | By design; use MongoDB |
| Evaluation is not run in CI | It calls the provider and costs money | Run `python -m evals` deliberately |
| Evaluation scoring is lexical | It checks that required facts appear, not that the prose is good | By design; an LLM judge would add cost and variance |

---

## 📊 Performance Optimization

- **Document Chunking**: Configurable chunk size (1000 chars, 150 overlap)
- **Vector Search**: Efficient similarity search with Qdrant
- **Async Operations**: Non-blocking I/O for better throughput
- **Caching**: Query results cached when applicable
- **Batch Processing**: Document processing in batches

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Make changes following CODE_STYLE_GUIDE.md
4. Commit with descriptive messages (`git commit -m 'feat: Add YourFeature'`)
5. Push to your branch (`git push origin feature/YourFeature`)
6. Open a Pull Request

### Code Quality
- Follow PEP 8 standards
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation
- Run linting: `flake8 src/`

---

## 📚 Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **LLM Framework** | LangChain | ~0.3.27 |
| **Workflow Orchestration** | LangGraph | ~0.5.4 |
| **Web Framework** | FastAPI | Latest |
| **ASGI Server** | Uvicorn | Latest |
| **UI Framework** | Streamlit | Latest |
| **Vector Database** | Qdrant (FAISS fallback) | 1.19 / 1.15 |
| **Chat Database** | MongoDB/InMemory | Latest |
| **Document Processing** | LangChain Community | ~0.3.27 |
| **LLM Provider** | OpenAI | ~0.3.28 |
| **Web Search** | Tavily | Latest |
| **Async DB** | Motor | Latest |
| **Data Validation** | Pydantic | ~2.11.7 |

---

## 📝 Documentation References

- [CODE_STYLE_GUIDE.md](CODE_STYLE_GUIDE.md) - Comprehensive coding standards
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick patterns and templates
- [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) - Full documentation index
- [.env.example](.env.example) - Every configuration option, documented

> `DOCUMENT_UPLOAD_FLOW.md`, `DOCUMENT_FLOW_VISUAL.md` and
> `QDRANT_SETUP_GUIDE.md` describe earlier internals and are kept for
> historical reference only. This README is authoritative.

---

## ❓ FAQ

**Q: How do I upload multiple documents?**  
A: Upload one document at a time through the Streamlit interface. Each upload creates a new indexed collection.

**Q: What's the maximum file size?**  
A: Limited by system memory and Qdrant storage. Typical limit is 100MB per file.

**Q: Can I use different LLM providers?**  
A: Currently configured for OpenAI. You can modify `src/llms/openai.py` to use other providers.

**Q: How is conversation history stored?**  
A: MongoDB stores all chat messages with timestamps and session IDs for full context retention.

**Q: Can I run this without web search?**  
A: Yes, remove Tavily dependency. Queries will use index or general LLM only.

---

## 💬 Support & Contact

For issues, questions, or suggestions:
- Open an [Issue](https://github.com/dhruvsinghal09/Adaptive-Rag/issues)
- Check existing documentation
- Review the code comments

---

## 🙏 Acknowledgments

- Built with LangChain and LangGraph
- Vector search powered by Qdrant
- LLM capabilities by OpenAI
- Web search by Tavily
- UI powered by Streamlit
- Thanks to the open-source community

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Dhruv Singhal**
- GitHub: [@dhruvsinghal09](https://github.com/dhruvsinghal09)
- Project: [Adaptive RAG](https://github.com/dhruvsinghal09/Adaptive-Rag)

---

## 📈 Project Status

### Working and covered by tests

- ✅ Adaptive RAG pipeline (route → retrieve → grade → rewrite → generate → verify)
- ✅ Per-user document upload, indexing and retrieval, with source citations
- ✅ **Streaming answers** over Server-Sent Events, including a restart signal
  when verification rejects an answer mid-stream
- ✅ **Answer-quality evaluation harness** scoring routing, retrieval, facts
  and fabrication against a golden dataset
- ✅ Document management — list, delete individually, or clear all
- ✅ Account deletion — removes documents, history and the account record
- ✅ Persistent vector storage in Qdrant, with an in-process FAISS fallback
- ✅ Horizontal scaling — multiple workers when Qdrant and MongoDB are configured
- ✅ JWT authentication, bcrypt hashing, and server-side sign-out
- ✅ Per-user isolation of documents and conversation history, verified against
  both vector backends
- ✅ Rate limiting with shared counters, bounding per-user model spend
- ✅ Token and cost accounting, per request and cumulative
- ✅ Bounded retry loops (no unbounded model spend)
- ✅ Input validation and upload hardening
- ✅ Structured logging with request correlation ids
- ✅ Health and readiness probes reporting backend status
- ✅ Streamlit UI wired to the API
- ✅ Container image and docker-compose stack, running as a non-root user
- ✅ TLS termination with automatic certificates and hardening headers
- ✅ CI: lint, tests on two Python versions, and a Docker build smoke test
- ✅ Automated test suite (369 tests, 94% coverage of `src/`)

### Not yet done

- ❌ **No end-to-end run against a live model provider** — every test uses a
  fake. The pipeline is verified structurally, not by a real answer, and the
  evaluation harness has never been run for score
- ❌ No distributed tracing
- ❌ No backups of the Qdrant or MongoDB volumes

**Suitable for:** production deployment using the `tls` compose profile,
with Qdrant and MongoDB configured — once you have confirmed a real query
against a live API key.

---

## 🗺️ Roadmap

- [ ] Enhanced context management
- [ ] Multi-language support
- [ ] Performance benchmarks
- [ ] Extended LLM provider support
- [ ] Advanced authentication
- [ ] Real-time collaboration
- [ ] Analytics dashboard
- [ ] Cost optimization

---

**Last Updated**: August 24, 2026  
**Status**: Functionally complete and tested; see Project Status for production caveats  
**Documentation**: ✅ Comprehensive
