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
│   │   ├── vector_store.py           # Per-user FAISS indexes, versioned
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
├── tests/                            # 160 tests (pytest)
│   ├── conftest.py                   # Fixtures, fakes, state reset
│   ├── test_config.py                # Settings validation
│   ├── test_security.py              # Hashing and JWT
│   ├── test_auth_api.py              # Auth endpoints and route protection
│   ├── test_api_query.py             # Query endpoint, validation, errors
│   ├── test_upload.py                # Upload validation and indexing
│   ├── test_upload_api.py            # Upload endpoint end to end
│   ├── test_vector_store.py          # Per-user isolation, cache invalidation
│   ├── test_chat_history.py          # Ownership scoping and trimming
│   ├── test_graph_tools.py           # Routing and loop bounds
│   ├── test_graph_nodes.py           # Node behaviour and degradation
│   └── test_frontend.py              # Page structure and API client
│
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
  "session_id": "user_session_123"
}
```

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

### 5. Clear a conversation

```http
DELETE /rag/sessions/{session_id}
Authorization: Bearer <access_token>
```

**Status codes:** `204` deleted · `401` missing/invalid token

---

### 6. Health probes

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
| Tavily API key | Optional | Enables the web-search route; without it those queries fall back to general knowledge |
| MongoDB | Optional | Durable users and chat history; without it both are kept in memory and lost on restart |

> Qdrant is **not** required. The active vector store is in-process FAISS.
> See [Known limitations](#-known-limitations).

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

### Still required before public exposure

- **HTTPS/TLS** — terminate at a reverse proxy; tokens are bearer credentials
- **Rate limiting** — no per-user or per-IP throttle; LLM spend is currently unbounded
- **CORS policy** — not configured (unnecessary for the server-rendered Streamlit UI, required for a browser SPA)
- **Token revocation** — tokens are valid until expiry; there is no deny-list
- **MongoDB credentials/TLS** — supply an authenticated connection string in production
- **Secret management** — `.env` is fine for local use; use a secret manager in deployment

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

### ⚠️ Single-worker constraint

**Run exactly one worker process.**

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

The FAISS index lives in the process's memory. With multiple workers a
user's upload lands in one process while their next query is served by
another, which answers *"no documents uploaded"*. The same applies to the
in-memory user and chat-history fallbacks when `MONGODB_URL` is unset.

Scaling horizontally requires an external vector store (the Qdrant code path
is present but disabled) — tracked under [Known limitations](#-known-limitations).

### Health probes

| Probe | Endpoint |
|---|---|
| Liveness | `GET /healthz` |
| Readiness | `GET /readyz` |

### Containerisation

No `Dockerfile` or `docker-compose.yml` is included yet.

---

## ⚠️ Known limitations

| Limitation | Impact | Status |
|---|---|---|
| FAISS index is in-memory | Uploaded documents are lost on restart | Qdrant path present but disabled |
| Single worker only | No horizontal scaling | Blocked by the above |
| No rate limiting | LLM spend is unbounded per user | Not implemented |
| No CORS/TLS config | Must sit behind a reverse proxy | Deployment concern |
| No token revocation | Logout is client-side only | Not implemented |
| Docs describe Qdrant | `QDRANT_SETUP_GUIDE.md` documents a path the code does not currently take | Historical |

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
| **Vector Database** | Qdrant/FAISS | Latest |
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
- ✅ Per-user document upload, indexing and retrieval
- ✅ JWT authentication with bcrypt password hashing
- ✅ Per-user isolation of documents and conversation history
- ✅ Bounded retry loops (no unbounded model spend)
- ✅ Input validation and upload hardening
- ✅ Structured logging with request correlation ids
- ✅ Health and readiness probes
- ✅ Streamlit UI wired to the API
- ✅ Automated test suite (160 tests)

### Not yet production-hardened

- ❌ No persistence for the vector index (in-memory FAISS; lost on restart)
- ❌ Single-worker only (see [Deployment](#-deployment))
- ❌ No rate limiting or per-user spend caps
- ❌ No TLS/CORS configuration, no container image, no CI pipeline
- ❌ No token revocation

**Suitable for:** local use, demos, internal single-instance deployment behind
a trusted proxy.
**Not yet suitable for:** untrusted public traffic or multi-replica deployment.

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
