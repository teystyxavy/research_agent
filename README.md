# Agentic Research Assistant

An autonomous research agent built with LangGraph and the ReAct (Reasoning + Acting) pattern. Given a topic, the agent plans a research strategy, searches the web and a private knowledge base, synthesises findings, and saves a structured markdown report — all without human intervention.

---

## Features

- **ReAct agent loop** — the LLM reasons about what to do next, calls tools, observes results, and iterates
- **RAG (Retrieval-Augmented Generation)** — upload private documents (PDF, TXT, MD) that the agent queries alongside the web
- **Multi-tool** — web search via Tavily, knowledge base retrieval via ChromaDB, calculator, and report saving
- **Persistent knowledge base** — documents ingested via CLI or the UI are stored in ChromaDB and survive restarts
- **Configurable** — all tuneable parameters exposed via `.env`, no code changes needed
- **Streamlit UI** — live step-by-step streaming of agent execution with document upload

---

## Architecture

The agent is a LangGraph state machine with three nodes and conditional routing:

```
         ┌─────────────────────────────────────────┐
         │                                         │
  START ──► agent ──► should_continue ──► tools ───┘
                            │
                            ├──► human_review ──► END  (hard stop at MAX_ITERATIONS)
                            │
                            └──► END  (no tool calls → research complete)
```

**Nodes:**
- `agent` — the LLM (Groq / Llama) reasons over the message history and decides which tool to call next
- `tools` — LangGraph's `ToolNode` executes the chosen tool and appends the result to state
- `human_review` — hard stop triggered after `MAX_ITERATIONS` tool calls to prevent runaway loops

**State (`AgentState`):**

| Field | Type | Purpose |
|---|---|---|
| `messages` | `List[BaseMessage]` | Full conversation history with `add_messages` reducer |
| `topic` | `str` | Research topic passed in at the start |
| `research_notes` | `List[str]` | Intermediate notes (reserved for future use) |
| `final_report` | `str` | Final report content (reserved for future use) |
| `iteration_count` | `int` | Tracks tool call iterations for the hard stop |
| `approved` | `bool` | Human review gate (currently always False → stops) |

**Tools:**

| Tool | Description |
|---|---|
| `web_search` | Searches the web via Tavily, returns truncated results |
| `retrieve_from_knowledge_base` | Semantic search over uploaded documents in ChromaDB |
| `save_report` | Writes the final markdown report to `./reports/` |
| `calculate` | Evaluates mathematical expressions |

**RAG pipeline:**

```
Upload (UI or CLI)
        │
        ▼
  File bytes (PDF / TXT / MD)
        │
  pypdf / UTF-8 decode
        │
  RecursiveCharacterTextSplitter  (CHUNK_SIZE / CHUNK_OVERLAP)
        │
  GoogleGenerativeAIEmbeddings    (EMBEDDING_MODEL)
        │
  ChromaDB  (CHROMA_PERSIST_DIR)  ◄──── retrieve_from_knowledge_base tool
```

---

## Project Structure

```
research_agent/
│
├── agent/
│   ├── config.py       # All config vars read from .env with defaults
│   ├── graph.py        # LangGraph state machine definition
│   ├── prompts.py      # System prompt for the research agent
│   ├── rag.py          # RAGAgent class — ChromaDB + embeddings + text splitting
│   ├── state.py        # AgentState TypedDict
│   └── tools.py        # Tool definitions registered with the agent
│
├── knowledge_base/     # Drop source documents here before running ingest.py
├── reports/            # Agent-generated reports saved here
├── chroma_db/          # ChromaDB vector store (auto-created, gitignored)
│
├── app.py              # Streamlit web application
├── ingest.py           # CLI script to pre-populate the knowledge base
├── .env                # API keys and configuration (gitignored)
└── .env.example        # Template for required environment variables
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd research_agent
python -m venv agent_venv
agent_venv\Scripts\activate        # Windows
# source agent_venv/bin/activate   # macOS / Linux
```

### 2. Install dependencies

```bash
pip install langchain langchain-groq langchain-google-genai langchain-chroma \
            langchain-community langgraph streamlit python-dotenv \
            tavily-python pypdf chromadb langchain-text-splitters
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required keys:

| Variable | Where to get it |
|---|---|
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) — free, no credit card |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) — free tier available |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) — for embeddings only |

---

## Usage

### Run the web app

```bash
streamlit run app.py
```

Enter a research topic in the text field and click **Run Agent**. The agent streams each step live — tool calls, search queries, and intermediate reasoning are shown as collapsible steps. The final report appears at the bottom when research is complete and is also saved to `./reports/`.

**To use the knowledge base from the UI:** upload PDF, TXT, or MD files via the sidebar before clicking Run. Files are ingested on upload and persist across sessions.

### Pre-populate the knowledge base (CLI)

Place documents in `./knowledge_base/` then run:

```bash
# Ingest all documents in ./knowledge_base/
python ingest.py

# Ingest from a custom directory
python ingest.py ./path/to/docs

# Clear the knowledge base and re-ingest
python ingest.py --reset
```

Supported formats: `.pdf`, `.txt`, `.md`

---

## Configuration Reference

All settings are in `.env`. The agent reads them at startup via `agent/config.py`.

```env
# --- Secrets ---
GROQ_API_KEY=           # Groq API key (LLM)
GOOGLE_API_KEY=         # Google API key (embeddings)
TAVILY_API_KEY=         # Tavily API key (web search)

# --- LLM ---
LLM_MODEL=llama-3.1-8b-instant     # Any Groq-supported model
LLM_TEMPERATURE=0                  # 0 = deterministic

# --- Agent behaviour ---
MAX_ITERATIONS=6                   # Hard stop after N tool calls

# --- Tools ---
TAVILY_MAX_RESULTS=3               # Search results per query
SEARCH_RESULT_MAX_CHARS=800        # Truncate each result to N chars (controls token usage)
REPORTS_DIR=./reports              # Where to save generated reports

# --- RAG ---
EMBEDDING_MODEL=models/text-embedding-004   # Google embedding model
CHROMA_PERSIST_DIR=./chroma_db             # ChromaDB storage path
KNOWLEDGE_BASE_DIR=./knowledge_base         # Default source document directory
KNOWLEDGE_BASE_NAME=agent_knowledge_base    # ChromaDB collection name
CHUNK_SIZE=1000                             # Characters per chunk
CHUNK_OVERLAP=200                           # Overlap between chunks
RAG_K=5                                     # Number of chunks to retrieve per query

# --- App ---
THREAD_ID=research-session-1       # LangGraph memory thread identifier
```

### Choosing a Groq model

| Model | TPM (free tier) | Notes |
|---|---|---|
| `llama-3.1-8b-instant` | 20,000 | Recommended for development — higher rate limits |
| `llama-3.3-70b-versatile` | 12,000 | Better output quality, lower rate limit |

---

## Design Decisions

**Why LangGraph over a simple chain?**
A chain runs in a fixed sequence. LangGraph allows the agent to dynamically decide how many tool calls to make, loop back after observing results, and branch on conditions (e.g. the iteration hard stop). The state machine also makes it straightforward to add proper human-in-the-loop interrupts in future.

**Why RAG as a tool, not a fixed first step?**
Making retrieval a tool means the agent decides when it's relevant — it won't waste tokens querying an empty knowledge base on every run, and it can mix KB results with web results based on what each query actually needs.

**Why truncate web search results?**
Full Tavily responses can be 3,000+ tokens each. With 5 results per query and several queries per run, the message history quickly exceeds free-tier LLM context limits. Truncating to 800 chars per result keeps the context manageable while preserving the key facts.

**Why ChromaDB over FAISS?**
ChromaDB persists automatically to disk without manual save/load calls, supports metadata filtering, and handles collection management (create/get) cleanly. FAISS requires explicit serialisation and is better suited to high-performance production scenarios.
