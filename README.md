# 🎓 College Admission Agent

A Retrieval-Augmented Generation (RAG) chatbot that answers student questions about
college admissions — covering eligibility, fee structures, course selection, and FAQs —
using only your own official documents as the knowledge source.

---

## Problem Statement

Prospective students frequently struggle to find accurate, up-to-date answers buried
across multiple admission documents. Manually searching PDFs and text files is slow and
error-prone. This agent solves that by ingesting your institution's documents once,
indexing them in a vector database, and letting students ask natural-language questions
and receive grounded, cited answers in real time — with zero hallucination risk because
the model is constrained to answer only from retrieved context.

---

## Architecture

```
IBM Cloud Object Storage
        │  (.txt files)
        ▼
   ingest.py
        │  TextLoader → RecursiveCharacterTextSplitter (chunk_size=500)
        │  HuggingFaceEmbeddings (all-MiniLM-L6-v2)
        ▼
   ChromaDB (./chroma_db)   ←──────────────────────────────┐
                                                            │
   app.py  (Streamlit UI)                                   │
        │                                                   │
        │  user question                                    │
        ▼                                                   │
   rag_chain.py                                             │
        │  Retriever (k=4) ──── vector similarity search ──┘
        │  ChatPromptTemplate  (system: answer only from context)
        │  ChatOllama  →  IBM Granite 3.2:2b (via Ollama)
        ▼
   answer + source filenames  →  Streamlit chat bubble
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **IDE / AI Assistant** | IBM Bob |
| **Document Storage** | IBM Cloud Object Storage (ibm-cos-sdk) |
| **Embeddings** | `all-MiniLM-L6-v2` via `sentence-transformers` |
| **Vector Store** | ChromaDB (persistent, local) |
| **LLM** | IBM Granite 3.2:2b served via Ollama |
| **RAG Framework** | LangChain (LCEL chain) |
| **UI** | Streamlit |
| **Language** | Python 3.10+ |

---

## Project Structure

```
College-Admission-Agent/
├── ingest.py           # Download from COS, chunk, embed, store in ChromaDB
├── rag_chain.py        # Load ChromaDB, build RAG chain, expose answer_question()
├── app.py              # Streamlit chat UI
├── requirements.txt    # Python dependencies
├── Admission_faq.txt   # Sample knowledge-base document
├── Course_Selection.txt
├── Eligibility.txt
├── Fee Structure.txt
└── chroma_db/          # Created automatically by ingest.py
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed and running
- An IBM Cloud account with a Cloud Object Storage instance

### 2. Clone & install dependencies

```bash
git clone <your-repo-url>
cd College-Admission-Agent
pip install -r requirements.txt
```

### 3. Pull IBM Granite via Ollama

```bash
ollama pull granite3.2:2b
```

Verify Ollama is serving:

```bash
ollama serve   # leave this running in a separate terminal
```

### 4. Upload documents to IBM Cloud Object Storage

Upload your `.txt` admission documents to a COS bucket. The four sample files included
in this repo (`Admission_faq.txt`, `Course_Selection.txt`, `Eligibility.txt`,
`Fee Structure.txt`) can be used as-is.

### 5. Set COS environment variables

```bash
# Linux / macOS
export COS_API_KEY="your-ibm-cloud-api-key"
export COS_INSTANCE_ID="crn:v1:bluemix:public:cloud-object-storage:..."
export COS_ENDPOINT="https://s3.us-south.cloud-object-storage.appdomain.cloud"
export COS_BUCKET="your-bucket-name"
```

```powershell
# Windows PowerShell
$env:COS_API_KEY     = "your-ibm-cloud-api-key"
$env:COS_INSTANCE_ID = "crn:v1:bluemix:public:cloud-object-storage:..."
$env:COS_ENDPOINT    = "https://s3.us-south.cloud-object-storage.appdomain.cloud"
$env:COS_BUCKET      = "your-bucket-name"
```

> **Where to find these values:**  
> - `COS_API_KEY` — IBM Cloud → Manage → Access (IAM) → API keys  
> - `COS_INSTANCE_ID` — COS instance → Service credentials → `resource_instance_id`  
> - `COS_ENDPOINT` — COS instance → Endpoints (choose your region)  
> - `COS_BUCKET` — the name of the bucket you created

### 6. Ingest documents

```bash
python ingest.py
```

Expected output:

```
Listing .txt files in bucket 'your-bucket-name' ...
Found 4 .txt file(s): ['Admission_faq.txt', 'Course_Selection.txt', ...]

  Downloading 'Admission_faq.txt' ... done.
    → 1 document(s), 12 chunk(s)
  ...

Total chunks across all files: 47

Loading embedding model (all-MiniLM-L6-v2) ...
Embedding chunks and persisting to ./chroma_db ...

Done. 47 chunk(s) stored in ./chroma_db
```

### 7. Launch the chat app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. The sidebar lists
all ingested documents. Type any admissions-related question and the agent will answer
with cited sources.

---

## Example Questions

- *"What are the eligibility criteria for undergraduate admission?"*
- *"What are the one-time or refundable fees?"*
- *"What branches or courses are available for B.Tech?"*
- *"What documents do I need to submit with my application?"*

---

## How It Works

1. **Ingest** — `ingest.py` downloads `.txt` files from IBM COS, splits them into
   500-character overlapping chunks, embeds each chunk with `all-MiniLM-L6-v2`, and
   stores the vectors in a local ChromaDB collection.

2. **Retrieve** — On each user question, `rag_chain.py` performs a cosine-similarity
   search and retrieves the 4 most relevant chunks.

3. **Generate** — The chunks are injected into a structured prompt that instructs IBM
   Granite to answer *only* from the provided context and cite the source filename.
   Granite returns a grounded, hallucination-resistant answer.

4. **Display** — `app.py` renders the conversation in a Streamlit chat interface,
   storing history in `st.session_state` and surfacing source filenames in a collapsible
   expander beneath each answer.

---

## License

MIT
