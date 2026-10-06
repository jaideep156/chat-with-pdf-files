# 📄 Talk to your PDFs

A Retrieval-Augmented Generation (RAG) app that lets you upload a PDF and ask questions about its content. Answers are generated **only from the document**, and every answer shows the pages it came from.

Built with Python, LangChain, FastEmbed, Chroma, Groq (GPT-OSS-20b) and Streamlit.

Click [here](https://talk-to-pdfs.streamlit.app) to try it out!

---

## ✨ Features

- Upload any text-based PDF and chat with it
- Answers grounded strictly in the document (no outside knowledge, no made-up facts)
- Page references for every answer
- Local, lightweight embeddings (ONNX via FastEmbed, no GPU needed)
- Each upload gets its own isolated vector store collection
- Automatic reset of the chat when a new PDF is uploaded (detected by content hash)
- Clear error messages for missing files, non-PDFs, and scanned/image-only PDFs

---

## 🧠 How it works

```
PDF
 └─► Load pages (PyPDFLoader)
      └─► Split into chunks (1000 chars, 200 overlap)
           └─► Embed chunks (all-MiniLM-L6-v2 via FastEmbed)
                └─► Store in Chroma (in-memory)

Question
 └─► Retrieve top-k similar chunks
      └─► Build context with page labels
           └─► Prompt GPT-OSS-20b on Groq
                └─► Answer + source pages
```

1. **Load**: `PyPDFLoader` reads the PDF page by page.
2. **Split**: `RecursiveCharacterTextSplitter` creates overlapping chunks so context isn't lost at chunk boundaries.
3. **Embed**: `FastEmbedEmbeddings` converts each chunk into a vector locally.
4. **Store**: Chroma indexes the vectors in a unique collection per upload.
5. **Retrieve**: your question is embedded and the most similar chunks are fetched.
6. **Generate**: the chunks and question go into a strict prompt sent to the Groq-hosted LLM.
7. **Cite**: the pages of the retrieved chunks are shown under each answer.

---

## 🗂️ Project structure

```
.
├── app.py              # Streamlit UI (upload, chat, sources)
├── rag.py              # RAG pipeline (load, split, embed, retrieve & generate)
├── requirements.txt    # dependencies
├── .env                # Your API key (not committed)
└── README.md
```

---

## 🚀 Getting started

### Step 1: Prerequisites

- **Python 3.10 or newer** 
- **pip**
- A free **Groq API key**: create one at <https://console.groq.com/keys>

### Step 2: Get the code

```bash
git clone https://github.com/jaideep156/chat-with-pdf-files
cd chat-with-pdf-files
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\Activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

> `pysqlite3-binary` is only installed on Linux. It provides a newer SQLite version that Chroma requires, and `app.py` swaps it in automatically at startup.

### Step 5: Add your API key

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Make sure `.env` is listed in your `.gitignore` so your key is never committed.

### Step 6: Run the app

```bash
streamlit run app.py
```

Streamlit will open the app at <http://localhost:8501>.

> **First run note:** FastEmbed downloads the embedding model (a small ONNX file) the first time you process a PDF. This only happens once.

### Step 7: Use it

1. Upload a PDF from the sidebar.
2. Wait for the "PDF processed successfully" message.
3. Type a question in the chat box.
4. Open the **📚 Sources** expander under an answer to see the referenced pages.
5. Use **🗑️ Clear Chat** to wipe the conversation, or upload another PDF to start fresh.

---

## ⚙️ Configuration

Tweak some behavior in the code:

| Setting | Where | Default | What it does |
|---|---|---|---|
| `chunk_size` | `rag.py` → `split_documents` | `1000` | Characters per chunk |
| `chunk_overlap` | `rag.py` → `split_documents` | `200` | Overlap between consecutive chunks |
| `model_name` (embeddings) | `rag.py` → `create_embeddings` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `model_name` (LLM) | `rag.py` → `create_llm` | `openai/gpt-oss-20b` | Groq-hosted chat model |
| `temperature` | `rag.py` → `create_llm` | `0` | `0` gives the most deterministic answers |
| `k` | `app.py` → `ask_question(...)` | `3` | Number of chunks retrieved per question |
| `PROMPT_TEMPLATE` | `rag.py` | strict "context only" | Controls how the LLM answers |
| `persist_directory` | `rag.py` → `process_pdf` | `None` (in-memory) | Set a path to save the vector store to disk |

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push the project to a GitHub repository (without `.env`).
2. Go to <https://share.streamlit.io> and click **New app**.
3. Select your repo, branch, and set the main file to `app.py`.
4. Open **Advanced settings → Secrets** and add:

   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

5. Click **Deploy**.

The `pysqlite3-binary` dependency and the import swap at the top of `app.py` are what make Chroma work on Streamlit Cloud's Linux environment.

---

## 🛠️ Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `GROQ_API_KEY is missing` | Add the key to `.env` (local) or to Secrets (Streamlit Cloud). Restart the app after editing. |
| `The PDF contains no readable text` | The PDF is scanned or image-only. Run OCR on it first (for example with `ocrmypdf`) and upload the result. |
| `sqlite3` version error from Chroma | On Linux, make sure `pysqlite3-binary` installed correctly. On macOS or Windows, update Python or SQLite. |
| First upload is slow | The embedding model is being downloaded and loaded. Later uploads reuse it. |
| Answer says "I couldn't find that information in the document" | The retrieved chunks didn't contain the answer. Try rephrasing, or increase `k` in `app.py`. |
| Model not found / Groq error | Check that the model name in `create_llm` is currently available on your Groq account. |

---

## 🔒 Privacy & Data Handling

Your PDF is processed locally. Only your question and a few small text snippets are sent to Groq to generate an answer.

### Is my data sent to OpenAI?

No. GPT-OSS-20b is an **open-weight model**: OpenAI published the model files, but this app does not call OpenAI's API. Inference runs on **Groq's** servers through the `langchain-groq` integration.

### What stays on the server running the app

| Step | Where it runs |
|---|---|
| PDF upload (temporary file, deleted right after processing) | Local |
| Text extraction and chunking | Local |
| Embeddings (FastEmbed, ONNX) | Local |
| Vector store (Chroma, in-memory) | Local |

### What is sent to Groq

For each question, only the following is sent:

- Your question
- The top 3 most relevant text chunks (about 1,000 characters each)

The full PDF is **never** uploaded to Groq. Note that across many questions, different parts of the document may be sent over time.

### How Groq handles that data

According to Groq's documentation, customer data for inference requests is not retained by default, and you can enable **Zero Data Retention** in the Groq Console under *Data Controls*. Groq may keep short-lived logs for reliability and abuse monitoring unless Zero Data Retention is enabled. Policies can change, so review Groq's current terms before using this app with sensitive material:
<https://console.groq.com/docs/your-data>

### Recommendations

- ✅ Fine for: public documents, personal notes, study material, demos
- ⚠️ Avoid: confidential, medical, legal or otherwise regulated documents unless your Groq account and data controls meet your requirements
- 🔐 Enable **Zero Data Retention** in your Groq Console for extra protection
- 🔑 NEVER commit your `.env` file or API key

### Running fully locally (no third parties)

The LLM is isolated in the `create_llm()` function in `rag.py`. To keep everything on your own machine or infrastructure, replace `ChatGroq` with a locally hosted open-weight model, for example GPT-OSS served through [Ollama](https://ollama.com) with `langchain-ollama`. Embeddings and the vector store are already local, so only that one function needs to change.



## ⚠️ Limitations

- **Text-based PDFs only.** Scanned documents are rejected unless OCR'd first.
- **One PDF at a time.** Uploading a new PDF replaces the current one and clears the chat.
- **In-memory storage.** The vector store is lost when the app restarts (unless you set `persist_directory`).
- **Each question is independent.** The chat history is displayed but not sent back to the LLM, so follow-ups like "what about the second one?" won't have context.
- **Tables and images** inside PDFs are not interpreted, only extracted text is used.

---

## 💡 Ideas for improvement

- Pass chat history to the LLM for follow-up questions
- Support multiple PDFs and other file types (DOCX, TXT)
- Stream the LLM response token by token
- Add OCR support for scanned PDFs
- Show the actual retrieved text snippets in the Sources expander

---

## 📦 Tech stack

| Component | Library |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Orchestration | [LangChain](https://python.langchain.com) |
| PDF parsing | [pypdf](https://pypdf.readthedocs.io) |
| Embeddings | [FastEmbed](https://github.com/qdrant/fastembed) (`all-MiniLM-L6-v2`) |
| Vector store | [Chroma](https://www.trychroma.com) |
| LLM | [Groq](https://groq.com) running `openai/gpt-oss-20b` |

---