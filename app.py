try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import hashlib
import tempfile

import streamlit as st

from rag import process_pdf, create_llm, create_embeddings, ask_question

st.set_page_config(page_title="Chat with PDF", page_icon="📄", layout="wide")


@st.cache_resource
def get_embeddings():
    """Load the embedding model once per server, not on every upload."""
    return create_embeddings()


if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None


def process_uploaded_pdf(uploaded_file):
    """Save the uploaded PDF temporarily and create the vector store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_pdf_path = temp_file.name

    try:
        return process_pdf(temp_pdf_path, embeddings=get_embeddings())
    finally:
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)


def show_sources(sources):
    """Render a de-duplicated list of referenced pages."""
    with st.expander("📚 Sources"):
        pages = sorted({s["page"] for s in (sources or []) if s.get("page") is not None})
        if pages:
            st.write("Referenced pages: " + ", ".join(str(p) for p in pages))
        else:
            st.write("No page information available.")



if st.session_state.llm is None:
    try:
        st.session_state.llm = create_llm(temperature=0)
    except KeyError:
        st.error("GROQ_API_KEY is missing. Add it to your app's Secrets.")
        st.stop()
    except Exception as e:
        st.error(f"Could not create the LLM: {e}")
        st.stop()


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📄 Chat with your PDF files")

    st.markdown(
        """
Upload a PDF and ask questions about its content.

**Tech stack**

- 🐍 Python
- 🦜 LangChain
- 🤗 FastEmbed embeddings
- 🗄️ Chroma
- 🦙 Groq (GPT-OSS-20b)
- 🎈 Streamlit
"""
    )
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Upload a text-based PDF to start chatting.",
    )
    st.divider()

    if st.session_state.vector_store is not None:
        st.success(f"Ready: {st.session_state.uploaded_file_name}")
    else:
        st.info("Upload a PDF to begin.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- Main page ----------
st.title("📄 Chat with your PDF")
st.markdown("Ask questions about your document using a GPT open-weight model.")

# ---------- Process uploaded PDF ----------
if uploaded_file is not None:
    # Detect new uploads by content hash, not by file name
    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.uploaded_file_hash != file_hash:
        with st.spinner("Processing your PDF... This may take a while..."):
            try:
                st.session_state.vector_store = process_uploaded_pdf(uploaded_file)
                st.session_state.uploaded_file_hash = file_hash
                st.session_state.uploaded_file_name = uploaded_file.name
                # Previous conversation belonged to another PDF
                st.session_state.messages = []
                st.success("PDF processed successfully! You can now ask questions.")
            except Exception as e:
                st.error(f"Could not process the PDF: {e}")
                st.stop()

if st.session_state.vector_store is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")

    st.markdown(
        """
### How it works

1. Upload a PDF
2. The PDF is split into smaller chunks
3. The chunks are converted into embeddings
4. Chroma searches for relevant chunks
5. GPT-OSS-20b generates an answer using those chunks

"""
    )

# ---------- Chat history ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources") is not None:
            show_sources(message["sources"])

# ---------- New question ----------
question = st.chat_input("Ask a question about your PDF...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload a PDF before asking a question.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask_question(
                    vector_store=st.session_state.vector_store,
                    question=question,
                    llm=st.session_state.llm,
                    k=3,
                )
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                show_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")