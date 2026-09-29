import uuid
from pathlib import Path
from textwrap import dedent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import FastEmbedEmbeddings
from dotenv import load_dotenv

load_dotenv()


def load_pdf(pdf_path: str):
    """Load a PDF and return a list of LangChain Document objects (one per page)."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The uploaded file must be a PDF.")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # Scanned/image-only PDFs load pages but with empty text
    if not documents or not any(d.page_content.strip() for d in documents):
        raise ValueError(
            "The PDF contains no readable text. It may be a scanned/image-only PDF."
        )

    return documents


def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split PDF documents into smaller chunks for embedding/retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Could not create chunks from the PDF.")
    return chunks


def create_embeddings():
    """Lightweight local embeddings (ONNX based)"""
    return FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_vector_store(chunks, embeddings, persist_directory: str | None = None):
    """Create a Chroma vector store from chunks, in its own unique collection."""
    if not chunks:
        raise ValueError("No chunks were provided")

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=f"pdf_{uuid.uuid4().hex}",  # keeps uploads separate
        persist_directory=persist_directory,  # None = in-memory
    )


def retrieve_documents(vector_store, question: str, k: int = 4):
    """Retrieve the most relevant document chunks for a question."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    return vector_store.similarity_search(question, k=k)


def create_llm(model_name: str = "openai/gpt-oss-20b", temperature: float = 0):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )


def build_context(documents):
    """Combine retrieved document chunks into one context string."""
    if not documents:
        return ""

    context_parts = []
    for document in documents:
        page_number = document.metadata.get("page")

        if page_number is not None:
            # PyPDFLoader uses zero-based page numbers.
            source = f"Page {page_number + 1}"
        else:
            source = "Unknown page"

        context_parts.append(f"[{source}]\n{document.page_content}")

    return "\n\n".join(context_parts)


# Dedented once at import time, so the prompt sent to the LLM is clean.
PROMPT_TEMPLATE = dedent(
    """\
    You are a helpful assistant that answers questions about a PDF document.

    Use ONLY the information provided in the context below.

    If the answer cannot be found in the context, say: "I couldn't find that information in the document."

    Do not make up information. Do not use outside knowledge.

    Context:
    {context}

    Question: {question}

    Answer:"""
)


def create_prompt(question: str, context: str):
    """Create the prompt that will be sent to the LLM."""
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, documents, llm):
    """Generate an answer using the question and retrieved documents."""
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not documents:
        return "I couldn't find relevant information in the document."

    context = build_context(documents)
    prompt = create_prompt(question=question, context=context)
    response = llm.invoke(prompt)

    return response.content


def get_sources(documents):
    """Return source/page information for retrieved chunks."""
    sources = []

    for document in documents:
        page_number = document.metadata.get("page")

        if page_number is not None:
            sources.append(
                {
                    "page": page_number + 1,
                    "source": document.metadata.get("source"),
                }
            )

    return sources


def process_pdf(pdf_path: str, embeddings=None, persist_directory: str | None = None):
    """PDF -> Load -> Split -> Embeddings -> Chroma vector store"""
    documents = load_pdf(pdf_path)
    chunks = split_documents(documents)

    if embeddings is None:
        embeddings = create_embeddings()

    return create_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        persist_directory=persist_directory,
    )


def ask_question(vector_store, question: str, llm, k: int = 4):
    """Question -> Retrieve chunks -> Build context -> Prompt LLM -> Answer."""
    documents = retrieve_documents(vector_store=vector_store, question=question, k=k)
    answer = generate_answer(question=question, documents=documents, llm=llm)
    sources = get_sources(documents)

    return {"answer": answer, "documents": documents, "sources": sources}