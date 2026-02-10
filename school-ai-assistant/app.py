import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="School AI Assistant", page_icon="🎓", layout="wide")
load_dotenv()


# ----------------------------
# Helper Functions
# ----------------------------
def extract_text_from_pdfs(pdf_files):
    combined_text = ""
    per_file_text = []

    for pdf in pdf_files:
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        per_file_text.append((pdf.name, text))
        combined_text += f"\n\n--- DOCUMENT: {pdf.name} ---\n\n{text}"

    return combined_text, per_file_text


def build_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180
    )
    chunks = splitter.split_text(text)

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunks, embedding_model)

    return db, chunks


def gemini_answer(question, context):
    # Get API key from Streamlit Secrets (Streamlit Cloud)
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        return "❌ Missing GEMINI_API_KEY. Please add it in Streamlit Secrets."

    genai.configure(api_key=api_key)

    # Use a valid Gemini model name
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
You are a school assistant chatbot.

RULES:
- Use the provided context as your main source.
- You are allowed to explain and reason using the context.
- If the context does not contain enough information, say exactly:
  "I couldn't find that in the provided school documents."
- Keep answers clear, short, and student-friendly.
- If the question needs steps (like physics or math), show steps briefly.
- Think step-by-step internally before answering, but only show the final answer clearly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = model.generate_content(prompt)
    return response.text



# ----------------------------
# UI
# ----------------------------
st.title("🎓 School AI Assistant")
st.caption("Upload school PDFs (rules, timetable, syllabus, notes) and ask questions. Answers are based on your documents.")

# Sidebar
with st.sidebar:
    st.header("⚙️ Controls")

    pdf_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.markdown("---")
    st.subheader("📌 Tips")
    st.write("• Upload **school rules**, **timetable**, **syllabus**, and **notes**.")
    st.write("• The assistant will answer only using those PDFs.")
    st.write("• Use the **Sources** dropdown to verify answers.")

    st.markdown("---")
    reset = st.button("🧹 Reset Chat")


# ----------------------------
# Session State
# ----------------------------
if "db" not in st.session_state:
    st.session_state.db = None

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

if "chat" not in st.session_state:
    st.session_state.chat = []

if reset:
    st.session_state.chat = []
    st.session_state.db = None
    st.session_state.docs_loaded = False
    st.rerun()


# ----------------------------
# Build DB Button
# ----------------------------
colA, colB = st.columns([1, 1])

with colA:
    st.subheader("1) Upload Documents")

with colB:
    st.subheader("2) Build Knowledge Base")


if pdf_files and len(pdf_files) > 0:
    st.success(f"✅ {len(pdf_files)} PDF(s) uploaded.")

    if st.button("📚 Build / Rebuild Knowledge Base"):
        with st.spinner("Reading PDFs and building vector database..."):
            combined_text, per_file_text = extract_text_from_pdfs(pdf_files)

            if len(combined_text.strip()) < 200:
                st.error("❌ Not enough readable text found in the PDFs. Try different PDFs (not scanned images).")
            else:
                db, chunks = build_vector_db(combined_text)
                st.session_state.db = db
                st.session_state.docs_loaded = True

        if st.session_state.docs_loaded:
            st.success("✅ Knowledge base built successfully!")
            st.info("Now ask questions in the chat below.")

else:
    st.warning("Upload at least 1 PDF to begin.")


st.markdown("---")


# ----------------------------
# Chat UI
# ----------------------------
st.subheader("💬 Chat")

if not st.session_state.docs_loaded:
    st.write("Upload PDFs and build the knowledge base first.")
else:
    # Show chat history
    for msg in st.session_state.chat:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(f"**🧑 You:** {content}")
        else:
            st.markdown(f"**🤖 Assistant:** {content}")

    st.markdown("")

    question = st.text_input("Ask a question from your school documents:")

    if st.button("Ask") and question.strip():
        # Add user message
        st.session_state.chat.append({"role": "user", "content": question})

        # Retrieve context
        db = st.session_state.db
        docs = db.similarity_search(question, k=8)
        context = "\n\n".join([d.page_content for d in docs])

        # Get answer
        with st.spinner("Thinking..."):
            answer = gemini_answer(question, context)

        # Add assistant message
        st.session_state.chat.append({"role": "assistant", "content": answer})

        # Display answer immediately
        st.markdown(f"**🤖 Assistant:** {answer}")

        # Show sources
        with st.expander("📌 Sources used (from your PDFs)"):
            for i, d in enumerate(docs):
                st.markdown(f"**Source {i+1}**")
                st.write(d.page_content)
                st.markdown("---")

        st.rerun()
