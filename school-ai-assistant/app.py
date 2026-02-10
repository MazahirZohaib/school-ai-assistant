import os
import glob
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="School AI Assistant", page_icon="🎓", layout="wide")


# ----------------------------
# Helper Functions
# ----------------------------
def extract_text_from_pdfs(pdf_paths):
    combined_text = ""
    per_file_text = []

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            filename = os.path.basename(pdf_path)
            per_file_text.append((filename, text))
            combined_text += f"\n\n--- DOCUMENT: {filename} ---\n\n{text}"

        except Exception as e:
            st.warning(f"⚠️ Could not read {pdf_path}: {e}")

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

    # Gemini model
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
# Load Default PDFs
# ----------------------------
PDF_FOLDER = "default_pdfs"
pdf_paths = glob.glob(f"{PDF_FOLDER}/*.pdf")


# ----------------------------
# UI
# ----------------------------
st.title("🎓 School AI Assistant")
st.caption("This assistant uses pre-loaded school documents. Students cannot upload or change files.")


# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("📚 Default Documents")

    if len(pdf_paths) == 0:
        st.error("❌ No default PDFs found.")
        st.write("Add PDFs inside the folder: `default_pdfs/`")
    else:
        st.success(f"✅ Loaded {len(pdf_paths)} default PDF(s).")
        for p in pdf_paths:
            st.write("• " + os.path.basename(p))

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
    st.rerun()


# ----------------------------
# Auto Build Knowledge Base
# ----------------------------
if len(pdf_paths) > 0 and not st.session_state.docs_loaded:
    with st.spinner("Building knowledge base from default PDFs..."):
        combined_text, per_file_text = extract_text_from_pdfs(pdf_paths)

        if len(combined_text.strip()) < 200:
            st.error("❌ Not enough readable text found in the default PDFs.")
        else:
            db, chunks = build_vector_db(combined_text)
            st.session_state.db = db
            st.session_state.docs_loaded = True


# ----------------------------
# Chat UI
# ----------------------------
st.markdown("---")
st.subheader("💬 Chat")

if len(pdf_paths) == 0:
    st.warning("No default PDFs found. Add PDFs to `default_pdfs/` and restart the app.")

elif not st.session_state.docs_loaded:
    st.warning("Knowledge base is not ready yet. Check your PDFs and try again.")

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
    question = st.text_input("Ask a question from the school documents:")

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
