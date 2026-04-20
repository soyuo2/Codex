import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_pdfs"
INDEX_DIR = DATA_DIR / "vectorstores"
CHAT_LOG_DIR = DATA_DIR / "chat_logs"
PDF_STATE_FILE = DATA_DIR / "active_pdf_set.json"

for directory in [DATA_DIR, UPLOAD_DIR, INDEX_DIR, CHAT_LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="GSM Guide Chatbot",
    page_icon="G",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main { font-size: 1.05rem; }
    .stChatMessage { margin-bottom: 1rem; }
    .stButton > button, .stDownloadButton > button {
        width: 100%;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("GSM Guide Chatbot")
st.subheader("Upload PDF files, reuse embeddings later, and stream answers in real time.")


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    return cleaned or "document.pdf"


def save_uploaded_files(uploaded_files) -> list[Path]:
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_name = sanitize_filename(uploaded_file.name)
        file_path = UPLOAD_DIR / file_name
        file_path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def build_pdf_set_signature(pdf_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(pdf_paths, key=lambda item: item.name.lower()):
        file_bytes = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(hashlib.sha256(file_bytes).digest())
    return digest.hexdigest()


def save_active_pdf_state(signature: str, pdf_paths: list[Path]) -> None:
    payload = {
        "signature": signature,
        "files": [str(path) for path in pdf_paths],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    PDF_STATE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_active_pdf_state() -> dict | None:
    if not PDF_STATE_FILE.exists():
        return None
    try:
        return json.loads(PDF_STATE_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")


def load_documents_from_pdfs(pdf_paths: list[Path]):
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    return documents


@st.cache_resource(show_spinner=False)
def get_retriever(signature: str, pdf_paths_as_str: tuple[str, ...]):
    pdf_paths = [Path(path) for path in pdf_paths_as_str]
    embeddings = get_embeddings()
    index_path = INDEX_DIR / signature

    if index_path.exists():
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        raw_documents = load_documents_from_pdfs(pdf_paths)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        split_documents = splitter.split_documents(raw_documents)
        vectorstore = FAISS.from_documents(split_documents, embeddings)
        vectorstore.save_local(str(index_path))

    return vectorstore.as_retriever(search_kwargs={"k": 4})


def get_chat_log_path(session_id: str) -> Path:
    return CHAT_LOG_DIR / f"{session_id}.json"


def persist_chat_history(session_id: str, messages: list[dict]) -> None:
    payload = {
        "session_id": session_id,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": messages,
    }
    get_chat_log_path(session_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_chat_history(session_id: str) -> list[dict]:
    path = get_chat_log_path(session_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("messages", [])
    except json.JSONDecodeError:
        return []


def export_chat_as_text(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"[{speaker}]")
        lines.append(message["content"])
        lines.append("")
    return "\n".join(lines).strip()


def ensure_session_defaults():
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "messages" not in st.session_state:
        stored_messages = load_chat_history(st.session_state.session_id)
        st.session_state.messages = stored_messages or [
            {
                "role": "assistant",
                "content": (
                    "Hello! Upload PDF files first. I will answer in Korean based only on those documents."
                ),
            }
        ]


ensure_session_defaults()

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


with st.sidebar:
    st.title("Controls")
    st.info(
        """
        - Upload one or more PDF files.
        - Embeddings are cached and reused for the same PDF set.
        - Chat answers are automatically saved to local JSON files.
        - You can also download the current chat as a text file.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload multiple PDF files at once.",
    )

    if uploaded_files:
        saved_pdf_paths = save_uploaded_files(uploaded_files)
        active_signature = build_pdf_set_signature(saved_pdf_paths)
        save_active_pdf_state(active_signature, saved_pdf_paths)
        st.success(f"{len(saved_pdf_paths)} PDF file(s) ready.")
    else:
        active_state = load_active_pdf_state()
        active_signature = active_state["signature"] if active_state else None
        saved_pdf_paths = (
            [
                Path(path)
                for path in active_state.get("files", [])
                if Path(path).exists()
            ]
            if active_state
            else []
        )

    if saved_pdf_paths:
        st.caption("Active PDFs")
        for path in saved_pdf_paths:
            st.write(f"- {path.name}")

    st.download_button(
        label="Download saved answers",
        data=export_chat_as_text(st.session_state.messages),
        file_name=f"gsm_chat_{st.session_state.session_id}.txt",
        mime="text/plain",
    )

    if st.button("Start new chat"):
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "New chat started. Upload PDFs and ask anything about them.",
            }
        ]
        persist_chat_history(st.session_state.session_id, st.session_state.messages)
        st.rerun()


retriever = None
if saved_pdf_paths:
    try:
        retriever = get_retriever(
            active_signature,
            tuple(str(path) for path in saved_pdf_paths),
        )
    except Exception as error:
        st.error(f"Failed to prepare PDF embeddings: {error}")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.6,
    streaming=True,
)

system_prompt = """
You are a warm senior student helping with GSM-related questions.
Always answer in Korean.
Use only the information in [Reference].
If the documents do not contain the answer, say clearly that the answer is not in the PDFs.

Rules:
1. Start with a one-line summary.
2. Keep the tone natural and friendly.
3. Bold important parts with markdown.
4. Break long sentences into shorter lines.
5. Do not invent facts.
6. Stay relevant to the user's question.
7. If there is an actionable next step, explain it clearly.

[Reference]
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{question}")]
)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
) if retriever else None


for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])


user_input = st.chat_input("Ask a question about the uploaded PDFs...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    persist_chat_history(st.session_state.session_id, st.session_state.messages)

    with st.chat_message("assistant"):
        if not rag_chain:
            response = "PDF files have not been uploaded yet. Please upload PDFs first."
            st.write(response)
        else:
            try:
                response = st.write_stream(rag_chain.stream(user_input))
            except Exception as error:
                response = f"An error occurred while generating the answer: {error}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    persist_chat_history(st.session_state.session_id, st.session_state.messages)
