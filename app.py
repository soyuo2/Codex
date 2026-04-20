import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploaded_pdfs"
INDEX_DIR = DATA_DIR / "vectorstores"
CHAT_LOG_DIR = DATA_DIR / "chat_logs"
PDF_STATE_FILE = DATA_DIR / "active_pdf_set.json"

for directory in [DATA_DIR, UPLOAD_DIR, INDEX_DIR, CHAT_LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="GSM 길잡이 챗봇",
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

st.title("GSM 길잡이 챗봇")
st.subheader("PDF 자료를 올리면, 저장된 임베딩을 다시 활용해 실시간으로 답변해줘요.")


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
                    "안녕! PDF 자료를 먼저 올려주면 그 내용을 바탕으로 한국어로 답변해줄게."
                ),
            }
        ]


ensure_session_defaults()

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


with st.sidebar:
    st.title("이용 안내")
    st.info(
        """
        - PDF 파일을 하나 이상 업로드할 수 있어요.
        - 같은 PDF 조합은 임베딩을 저장해두고 다시 재사용해요.
        - 대화와 답변은 자동으로 저장돼요.
        - 현재 대화는 텍스트 파일로 내려받을 수 있어요.
        """
    )

    uploaded_files = st.file_uploader(
        "PDF 파일 업로드",
        type=["pdf"],
        accept_multiple_files=True,
        help="여러 개의 PDF 파일을 한 번에 업로드할 수 있어요.",
    )

    if uploaded_files:
        saved_pdf_paths = save_uploaded_files(uploaded_files)
        active_signature = build_pdf_set_signature(saved_pdf_paths)
        save_active_pdf_state(active_signature, saved_pdf_paths)
        st.success(f"PDF {len(saved_pdf_paths)}개를 준비했어요.")
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
        st.caption("현재 연결된 PDF")
        for path in saved_pdf_paths:
            st.write(f"- {path.name}")

    st.download_button(
        label="저장된 답변 내려받기",
        data=export_chat_as_text(st.session_state.messages),
        file_name=f"gsm_chat_{st.session_state.session_id}.txt",
        mime="text/plain",
    )

    if st.button("새 대화 시작"):
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "새 대화를 시작했어. PDF를 올리고 궁금한 내용을 편하게 물어봐.",
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
        st.error(f"PDF 임베딩을 준비하는 중 오류가 발생했어요: {error}")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.6,
    streaming=True,
)

system_prompt = """
너는 GSM 관련 질문에 답해주는 친근한 선배야.
항상 한국어로 답하고, 아래 [참고 자료]에 있는 내용만 바탕으로 설명해.
PDF에 없는 내용은 추측하지 말고, 자료에 없다고 분명하게 말해줘.

규칙:
1. 첫 줄에는 한 줄 요약을 적어줘.
2. 말투는 자연스럽고 친근하게 유지해.
3. 중요한 부분은 **굵게** 표시해.
4. 문장이 너무 길어지지 않게 적당히 나눠줘.
5. 사실을 지어내지 마.
6. 사용자의 질문과 관련된 내용만 중심으로 답해.
7. 바로 실천할 수 있는 다음 단계가 있으면 구체적으로 적어줘.

[참고 자료]
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


user_input = st.chat_input("업로드한 PDF에 대해 질문해보세요...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    persist_chat_history(st.session_state.session_id, st.session_state.messages)

    with st.chat_message("assistant"):
        if not rag_chain:
            response = "아직 PDF가 업로드되지 않았어요. 먼저 PDF 파일을 올려주세요."
            st.write(response)
        else:
            try:
                response = st.write_stream(rag_chain.stream(user_input))
            except Exception as error:
                response = f"답변을 생성하는 중 오류가 발생했어요: {error}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    persist_chat_history(st.session_state.session_id, st.session_state.messages)
