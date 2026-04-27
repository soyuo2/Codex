import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "vectorstores"
CHAT_LOG_DIR = DATA_DIR / "chat_logs"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
DEFAULT_JSONL_NAME = "gsm_guide_rag_chunks.jsonl"
STYLE_PATH = BASE_DIR / "styles.css"
PROMPT_PATH = BASE_DIR / "system_prompt.txt"

DOC_METADATA_KEYS = (
    "chunk_id",
    "respondent_id",
    "category",
    "question",
    "question_key",
    "major",
    "cohort_or_grade",
    "current_status",
    "source_row",
)
DISPLAY_METADATA = (
    ("질문", "question"),
    ("카테고리", "category"),
    ("기수", "cohort_or_grade"),
    ("상태", "current_status"),
    ("전공", "major"),
)
DEFAULT_GREETING = (
    "안녕 후배야! 혹시 학교 생활이나 프로젝트, 진로 고민이 있으면 편하게 물어봐. "
    "준비된 상담 데이터와 학교 자료를 바탕으로 선배처럼 친근하게 답해줄게."
)

for directory in (DATA_DIR, INDEX_DIR, CHAT_LOG_DIR, KNOWLEDGE_BASE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="GSM 길잡이 선배",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def inject_styles() -> None:
    html(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>")


def find_source_jsonl() -> Path | None:
    candidates = [
        KNOWLEDGE_BASE_DIR / DEFAULT_JSONL_NAME,
        BASE_DIR / DEFAULT_JSONL_NAME,
        *sorted(KNOWLEDGE_BASE_DIR.glob("*.jsonl")),
        *sorted(BASE_DIR.glob("*.jsonl")),
        *sorted(BASE_DIR.rglob(DEFAULT_JSONL_NAME)),
        *sorted(BASE_DIR.rglob("*.jsonl")),
    ]
    return next((path for path in candidates if path.is_file()), None)


def data_signature(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    digest.update(str(path.stat().st_size).encode("utf-8"))
    digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def row_text(row: dict) -> str:
    return str(row.get("text") or row.get("answer") or "").strip()


def row_metadata(row: dict) -> dict:
    tags = row.get("tags", "")
    metadata = {key: row.get(key, "") for key in DOC_METADATA_KEYS}
    metadata["tags"] = ", ".join(tags) if isinstance(tags, list) else str(tags)
    return metadata


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")


def load_documents(jsonl_path: Path) -> list[Document]:
    documents = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            row = json.loads(line)
            if text := row_text(row):
                documents.append(Document(page_content=text, metadata=row_metadata(row)))
    return documents


@st.cache_data(show_spinner=False)
def count_documents(jsonl_path: str) -> int:
    with Path(jsonl_path).open("r", encoding="utf-8") as file:
        return sum(bool(row_text(json.loads(line))) for line in file if line.strip())


@st.cache_resource(show_spinner=False)
def get_retriever(signature: str, jsonl_path: str):
    embeddings = get_embeddings()
    index_path = INDEX_DIR / signature

    if index_path.exists():
        vectorstore = FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(load_documents(Path(jsonl_path)), embeddings)
        vectorstore.save_local(str(index_path))

    return vectorstore.as_retriever(search_kwargs={"k": 6})


def format_retrieved_docs(docs: list[Document]) -> str:
    blocks = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        meta = [
            f"{label}: {metadata[key]}"
            for label, key in DISPLAY_METADATA
            if str(metadata.get(key, "")).strip()
        ]
        blocks.append(
            f"[참고 {index}]\n{' | '.join(meta) or '메타데이터 없음'}\n내용: {doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def chat_log_path(session_id: str) -> Path:
    return CHAT_LOG_DIR / f"{session_id}.json"


def load_chat_history(session_id: str) -> list[dict]:
    path = chat_log_path(session_id)
    if not path.exists():
        return []

    try:
        return json.loads(path.read_text(encoding="utf-8")).get("messages", [])
    except json.JSONDecodeError:
        return []


def save_chat_history() -> None:
    payload = {
        "session_id": st.session_state.session_id,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "messages": st.session_state.messages,
    }
    chat_log_path(st.session_state.session_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_saved_sessions() -> list[dict]:
    sessions = []
    for path in sorted(CHAT_LOG_DIR.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        preview = next(
            (
                message["content"].strip().replace("\n", " ")
                for message in payload.get("messages", [])
                if message.get("role") == "user" and message.get("content")
            ),
            "대화 내용이 비어 있어요",
        )
        preview = preview[:40] + "..." if len(preview) > 40 else preview
        saved_at = payload.get("saved_at", "")
        sessions.append(
            {
                "session_id": payload.get("session_id", path.stem),
                "label": f"{saved_at} | {preview}" if saved_at else preview,
            }
        )
    return sessions


def init_session() -> None:
    st.session_state.setdefault("session_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    st.session_state.setdefault("chat_question", "")

    if "messages" not in st.session_state:
        st.session_state.messages = load_chat_history(st.session_state.session_id) or [
            {"role": "assistant", "content": DEFAULT_GREETING}
        ]


def queue_chat_input() -> None:
    question = st.session_state.get("chat_question", "").strip()
    if question:
        st.session_state.pending_user_input = question
        st.session_state.chat_question = ""


def render_chat_input() -> str | None:
    with st.container():
        st.text_area(
            "메시지 입력",
            key="chat_question",
            placeholder="무엇이든 물어보세요",
            label_visibility="collapsed",
            height=96,
        )

        toolbar_col, spacer_col, submit_col = st.columns([0.28, 0.6, 0.12])
        with toolbar_col:
            html(
                """
                <div class="chat-input-toolbar">
                    <span class="chat-input-plus">+</span>
                    <span class="chat-input-expand">확장</span>
                    <span class="chat-input-caret">⌄</span>
                </div>
                """
            )

        submit_col.button("↑", key="chat_submit", use_container_width=True, on_click=queue_chat_input)
    return st.session_state.pop("pending_user_input", None)


def append_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})
    save_chat_history()


def render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def render_sidebar(jsonl_path: Path | None, chunk_count: int) -> None:
    with st.sidebar:
        html('<div class="sidebar-logo">GSM</div>')
        html('<div class="sidebar-section-title">💡 이용 안내</div>')
        html(
            """
            <div class="sidebar-card">
                <div class="sidebar-card-title">질문 예시</div>
                <ul class="sidebar-list">
                    <li>1학년 때부터 하면 좋은 게 뭐야?</li>
                    <li>기숙사 생활 꿀팁 알려줘</li>
                    <li>프로젝트 팀원을 맞추면 어떻게 해야 해?</li>
                    <li>전공 선택이 고민될 때는 어떻게 해?</li>
                </ul>
            </div>
            """
        )
        html(
            """
            <div class="warning-card">
                ⚠️ 답변은 준비된 상담 데이터와 학교 자료를 바탕으로 생성돼요.<br>
                중요한 결정은 꼭 직접 한 번 더 확인해 주세요.
            </div>
            """
        )
        st.markdown("---")
        html('<div class="section-chip">안내</div>')

        if jsonl_path:
            html(
                f"""
                <div class="info-box">
                    <strong>준비된 상담 데이터가 연결되어 있어요.</strong><br>
                    현재 {chunk_count}개의 RAG 청크를 기반으로 답변하고 있어요.<br>
                    데이터 파일: {jsonl_path.name}
                </div>
                """
            )
        else:
            html(
                """
                <div class="info-box">
                    <strong>상담 데이터가 아직 연결되지 않았어요.</strong><br>
                    관리자가 지식베이스 파일을 준비하면 바로 사용할 수 있어요.
                </div>
                """
            )

        sessions = list_saved_sessions()
        if sessions:
            st.markdown("---")
            html('<div class="section-chip">이전 대화</div>')
            options = {session["label"]: session["session_id"] for session in sessions}
            selected = st.selectbox("저장된 대화 선택", options=list(options), label_visibility="collapsed")
            if st.button("선택한 대화 불러오기"):
                st.session_state.session_id = options[selected]
                st.session_state.messages = load_chat_history(st.session_state.session_id)
                st.rerun()

        if st.button("새 대화 시작"):
            st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.messages = [{"role": "assistant", "content": "새 대화를 시작했어! 또 궁금한 걸 편하게 물어봐."}]
            save_chat_history()
            st.rerun()


def render_hero() -> None:
    html(
        """
        <div class="hero-wrap">
            <div class="hero-title">🎓 GSM 길잡이 선배</div>
            <div class="hero-subtitle">학교 생활, 프로젝트, 진로 고민까지! 선배가 다 알려줄게.</div>
            <div class="hero-badge">
                <span class="hero-icon">💬</span>
                <span>안녕 후배야! 궁금한 게 있으면 편하게 물어봐. 준비된 상담 데이터와 학교 자료를 바탕으로 차근차근 답해줄게.</span>
            </div>
        </div>
        """
    )


def prepare_retriever(jsonl_path: Path | None, signature: str | None):
    if not jsonl_path:
        st.error("지식베이스 파일을 찾지 못했어요. `gsm_guide_rag_chunks.jsonl` 파일이 프로젝트 안에 있는지 확인해줘.")
        return None

    try:
        return get_retriever(signature, str(jsonl_path))
    except Exception as error:
        st.error(f"지식베이스를 준비하는 중 오류가 발생했어요: {error}")
        return None


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        temperature=0.8,
        streaming=True,
    )


def build_rag_chain(retriever):
    if not retriever:
        return None

    prompt = ChatPromptTemplate.from_messages(
        [("system", PROMPT_PATH.read_text(encoding="utf-8")), ("human", "{question}")]
    )
    return (
        {"context": retriever | format_retrieved_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )


def stream_reply(question: str, chain) -> str:
    if not chain:
        response = "지금은 답변에 사용할 상담 데이터가 준비되지 않았어. 잠시 뒤 다시 시도해줘."
        st.write(response)
        return response

    try:
        return st.write_stream(chain.stream(question))
    except Exception as error:
        response = f"답변을 생성하는 중 오류가 발생했어: {error}"
        st.error(response)
        return response


def main() -> None:
    init_session()
    inject_styles()

    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    jsonl_path = find_source_jsonl()
    signature = data_signature(jsonl_path) if jsonl_path else None
    chunk_count = count_documents(str(jsonl_path)) if jsonl_path else 0
    chain = build_rag_chain(prepare_retriever(jsonl_path, signature))

    render_sidebar(jsonl_path, chunk_count)
    render_hero()
    render_messages()

    if user_input := render_chat_input():
        st.chat_message("user").write(user_input)
        append_message("user", user_input)
        with st.chat_message("assistant"):
            append_message("assistant", stream_reply(user_input, chain))


main()
