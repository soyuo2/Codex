import hashlib
import json
import os
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
INDEX_DIR = DATA_DIR / "vectorstores"
CHAT_LOG_DIR = DATA_DIR / "chat_logs"
PDF_DIR = BASE_DIR

for directory in [DATA_DIR, INDEX_DIR, CHAT_LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


st.set_page_config(
    page_title="GSM 길잡이 선배",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(199, 220, 255, 0.45), transparent 30%),
            linear-gradient(180deg, #f8fbff 0%, #ffffff 42%, #f7f9fc 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.7);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eef3fb 0%, #f5f7fb 100%);
        border-right: 1px solid rgba(63, 88, 136, 0.08);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 1.5rem;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2.6rem;
        padding-bottom: 7rem;
    }

    .hero-wrap {
        text-align: center;
        padding: 2rem 0 0.75rem 0;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #1f2a44;
        letter-spacing: -0.04em;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.7rem;
        font-weight: 700;
        color: #27324a;
        letter-spacing: -0.03em;
        margin-bottom: 2rem;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.85rem 1.1rem;
        background: #ffffff;
        border: 1px solid #e4ebf7;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(31, 42, 68, 0.06);
        color: #2e3a55;
        font-size: 1.02rem;
        font-weight: 600;
    }

    .hero-icon {
        width: 36px;
        height: 36px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #ffb34d 0%, #ff8f2d 100%);
        color: white;
        font-size: 1.1rem;
        box-shadow: 0 8px 18px rgba(255, 155, 50, 0.28);
    }

    .sidebar-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 120px;
        height: 120px;
        margin: 0 auto 1.2rem auto;
        border-radius: 28px;
        background: linear-gradient(180deg, #ffffff 0%, #edf4ff 100%);
        border: 1px solid rgba(43, 93, 174, 0.12);
        box-shadow: 0 18px 40px rgba(44, 65, 105, 0.08);
        color: #1d5fbf;
        font-size: 2rem;
        font-weight: 800;
    }

    .sidebar-section-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #24324b;
        margin: 0.3rem 0 1rem 0;
        letter-spacing: -0.03em;
    }

    .sidebar-card {
        background: linear-gradient(180deg, #dfeafe 0%, #d9e7ff 100%);
        border: 1px solid rgba(76, 120, 197, 0.16);
        border-radius: 20px;
        padding: 1.1rem 1.1rem 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 28px rgba(76, 120, 197, 0.10);
    }

    .sidebar-card-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #23417a;
        margin-bottom: 0.75rem;
    }

    .sidebar-list {
        margin: 0;
        padding-left: 1.2rem;
        color: #264a87;
        font-size: 1rem;
        line-height: 1.7;
        font-weight: 600;
    }

    .warning-card {
        background: linear-gradient(180deg, #fff6d8 0%, #fff2bf 100%);
        border: 1px solid rgba(209, 176, 79, 0.22);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        color: #826420;
        font-size: 1rem;
        line-height: 1.7;
        font-weight: 700;
        box-shadow: 0 12px 28px rgba(209, 176, 79, 0.10);
    }

    .section-chip {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #edf4ff;
        color: #3262b6;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }

    .info-box {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid #e2e9f6;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        color: #31415f;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-top: 0.5rem;
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(208, 220, 240, 0.8);
        border-radius: 22px;
        padding: 0.35rem 0.35rem;
        box-shadow: 0 14px 40px rgba(29, 54, 104, 0.05);
        margin-bottom: 1rem;
    }

    [data-testid="stChatInput"] {
        background: rgba(255, 255, 255, 0.94);
        border-top: none;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 20px !important;
        border: 1px solid #dfe7f4 !important;
        min-height: 58px !important;
        padding-top: 0.9rem !important;
    }

    .stButton > button, .stDownloadButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid #d6e1f2;
        min-height: 46px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def find_source_pdfs() -> list[Path]:
    return sorted(
        [path for path in PDF_DIR.glob("*.pdf") if path.is_file()],
        key=lambda item: item.name.lower(),
    )


def build_pdf_set_signature(pdf_paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in pdf_paths:
        file_bytes = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("utf-8"))
        digest.update(hashlib.sha256(file_bytes).digest())
    return digest.hexdigest()


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
        speaker = "사용자" if message["role"] == "user" else "길잡이 선배"
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
                    "안녕 후배야! 😊 학교 생활이나 프로젝트, 진로 고민이 있으면 편하게 물어봐. "
                    "학교 자료를 바탕으로 선배처럼 친근하게 답해줄게."
                ),
            }
        ]


ensure_session_defaults()

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

pdf_paths = find_source_pdfs()
pdf_signature = build_pdf_set_signature(pdf_paths) if pdf_paths else None

with st.sidebar:
    st.markdown('<div class="sidebar-logo">GSM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section-title">📌 이용 안내</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">질문 예시</div>
            <ul class="sidebar-list">
                <li>1학년 때부터 하면 좋은 건 뭐야?</li>
                <li>기숙사 생활 꿀팁 알려줘!</li>
                <li>프로젝트 팀장을 맡으면 어떻게 해야 해?</li>
                <li>전공 선택이 고민일 때는 어떻게 해?</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="warning-card">
            ⚠️ 답변은 준비된 학교 자료를 바탕으로 생성돼요.<br>
            중요한 결정은 꼭 직접 한 번 더 확인해 주세요.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="section-chip">안내</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box"><strong>바로 질문해 보세요.</strong><br>사용자는 질문만 입력하면 되고, 챗봇이 준비된 자료를 바탕으로 답변해요.</div>',
        unsafe_allow_html=True,
    )

    st.download_button(
        label="답변 기록 내려받기",
        data=export_chat_as_text(st.session_state.messages),
        file_name=f"gsm_chat_{st.session_state.session_id}.txt",
        mime="text/plain",
    )

    if st.button("새 대화 시작"):
        st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "새 대화를 시작했어! 😊 궁금한 걸 다시 편하게 물어봐.",
            }
        ]
        persist_chat_history(st.session_state.session_id, st.session_state.messages)
        st.rerun()


retriever = None
if pdf_paths:
    try:
        retriever = get_retriever(
            pdf_signature,
            tuple(str(path) for path in pdf_paths),
        )
    except Exception as error:
        st.error(f"PDF 임베딩을 준비하는 중 오류가 발생했어요: {error}")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-preview",
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

st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">🎓 GSM 길잡이 선배</div>
        <div class="hero-subtitle">학교 생활, 프로젝트, 진로 고민까지! 선배가 다 알려줄게.</div>
        <div class="hero-badge">
            <span class="hero-icon">💬</span>
            <span>안녕 후배야! 궁금한 게 있으면 편하게 물어봐. 준비된 학교 자료를 바탕으로 차근차근 답해줄게.</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


user_input = st.chat_input("선배에게 질문하기...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    persist_chat_history(st.session_state.session_id, st.session_state.messages)

    with st.chat_message("assistant"):
        if not rag_chain:
            response = "지금은 답변에 사용할 자료가 준비되지 않았어. 잠시 후 다시 시도해줘."
            st.write(response)
        else:
            try:
                response = st.write_stream(rag_chain.stream(user_input))
            except Exception as error:
                response = f"답변을 생성하는 중 오류가 발생했어: {error}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    persist_chat_history(st.session_state.session_id, st.session_state.messages)
