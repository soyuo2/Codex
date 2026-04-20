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

for directory in [DATA_DIR, INDEX_DIR, CHAT_LOG_DIR, KNOWLEDGE_BASE_DIR]:
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
        color: #1f2a44 !important;
    }

    [data-testid="stChatMessage"] * {
        color: #1f2a44 !important;
    }

    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] * {
        color: #1f2a44 !important;
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

    .stButton > button {
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


def find_source_jsonl() -> Path | None:
    candidate_paths = [
        KNOWLEDGE_BASE_DIR / DEFAULT_JSONL_NAME,
        BASE_DIR / DEFAULT_JSONL_NAME,
    ]
    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path

    for path in sorted(KNOWLEDGE_BASE_DIR.glob("*.jsonl")):
        if path.is_file():
            return path

    for path in sorted(BASE_DIR.glob("*.jsonl")):
        if path.is_file():
            return path

    for path in sorted(BASE_DIR.rglob(DEFAULT_JSONL_NAME)):
        if path.is_file():
            return path

    for path in sorted(BASE_DIR.rglob("*.jsonl")):
        if path.is_file():
            return path

    return None


def build_data_signature(data_path: Path) -> str:
    file_bytes = data_path.read_bytes()
    digest = hashlib.sha256()
    digest.update(data_path.name.encode("utf-8"))
    digest.update(str(data_path.stat().st_size).encode("utf-8"))
    digest.update(hashlib.sha256(file_bytes).digest())
    return digest.hexdigest()


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")


def load_documents_from_jsonl(jsonl_path: Path) -> tuple[list[Document], int]:
    documents = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text") or row.get("answer") or "").strip()
            if not text:
                continue

            metadata = {
                "chunk_id": row.get("chunk_id", ""),
                "respondent_id": row.get("respondent_id", ""),
                "category": row.get("category", ""),
                "question": row.get("question", ""),
                "question_key": row.get("question_key", ""),
                "major": row.get("major", ""),
                "cohort_or_grade": row.get("cohort_or_grade", ""),
                "current_status": row.get("current_status", ""),
                "tags": ", ".join(row.get("tags", [])) if isinstance(row.get("tags"), list) else str(row.get("tags", "")),
                "source_row": row.get("source_row", ""),
            }
            documents.append(Document(page_content=text, metadata=metadata))

    return documents, len(documents)


@st.cache_resource(show_spinner=False)
def get_retriever(signature: str, jsonl_path_str: str):
    jsonl_path = Path(jsonl_path_str)
    embeddings = get_embeddings()
    index_path = INDEX_DIR / signature

    if index_path.exists():
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    else:
        documents, _ = load_documents_from_jsonl(jsonl_path)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(str(index_path))

    return vectorstore.as_retriever(search_kwargs={"k": 4})


def get_chat_log_path(session_id: str) -> Path:
    return CHAT_LOG_DIR / f"{session_id}.json"


def list_saved_chat_sessions() -> list[dict]:
    sessions = []
    for path in sorted(CHAT_LOG_DIR.glob("*.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        messages = payload.get("messages", [])
        preview = "대화 내용이 비어 있어요."
        for message in messages:
            if message.get("role") == "user" and message.get("content"):
                preview = message["content"].strip().replace("\n", " ")
                break

        preview = preview[:40] + "..." if len(preview) > 40 else preview
        session_id = payload.get("session_id", path.stem)
        saved_at = payload.get("saved_at", "")
        sessions.append(
            {
                "session_id": session_id,
                "saved_at": saved_at,
                "label": f"{saved_at} | {preview}" if saved_at else preview,
            }
        )

    return sessions


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
                    "준비된 상담 데이터와 학교 자료를 바탕으로 선배처럼 친근하게 답해줄게."
                ),
            }
        ]


ensure_session_defaults()

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

jsonl_path = find_source_jsonl()
data_signature = build_data_signature(jsonl_path) if jsonl_path else None
chunk_count = 0
if jsonl_path:
    _, chunk_count = load_documents_from_jsonl(jsonl_path)

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
            ⚠️ 답변은 준비된 상담 데이터와 학교 자료를 바탕으로 생성돼요.<br>
            중요한 결정은 꼭 직접 한 번 더 확인해 주세요.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown('<div class="section-chip">안내</div>', unsafe_allow_html=True)

    if jsonl_path:
        st.markdown(
            (
                '<div class="info-box"><strong>준비된 상담 데이터가 연결되어 있어요.</strong><br>'
                f'현재 {chunk_count}개의 RAG 청크를 기반으로 답변하고 있어요.<br>'
                f'데이터 파일: {jsonl_path.name}</div>'
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-box"><strong>상담 데이터가 아직 연결되지 않았어요.</strong><br>관리자가 지식베이스 파일을 준비하면 바로 사용할 수 있어요.</div>',
            unsafe_allow_html=True,
        )

    saved_sessions = list_saved_chat_sessions()
    if saved_sessions:
        st.markdown("---")
        st.markdown('<div class="section-chip">이전 대화</div>', unsafe_allow_html=True)
        session_options = {session["label"]: session["session_id"] for session in saved_sessions}
        selected_label = st.selectbox(
            "저장된 대화 선택",
            options=list(session_options.keys()),
            label_visibility="collapsed",
        )

        if st.button("선택한 대화 불러오기"):
            selected_session_id = session_options[selected_label]
            st.session_state.session_id = selected_session_id
            st.session_state.messages = load_chat_history(selected_session_id)
            st.rerun()

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
if jsonl_path:
    try:
        retriever = get_retriever(
            data_signature,
            str(jsonl_path),
        )
    except Exception as error:
        st.error(f"지식베이스를 준비하는 중 오류가 발생했어: {error}")

if not jsonl_path:
    st.error(
        "지식베이스 파일을 찾지 못했어. "
        "`gsm_guide_rag_chunks.jsonl` 파일이 프로젝트 안에 포함되어 있는지 확인해줘."
    )

if jsonl_path and not retriever:
    st.error(
        "지식베이스 파일은 찾았지만 임베딩이나 벡터 저장소를 준비하지 못했어. "
        "배포 로그를 확인해서 구체적인 오류를 봐줘."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.6,
    streaming=True,
)

system_prompt = """당신은 광주소프트웨어마이스터고(GSM)를 너무나 사랑하는 '유쾌하고 따뜻한 3학년 선배'입니다.
후배가 질문을 하면, 아래 [참고 정보]를 꼼꼼히 읽은 뒤 **완벽하게 소화해서 선배의 언어로 재구성해** 대답해 주세요.

[절대 지켜야 할 답변 규칙] 🚨
1. 기계적인 나열 절대 금지: "정보에 따르면~", "다음과 같습니다.", "1번, 2번" 처럼 딱딱하게 번호를 매기며 로봇처럼 읽어주지 마세요.
2. 자연스러운 스토리텔링: [참고 정보]의 문장들을 그대로 복사+붙여넣기 하지 마세요. 여러 선배들의 꿀팁을 하나로 자연스럽게 엮어서 "내가 경험해 보니까 이렇더라~" 하는 식으로 썰을 풀듯이 말해주세요.
3. 완벽한 구어체 사용: "안녕 후배야!", "이건 진짜 꿀팁인데~", "다들 화이팅하자!" 처럼 친한 동네 형/누나/언니/오빠 같은 말투(반말과 해요체를 섞어서)를 사용하세요.
4. 개인정보 차단: 이름, 이메일 등은 절대 언급하지 마세요.
5. 모르는 질문 대처: 정보에 없는 내용을 물어보면 지어내지 말고 "앗, 그건 나도 잘 모르겠어! 다른 선배나 선생님께 여쭤보는 게 좋겠다ㅎㅎ"라고 쿨하게 넘기세요.
6. **핵심 요약**: 답변 시작 부분에 한 줄로 핵심 요약을 해주세요.
7. **가독성 강조**: 중요한 단어나 문구는 **굵게(Bold)** 표시하세요.
8. **적절한 줄바꿈**: 문장이 너무 길어지지 않게 엔터(줄바꿈)를 자주 사용하세요.
9. **이모지 활용**: 친근감을 위해 문장 끝에 적절한 이모지를 사용하세요.
10. **구조화**: 내용이 많으면 '첫째, 둘째' 또는 '먼저, 그 다음은' 등의 표현을 써서 흐름을 만드세요.
11. 관련성 : 질문에 관련 있는 내용만 답하고 나머지 팁은 추가 적인 팁으로 표현하거나 없애세요.
12. **질문 우선순위**: 후배가 '백엔드', '공부법' 등 특정 주제를 물어보면 [참고 정보]에서 **그 주제와 직접 관련된 내용**을 최우선으로 찾아 답변하세요. 
13. **불필요한 조언 금지**: 질문과 상관없는 '인간관계', '선배와 친해지기' 등의 일반적인 조언은 [참고 정보]에 해당 내용이 메인이 아니라면 언급하지 마세요.
14. **구체적 수치/방법**: 데이터에 공부 사이트, 언어, 프레임워크 등이 있다면 생략하지 말고 정확하게 전달하세요.

[참고 정보]:
{context}"""

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
            <span>안녕 후배야! 궁금한 게 있으면 편하게 물어봐. 준비된 상담 데이터와 학교 자료를 바탕으로 차근차근 답해줄게.</span>
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
            response = "지금은 답변에 사용할 상담 데이터가 준비되지 않았어. 잠시 후 다시 시도해줘."
            st.write(response)
        else:
            try:
                response = st.write_stream(rag_chain.stream(user_input))
            except Exception as error:
                response = f"답변을 생성하는 중 오류가 발생했어: {error}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
    persist_chat_history(st.session_state.session_id, st.session_state.messages)
