import os
import uuid
import glob
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import google.generativeai as genai
from supabase import create_client, Client
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ───────────────────────────────────────
# 설정 (Render 환경변수)
# ───────────────────────────────────────
GEMINI_API_KEY  = os.environ.get("GEMINI_API_KEY")
SUPABASE_URL    = os.environ.get("SUPABASE_URL")
SUPABASE_KEY    = os.environ.get("SUPABASE_KEY")

genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash-preview-05-20"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ───────────────────────────────────────
# data/ 폴더 읽기
# ───────────────────────────────────────
def load_knowledge_base() -> str:
    knowledge = ""
    data_dir = "data"
    if not os.path.exists(data_dir):
        return "지식 데이터 없음"
    for filepath in sorted(glob.glob(f"{data_dir}/*.txt")):
        filename = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        knowledge += f"\n\n=== {filename} ===\n{content}"
    return knowledge.strip() or "지식 데이터 없음"

KNOWLEDGE_BASE = load_knowledge_base()
print(f"✅ 지식 데이터 로드 완료 ({len(KNOWLEDGE_BASE)} 글자)")

# ───────────────────────────────────────
# 세션 (서버 메모리 - 요청간 임시 저장)
# ───────────────────────────────────────
@dataclass
class StudentSession:
    session_id: str
    student_code: Optional[str] = None
    student_name: Optional[str] = None
    assessment_info: Optional[str] = None
    grade: Optional[str] = None
    subject: Optional[str] = None
    desired_career: Optional[str] = None
    step: str = "init"
    selected_topic: Optional[str] = None
    recommended_topics: Optional[str] = None
    recommended_resources: Optional[str] = None
    evaluation_result: Optional[str] = None
    history: list = field(default_factory=list)

_sessions: dict = {}

def get_or_create_session(session_id: str) -> StudentSession:
    if session_id not in _sessions:
        _sessions[session_id] = StudentSession(session_id=session_id)
    return _sessions[session_id]

def add_history(session: StudentSession, role: str, text: str):
    session.history.append({"role": role, "parts": [text]})
    if len(session.history) > 20:
        session.history = session.history[-20:]

# ───────────────────────────────────────
# Supabase 헬퍼
# ───────────────────────────────────────
def db_get_student(code: str) -> Optional[dict]:
    """students 테이블에서 코드로 학생 조회"""
    try:
        res = supabase.table("students").select("*").eq("code", code.upper()).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"학생 조회 오류: {e}")
        return None

def db_get_conversation(code: str) -> Optional[dict]:
    """conversations 테이블에서 학생의 마지막 대화 조회"""
    try:
        res = supabase.table("conversations").select("*").eq("student_code", code.upper()).order("updated_at", desc=True).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"대화 조회 오류: {e}")
        return None

def db_save_conversation(session: StudentSession):
    """conversations 테이블에 upsert (있으면 업데이트, 없으면 삽입)"""
    try:
        now = datetime.utcnow().isoformat()
        data = {
            "student_code":    session.student_code,
            "student_name":    session.student_name,
            "subject":         session.subject or "",
            "grade":           session.grade or "",
            "career":          session.desired_career or "",
            "assessment_info": session.assessment_info or "",
            "selected_topic":  session.selected_topic or "",
            "topics":          session.recommended_topics or "",
            "resources":       session.recommended_resources or "",
            "evaluation":      session.evaluation_result or "",
            "updated_at":      now,
        }
        # 기존 행 있으면 업데이트, 없으면 삽입
        existing = db_get_conversation(session.student_code)
        if existing:
            supabase.table("conversations").update(data).eq("student_code", session.student_code.upper()).execute()
        else:
            supabase.table("conversations").insert(data).execute()
    except Exception as e:
        print(f"대화 저장 오류: {e}")

# ───────────────────────────────────────
# Gemini 호출
# ───────────────────────────────────────
def call_text(system: str, user_msg: str, history: list = None) -> str:
    model = genai.GenerativeModel(model_name=MODEL, system_instruction=system)
    chat = model.start_chat(history=history or [])
    return chat.send_message(user_msg).text

def call_vision(system: str, image_bytes: bytes, mime_type: str, prompt: str) -> str:
    model = genai.GenerativeModel(model_name=MODEL, system_instruction=system)
    return model.generate_content([
        {"mime_type": mime_type, "data": image_bytes},
        prompt,
    ]).text

# ───────────────────────────────────────
# FastAPI
# ───────────────────────────────────────
app = FastAPI(title="수행평가 AI 코치")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "수행평가 AI 코치 (Gemini 2.5 Flash + Supabase) 실행 중!"}

# ───────────────────────────────────────
# 로그인
# ───────────────────────────────────────
class LoginRequest(BaseModel):
    code: str
    name: str

@app.post("/login")
def login(req: LoginRequest):
    code = req.code.strip().upper()
    name = req.name.strip()

    # students 테이블에서 확인
    student = db_get_student(code)
    if not student:
        raise HTTPException(status_code=401, detail="등록되지 않은 코드예요. 선생님께 문의하세요.")
    if student["name"].strip() != name:
        raise HTTPException(status_code=401, detail="이름이 일치하지 않아요. 다시 확인해주세요.")

    # 세션 생성
    session_id = str(uuid.uuid4())
    session = get_or_create_session(session_id)
    session.student_code = code
    session.student_name = name

    # 이전 대화 불러오기
    prev = db_get_conversation(code)
    if prev:
        # 세션에 이전 상태 복원
        session.subject           = prev.get("subject", "")
        session.grade             = prev.get("grade", "")
        session.desired_career    = prev.get("career", "")
        session.assessment_info   = prev.get("assessment_info", "")
        session.selected_topic    = prev.get("selected_topic", "")
        session.recommended_topics    = prev.get("topics", "")
        session.recommended_resources = prev.get("resources", "")
        session.evaluation_result = prev.get("evaluation", "")

    _sessions[session_id] = session

    return {
        "status": "success",
        "session_id": session_id,
        "name": name,
        "code": code,
        "previous": prev,
        "message": f"안녕하세요, {name}님! 👋"
    }

# ───────────────────────────────────────
# 세션 상태
# ───────────────────────────────────────
@app.post("/session/create")
def create_session():
    session_id = str(uuid.uuid4())
    get_or_create_session(session_id)
    return {"session_id": session_id}

@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"message": "세션 삭제 완료"}

@app.get("/session/{session_id}/status")
def get_session_status(session_id: str):
    s = get_or_create_session(session_id)
    return {
        "session_id":   session_id,
        "step":         s.step,
        "student_name": s.student_name,
        "student_code": s.student_code,
        "has_assessment": s.assessment_info is not None,
        "grade":        s.grade,
        "subject":      s.subject,
        "desired_career": s.desired_career,
        "selected_topic": s.selected_topic,
    }

# ───────────────────────────────────────
# 1번: 안내문 분석
# ───────────────────────────────────────
@app.post("/analyze-assessment")
async def analyze_assessment(
    session_id: str = Form(...),
    image: UploadFile = File(...),
):
    session = get_or_create_session(session_id)
    if session.assessment_info:
        return {"status": "cached", "assessment_info": session.assessment_info}

    image_bytes = await image.read()
    ext = image.filename.split(".")[-1].lower()
    mime_type = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")

    system = """
당신은 고등학교 수행평가 안내문 분석 전문가입니다.
반드시 아래 형식으로 정확히 추출하세요:

[수행평가 기본 정보]
- 교과/과목:
- 학년:
- 수행평가 유형:
- 주제/제목:
- 제출 형식:
- 제출 기한:

[평가 기준 및 배점]
(항목별 배점 정리)

[세부 요구사항]
- 필수 포함 내용:
- 특이사항:

확인 안 되는 항목은 '정보 없음'으로 표시하세요.
""".strip()

    result = call_vision(system, image_bytes, mime_type, "수행평가 안내문의 모든 정보를 추출해주세요.")
    session.assessment_info = result
    session.step = "assessed"
    add_history(session, "user", "수행평가 안내문 분석 요청")
    add_history(session, "model", result)
    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    return {"status": "success", "assessment_info": result}

# ───────────────────────────────────────
# 2번: 주제 추천
# ───────────────────────────────────────
class StudentInfoRequest(BaseModel):
    session_id: str
    grade: str
    subject: str
    desired_career: str
    assessment_info: Optional[str] = None

@app.post("/recommend-topics")
def recommend_topics(req: StudentInfoRequest):
    session = get_or_create_session(req.session_id)
    session.grade          = req.grade
    session.subject        = req.subject
    session.desired_career = req.desired_career
    if req.assessment_info:
        session.assessment_info = req.assessment_info
    _sessions[req.session_id] = session

    assessment_text = session.assessment_info or "수행평가 안내문 정보 없음"

    system = f"""
당신은 고등학교 수행평가 주제 추천 전문가입니다.
아래 지식 데이터를 참고해서 학생 맞춤 주제를 추천하세요.

[지식 데이터]
{KNOWLEDGE_BASE}

반드시 아래 형식으로 3개 추천:

추천 1: [구체적인 주제명]
- 핵심 내용: (2-3문장)
- 추천 이유: (평가기준 연계 + 진로 연계)
- 점수 강점: (어떤 항목에서 높은 점수 받을 수 있는지)

추천 2: [구체적인 주제명]
- 핵심 내용:
- 추천 이유:
- 점수 강점:

추천 3: [구체적인 주제명]
- 핵심 내용:
- 추천 이유:
- 점수 강점:
""".strip()

    user_msg = f"""
[학생 정보]
- 학년: {req.grade}
- 과목: {req.subject}
- 희망 진로: {req.desired_career}

[수행평가 안내문]
{assessment_text}

맞춤 주제 3개를 추천해주세요.
"""

    result = call_text(system, user_msg, session.history)
    session.recommended_topics = result
    add_history(session, "user", user_msg)
    add_history(session, "model", result)
    session.step = "topic_recommended"
    _sessions[req.session_id] = session

    if session.student_code:
        db_save_conversation(session)

    return {"status": "success", "topics": result}

# ───────────────────────────────────────
# 3번: 자료 추천
# ───────────────────────────────────────
class ResourceRequest(BaseModel):
    session_id: str
    selected_topic: str

@app.post("/find-resources")
def find_resources(req: ResourceRequest):
    session = get_or_create_session(req.session_id)
    session.selected_topic = req.selected_topic
    _sessions[req.session_id] = session

    system = f"""
당신은 수행평가 자료 추천 전문가입니다.
아래 지식 데이터에서 주제에 맞는 자료를 찾아 추천하세요.

[지식 데이터]
{KNOWLEDGE_BASE}

반드시 아래 형식으로 최대 3개 추천:

[자료 1]
- 제목:
- 유형: (도서/영상/논문/사이트)
- 핵심 내용: (2-3문장)
- 수행평가 활용법: (구체적으로)
- 진로 연계:

[자료 2]
- 제목:
- 유형:
- 핵심 내용:
- 수행평가 활용법:
- 진로 연계:

[자료 3]
- 제목:
- 유형:
- 핵심 내용:
- 수행평가 활용법:
- 진로 연계:
""".strip()

    user_msg = f"""
[선택 주제] {req.selected_topic}
[희망 진로] {session.desired_career or '미입력'}
[과목] {session.subject or '미입력'}

이 주제로 수행평가 작성할 때 활용할 수 있는 자료를 추천해주세요.
"""

    result = call_text(system, user_msg, session.history)
    session.recommended_resources = result
    add_history(session, "user", user_msg)
    add_history(session, "model", result)
    _sessions[req.session_id] = session

    if session.student_code:
        db_save_conversation(session)

    return {"status": "success", "resources": result}

# ───────────────────────────────────────
# 4번: 완성본 평가 (텍스트)
# ───────────────────────────────────────
@app.post("/evaluate-text")
def evaluate_text(
    session_id: str = Form(...),
    submission_text: str = Form(...),
):
    session = get_or_create_session(session_id)
    assessment_text = session.assessment_info or "평가 기준 정보 없음"

    system = f"""
당신은 고등학교 수행평가 채점 전문가입니다.
아래 지식 데이터를 참고해서 학생 제출물을 평가하세요.

[지식 데이터]
{KNOWLEDGE_BASE}

각 평가 항목마다:
## [항목명] ★★★★☆
✅ 잘한 점: (구체적 근거)
⚠️ 아쉬운 점: (부족한 부분)
📌 보완할 점: (구체적 행동)

마지막에:
## 종합 평가
예상 점수: X점 / 100점
총평: (2-3문장)
""".strip()

    user_msg = f"""
[평가 기준]
{assessment_text}

[선택 주제] {session.selected_topic or '미입력'}
[희망 진로] {session.desired_career or '미입력'}

[학생 제출물]
{submission_text}

항목별로 평가해주세요.
"""

    result = call_text(system, user_msg, session.history)
    session.evaluation_result = result
    add_history(session, "user", user_msg)
    add_history(session, "model", result)
    session.step = "evaluated"
    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    return {"status": "success", "evaluation": result}

# ───────────────────────────────────────
# 4번: 완성본 평가 (이미지)
# ───────────────────────────────────────
@app.post("/evaluate-image")
async def evaluate_image(
    session_id: str = Form(...),
    image: UploadFile = File(...),
):
    session = get_or_create_session(session_id)
    assessment_text = session.assessment_info or "평가 기준 정보 없음"

    image_bytes = await image.read()
    ext = image.filename.split(".")[-1].lower()
    mime_type = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png"}.get(ext,"image/jpeg")

    system = f"""
당신은 고등학교 수행평가 채점 전문가입니다.

[평가 기준]
{assessment_text}

[선택 주제] {session.selected_topic or '미입력'}
[희망 진로] {session.desired_career or '미입력'}

각 항목마다:
## [항목명] ★★★★☆
✅ 잘한 점:
⚠️ 아쉬운 점:
📌 보완할 점:

마지막에:
## 종합 평가
예상 점수: X점 / 100점
총평:
""".strip()

    result = call_vision(system, image_bytes, mime_type, "이 이미지가 학생 수행평가 제출물입니다. 평가해주세요.")
    session.evaluation_result = result
    session.step = "evaluated"
    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    return {"status": "success", "evaluation": result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
