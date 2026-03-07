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
MODEL = "gemini-2.5-flash"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ───────────────────────────────────────
# AI가 반드시 최우선으로 지켜야 할 핵심 원칙
# ───────────────────────────────────────
CORE_PRINCIPLES = """
╔══════════════════════════════════════════════════════════════╗
║       AI 수행평가 코치 — 모든 응답 전 반드시 준수할 핵심 원칙        ║
╚══════════════════════════════════════════════════════════════╝

[원칙 1] 이전 주제의 심화·확장 주제를 최우선으로 선정한다.
  - 학생이 이전에 탐구한 주제가 있다면, 그것을 단순 반복하지 않고
    한 단계 더 깊이 들어가거나 반대 관점·한계·문제점을 탐구하는 방향으로 연결한다.
  - 예) 이전 주제: "AI 발전이 신약 개발의 속도와 정확도를 높인다"
         → 다음 주제: "AI의 블랙박스 특성이 신약 개발 임상시험의 신뢰성에 미치는 한계와 문제"
  - 이전 주제가 없을 경우 이 원칙은 건너뛴다.

[원칙 2] 추후 더 깊은 탐구로 이어질 수 있는 주제를 최우선으로 선정한다.
  - 선정한 주제는 반드시 아래 서사 구조가 자연스럽게 성립해야 한다:
    "~에 흥미를 느껴 ~을 탐구했다.
     이 주제는 ~와 연결되어 있으며,
     이후 ~한 활동(실험·논문 읽기·프로젝트 등)으로 심화할 계획이다."
  - 주제 추천 시 각 주제마다 위 서사 구조 예시를 1~2문장으로 함께 제시한다.

[원칙 3] 표절 방지를 철저히 안내한다.
  - 인터넷·논문·책에서 그대로 가져온 문장이 의심될 경우 반드시 경고하고,
    학생 자신의 언어로 재작성하도록 안내한다.
  - 자료 추천 시 "참고용"임을 명시하고, 직접 인용이 아닌 자신의 언어로 활용하도록 강조한다.

[원칙 4] 평가 기준의 모든 항목을 반드시 충족하도록 안내한다.
  - 수행평가 안내문에 명시된 평가 항목과 배점을 항상 기준으로 삼는다.
  - 주제 추천·자료 추천·완성본 평가 모든 단계에서
    "이 항목은 평가 기준 중 [○○] 항목과 연결됩니다"를 명시한다.
  - 빠진 평가 항목이 있으면 반드시 지적한다.

[원칙 5] 학년별 탐구 방향을 반드시 고려하여 주제를 설계한다.
  - 1학년: 해당 과목 내용에 대한 심화 탐구 + 다양한 진로 탐색 가능성 열기
           → 넓고 흥미로운 주제, 진로를 탐색하는 과정이 담길 수 있는 방향
  - 2학년: 희망 진로와 직접 연계된 주제
           → "이 과목 내용이 내 진로에 어떻게 연결되는가"가 명확히 드러나는 방향
  - 3학년: 희망 학과·전공과 연계, 구체적 세부 분야 탐구
           → 대학 전공 수업·연구와 자연스럽게 이어지는 심층적·구체적 주제

[원칙 6] 불확실한 정보는 절대 제공하지 않는다.
  - 사실 여부가 불확실한 수치·사례·연구 결과는 제시하지 않는다.
  - 확실하지 않으면 "확인이 필요합니다" 또는 "공식 자료를 직접 확인하세요"라고 안내한다.
""".strip()


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
# 세션
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
    previous_topic: Optional[str] = None   # ★ 이전에 했던 주제
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
    try:
        res = supabase.table("students").select("*").eq("code", code.upper()).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"학생 조회 오류: {e}")
        return None

def db_check_call_limit(code: str) -> tuple[bool, int, int]:
    """호출 가능 여부 반환. (가능여부, 현재횟수, 최대횟수)"""
    try:
        student = db_get_student(code)
        if not student:
            return False, 0, 0
        limit = student.get("call_limit") or 0
        count = student.get("call_count") or 0
        if limit == 0:  # 무제한
            return True, count, limit
        return count < limit, count, limit
    except Exception as e:
        print(f"호출 제한 확인 오류: {e}")
        return True, 0, 0  # 오류 시 통과

def db_increment_call_count(code: str):
    """호출 횟수 1 증가"""
    try:
        student = db_get_student(code)
        if not student:
            return
        count = (student.get("call_count") or 0) + 1
        supabase.table("students").update({"call_count": count}).eq("code", code.upper()).execute()
    except Exception as e:
        print(f"호출 횟수 업데이트 오류: {e}")

def db_get_conversation(code: str) -> Optional[dict]:
    try:
        res = supabase.table("conversations").select("*").eq("student_code", code.upper()).order("updated_at", desc=True).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"대화 조회 오류: {e}")
        return None

def db_save_conversation(session: StudentSession):
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
        existing = db_get_conversation(session.student_code)
        if existing:
            supabase.table("conversations").update(data).eq("student_code", session.student_code.upper()).execute()
        else:
            supabase.table("conversations").insert(data).execute()
    except Exception as e:
        print(f"대화 저장 오류: {e}")


# ───────────────────────────────────────
# Gemini 호출 (호출 제한 포함)
# ───────────────────────────────────────
def call_text(system: str, user_msg: str, history: list = None, student_code: str = None) -> str:
    if student_code:
        allowed, count, limit = db_check_call_limit(student_code)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"이용 횟수를 모두 사용했어요. (사용: {count}/{limit}회)")
        db_increment_call_count(student_code)
    model = genai.GenerativeModel(model_name=MODEL, system_instruction=system)
    chat = model.start_chat(history=history or [])
    return chat.send_message(user_msg).text

def call_vision(system: str, image_bytes: bytes, mime_type: str, prompt: str, student_code: str = None) -> str:
    if student_code:
        allowed, count, limit = db_check_call_limit(student_code)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"이용 횟수를 모두 사용했어요. (사용: {count}/{limit}회)")
        db_increment_call_count(student_code)
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

    student = db_get_student(code)
    if not student:
        raise HTTPException(status_code=401, detail="등록되지 않은 코드예요. 선생님께 문의하세요.")
    if student["name"].strip() != name:
        raise HTTPException(status_code=401, detail="이름이 일치하지 않아요. 다시 확인해주세요.")

    session_id = str(uuid.uuid4())
    session = get_or_create_session(session_id)
    session.student_code = code
    session.student_name = name

    prev = db_get_conversation(code)
    if prev:
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
        "call_limit": student.get("call_limit") or 0,
        "call_count": student.get("call_count") or 0,
        "message": f"안녕하세요, {name}님! 👋"
    }


# ───────────────────────────────────────
# 세션 관리
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
        "session_id":     session_id,
        "step":           s.step,
        "student_name":   s.student_name,
        "student_code":   s.student_code,
        "has_assessment": s.assessment_info is not None,
        "grade":          s.grade,
        "subject":        s.subject,
        "desired_career": s.desired_career,
        "previous_topic": s.previous_topic,
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

    image_bytes = await image.read()
    ext = image.filename.split(".")[-1].lower()
    mime_type = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","webp":"image/webp"}.get(ext,"image/jpeg")

    system = f"""
{CORE_PRINCIPLES}

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
불확실한 정보는 절대 임의로 추가하지 마세요 (원칙 6).
""".strip()

    result = call_vision(system, image_bytes, mime_type, "수행평가 안내문의 모든 정보를 추출해주세요.", student_code=session.student_code)

    # 여러 장 업로드 시 결과를 누적
    session.assessment_info = (session.assessment_info or "") + "\n\n" + result
    session.assessment_info = session.assessment_info.strip()
    session.step = "assessed"
    add_history(session, "user", "수행평가 안내문 분석 요청")
    add_history(session, "model", result)
    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}
    return {"status": "success", "assessment_info": result, "call_count": student.get("call_count", 0) if student else 0, "call_limit": student.get("call_limit", 0) if student else 0}


# ───────────────────────────────────────
# 2번: 주제 추천
# ───────────────────────────────────────
class StudentInfoRequest(BaseModel):
    session_id: str
    grade: str
    subject: str
    desired_career: str
    assessment_info: Optional[str] = None
    previous_topic: Optional[str] = None   # ★ 이전에 했던 주제

@app.post("/recommend-topics")
def recommend_topics(req: StudentInfoRequest):
    session = get_or_create_session(req.session_id)
    session.grade          = req.grade
    session.subject        = req.subject
    session.desired_career = req.desired_career
    if req.previous_topic:
        session.previous_topic = req.previous_topic
    if req.assessment_info:
        session.assessment_info = req.assessment_info
    _sessions[req.session_id] = session

    assessment_text  = session.assessment_info or "수행평가 안내문 정보 없음"
    previous_topic   = session.previous_topic or req.previous_topic or "없음"

    # 학년별 방향 힌트 생성
    grade_guidance_map = {
        "고1": "1학년이므로 과목 내용에 대한 심화 탐구와 다양한 진로 탐색 가능성을 열어주는 넓고 흥미로운 주제를 선정한다.",
        "고2": "2학년이므로 희망 진로와 직접 연계되어 '이 과목 내용이 내 진로에 어떻게 연결되는가'가 명확히 드러나는 주제를 선정한다.",
        "고3": "3학년이므로 희망 학과·전공과 연계하여 대학 전공 수업·연구와 자연스럽게 이어지는 심층적이고 구체적인 주제를 선정한다.",
    }
    grade_guidance = grade_guidance_map.get(req.grade, "학년에 맞는 탐구 깊이를 고려하여 주제를 선정한다.")

    system = f"""
{CORE_PRINCIPLES}

당신은 고등학교 수행평가 주제 추천 전문가입니다.
아래 지식 데이터를 참고해서 핵심 원칙을 철저히 지키며 학생 맞춤 주제를 추천하세요.

[지식 데이터]
{KNOWLEDGE_BASE}

[학년별 추가 지침]
{grade_guidance}

출력 규칙 (반드시 준수):
1. *, **, ##, ### 같은 마크다운 기호를 절대 사용하지 않는다.
2. 영어 단어나 영어 표현을 괄호 안에 넣지 않는다. 순수 한국어로만 작성한다.
3. [원칙 N:], [원칙] 같은 원칙 번호나 내부 지침 내용을 절대 출력하지 않는다.
4. 주제명은 반드시 수행평가에서 실제로 탐구할 구체적인 한국어 주제명으로 작성한다. '무작위 질문 대비', 'AI 추천 주제' 같은 표현을 절대 쓰지 않는다.
5. 항목은 숫자 번호(1. 2. 3.)로 구분한다.
6. 말투는 자연스러운 한국어 문장으로 쓴다.

반드시 아래 형식으로 3개 추천:

추천 1: (구체적인 한국어 주제명)
1. 핵심 내용: (이 주제에서 탐구할 내용을 2-3문장으로 설명)
2. 이전 주제와의 연결: (이전 주제가 있다면 어떻게 심화되는지. 없으면 이 항목 생략)
3. 추후 심화 방향: (~에 흥미를 느껴 ~을 탐구했다. 이 주제는 ~와 연결되며, 이후 ~한 활동으로 심화할 계획이다. 형식으로 한국어로만 작성)
4. 추천 이유: (평가 기준 연계와 진로 연계를 자연스러운 문장으로 설명. 원칙 번호 언급 금지)
5. 점수 강점: (어떤 평가 항목에서 높은 점수를 받을 수 있는지)

추천 2: (구체적인 한국어 주제명)
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 추후 심화 방향:
4. 추천 이유:
5. 점수 강점:

추천 3: (구체적인 한국어 주제명)
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 추후 심화 방향:
4. 추천 이유:
5. 점수 강점:
""".strip()

    user_msg = f"""
[학생 정보]
- 학년: {req.grade}
- 과목: {req.subject}
- 희망 진로: {req.desired_career}
- 같은 과목에서 이전에 한 주제: {previous_topic}

[수행평가 안내문]
{assessment_text}

핵심 원칙을 최우선으로 지키면서 맞춤 주제 3개를 추천해주세요.
특히 이전 주제({previous_topic})가 있다면 그것을 심화·확장하는 방향을 최우선으로 고려하세요.
"""

    result = call_text(system, user_msg, session.history, student_code=session.student_code)
    session.recommended_topics = result
    add_history(session, "user", user_msg)
    add_history(session, "model", result)
    session.step = "topic_recommended"
    _sessions[req.session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}
    return {"status": "success", "topics": result, "call_count": student.get("call_count", 0) if student else 0, "call_limit": student.get("call_limit", 0) if student else 0}


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
{CORE_PRINCIPLES}

당신은 수행평가 자료 추천 전문가입니다.
아래 지식 데이터에서 주제에 맞는 자료를 찾아 추천하세요.

[지식 데이터]
{KNOWLEDGE_BASE}

출력 규칙 (반드시 준수):
1. *, **, ##, ### 같은 마크다운 기호를 절대 사용하지 않는다.
2. 영어 단어를 괄호 안에 넣지 않는다. 순수 한국어로만 작성한다.
3. 원칙 번호나 내부 지침 내용을 절대 출력하지 않는다.
4. 항목은 숫자 번호로 구분한다.

반드시 아래 형식으로 최대 3개 추천:

자료 1
1. 제목:
2. 유형: (도서/영상/논문/사이트)
3. 핵심 내용: (2-3문장)
4. 수행평가 활용법: (구체적으로 어떤 평가 항목에 연결되는지 포함)
5. 추후 심화 연결: (이 자료를 발판으로 어떤 심화 탐구가 가능한지)
6. 진로 연계:
⚠️ 주의: 이 자료의 내용을 그대로 옮기지 말고, 반드시 본인의 언어로 재구성하세요.

자료 2
1. 제목:
2. 유형:
3. 핵심 내용:
4. 수행평가 활용법:
5. 추후 심화 연결:
6. 진로 연계:
⚠️ 주의: 이 자료의 내용을 그대로 옮기지 말고, 반드시 본인의 언어로 재구성하세요.

자료 3
1. 제목:
2. 유형:
3. 핵심 내용:
4. 수행평가 활용법:
5. 추후 심화 연결:
6. 진로 연계:
⚠️ 주의: 이 자료의 내용을 그대로 옮기지 말고, 반드시 본인의 언어로 재구성하세요.
""".strip()

    user_msg = f"""
선택 주제: {req.selected_topic}
희망 진로: {session.desired_career or '미입력'}
과목: {session.subject or '미입력'}
학년: {session.grade or '미입력'}
이전에 했던 주제: {session.previous_topic or '없음'}

이 주제로 수행평가 작성할 때 활용할 수 있는 자료를 추천해주세요.
추후 심화 탐구로 이어질 수 있는 자료를 우선 추천해주세요.
"""

    result = call_text(system, user_msg, session.history, student_code=session.student_code)
    session.recommended_resources = result
    add_history(session, "user", user_msg)
    add_history(session, "model", result)
    _sessions[req.session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}
    return {"status": "success", "resources": result, "call_count": student.get("call_count", 0) if student else 0, "call_limit": student.get("call_limit", 0) if student else 0}


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
{CORE_PRINCIPLES}

당신은 고등학교 수행평가 채점 전문가입니다.
아래 지식 데이터를 참고해서 핵심 원칙을 철저히 지키며 학생 제출물을 평가하세요.

[지식 데이터]
{KNOWLEDGE_BASE}

출력 규칙 (반드시 준수):
1. *, **, ##, ### 같은 마크다운 기호를 절대 사용하지 않는다.
2. 영어 단어를 괄호 안에 넣지 않는다. 순수 한국어로 작성한다.
3. 원칙 번호나 내부 지침 내용을 절대 출력하지 않는다.
4. 항목은 숫자 번호로 구분한다.

평가 항목마다 아래 형식으로 작성:

항목명 (별점 ★로 표시)
1. 잘한 점: (구체적 근거)
2. 아쉬운 점: (부족한 부분)
3. 보완할 점: (구체적 행동 제시)

추가 체크

표절 위험 문장
(의심 문장이 있으면 명시, 없으면 특이사항 없음)

심화 탐구 연결성
(이전 주제와의 연결과 추후 탐구 가능성이 잘 드러나는지 평가)

종합 평가
예상 점수: X점 / 100점
총평: (2-3문장)
""".strip()

    user_msg = f"""
평가 기준:
{assessment_text}

선택 주제: {session.selected_topic or '미입력'}
희망 진로: {session.desired_career or '미입력'}
학년: {session.grade or '미입력'}
이전에 했던 주제: {session.previous_topic or '없음'}

학생 제출물:
{submission_text}

핵심 원칙을 최우선으로 지키면서 항목별로 평가해주세요.
"""

    result = call_text(system, user_msg, session.history, student_code=session.student_code)
    session.evaluation_result = result
    add_history(session, "user", user_msg)
    add_history(session, "model", result)
    session.step = "evaluated"
    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}
    return {"status": "success", "evaluation": result, "call_count": student.get("call_count", 0) if student else 0, "call_limit": student.get("call_limit", 0) if student else 0}


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
{CORE_PRINCIPLES}

당신은 고등학교 수행평가 채점 전문가입니다.

[평가 기준]
{assessment_text}

[선택 주제] {session.selected_topic or '미입력'}
[희망 진로] {session.desired_career or '미입력'}
[학년] {session.grade or '미입력'}
[이전에 했던 주제] {session.previous_topic or '없음'}

평가 시 반드시 확인:
- 평가 기준 모든 항목 포함 여부 (원칙 4)
- 표절 의심 문장 경고 (원칙 3)
- 이전 주제와의 심화·연결 (원칙 1)
- 추후 탐구로 이어지는 마무리 (원칙 2)
- 학년에 맞는 탐구 수준 (원칙 5)

각 항목마다:
## [항목명] ★★★★☆
✅ 잘한 점:
⚠️ 아쉬운 점:
📌 보완할 점:

## 🔍 표절 위험 문장
## 🔗 심화 탐구 연결성

## 종합 평가
예상 점수: X점 / 100점
총평:
""".strip()

    result = call_vision(system, image_bytes, mime_type, "이 이미지가 학생 수행평가 제출물입니다. 핵심 원칙을 최우선으로 지키며 평가해주세요.", student_code=session.student_code)
    session.evaluation_result = result
    session.step = "evaluated"
    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}
    return {"status": "success", "evaluation": result, "call_count": student.get("call_count", 0) if student else 0, "call_limit": student.get("call_limit", 0) if student else 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
