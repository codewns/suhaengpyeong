import os
import uuid
import glob
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from google import genai
from google.genai import types
from supabase import create_client, Client
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ───────────────────────────────────────
# 설정
# ───────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 수행평가 전용 Supabase
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# 기존 홈페이지 Supabase: 위닝 수행 주제 DB / 자료 DB 저장소
WINNING_SUPABASE_URL = os.environ.get("WINNING_SUPABASE_URL")
WINNING_SUPABASE_KEY = os.environ.get("WINNING_SUPABASE_KEY")
WINNING_KNOWLEDGE_TABLE = os.environ.get(
    "WINNING_KNOWLEDGE_TABLE",
    "winning_assessment_knowledge_items"
)

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수가 없습니다.")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("수행평가 Supabase 환경변수가 없습니다.")

client_genai = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

winning_supabase: Optional[Client] = None
if WINNING_SUPABASE_URL and WINNING_SUPABASE_KEY:
    winning_supabase = create_client(WINNING_SUPABASE_URL, WINNING_SUPABASE_KEY)


# ───────────────────────────────────────
# AI 핵심 원칙
# ───────────────────────────────────────

CORE_PRINCIPLES = """
╔══════════════════════════════════════════════════════════════╗
║       AI 수행평가 코치 — 모든 응답 전 반드시 준수할 핵심 원칙        ║
╚══════════════════════════════════════════════════════════════╝

[원칙 1] 이전 주제의 심화·확장 주제를 최우선으로 선정한다.
  - 학생이 이전에 탐구한 주제가 있다면, 그것을 단순 반복하지 않고
    한 단계 더 깊이 들어가거나 반대 관점·한계·문제점을 탐구하는 방향으로 연결한다.
  - 이전 주제가 없을 경우 이 원칙은 건너뛴다.

[원칙 2] 추후 더 깊은 탐구로 이어질 수 있는 주제를 우선한다.
  - 선정한 주제는 "~에 흥미를 느껴 ~을 탐구했고, 이후 ~로 심화했다"는 흐름이 가능해야 한다.

[원칙 3] 표절 방지를 철저히 안내한다.
  - 자료는 참고용이며, 학생 자신의 언어로 재구성하도록 안내한다.
  - 원문 문장이나 선배 사례 문장을 그대로 따라 쓰게 하면 안 된다.

[원칙 4] 평가 기준을 반드시 우선한다.
  - 수행평가 안내문에 명시된 평가 항목과 제출 조건을 최우선 기준으로 삼는다.
  - 안내문 조건과 충돌하는 내부 DB 자료는 사용하지 않는다.

[원칙 5] 학년별 탐구 깊이를 고려한다.
  - 고1: 과목 내용 심화 + 진로 탐색 가능성
  - 고2: 희망 진로와 직접 연계
  - 고3: 희망 학과·전공과 연계된 심층 탐구

[원칙 6] 불확실한 정보는 제공하지 않는다.
  - 사실 여부가 불확실한 수치·자료·링크는 만들지 않는다.
  - 확실하지 않으면 "확인 필요"라고 쓴다.
""".strip()


CROSS_SUBJECT_CONNECTION_GUIDE = """
[다른 과목 선배 데이터 연계 판단 기준]

다른 과목 선배 데이터를 활용할 때는 반드시 아래 기준으로 판단한다.

1. 같은 진로 분야라도 현재 과목의 성격과 맞지 않으면 사용하지 않는다.
2. 다른 과목 데이터를 그대로 반복하지 말고, 현재 과목의 방식으로 재해석할 수 있을 때만 사용한다.
3. 과학 → 영어 연계:
   - 과학 원리 설명을 반복하지 않는다.
   - 영문 기사, 국제기구 자료, 캠페인 문구, 과학 커뮤니케이션, 영어 발표·에세이 구조로 재해석한다.
4. 과학 → 국어 연계:
   - 과학 개념 설명보다 비문학 글의 논증 구조, 표현 전략, 관점 분석으로 바꾼다.
5. 과학 → 사회 연계:
   - 과학 지식을 사회 문제, 정책, 이해관계자, 윤리 쟁점으로 확장한다.
6. 사회 → 국어 연계:
   - 사회 쟁점을 글의 설득 방식, 표현 전략, 담론 구조 분석으로 바꾼다.
7. 국어 → 영어 연계:
   - 국어에서 다룬 주제를 영문 자료 분석, 영어 발표, 영어 에세이로 확장한다.
8. 수학 → 과학 연계:
   - 수학적 모델, 그래프, 통계, 변화량, 확률을 과학 탐구의 분석 도구로 사용한다.
9. 연계가 자연스럽지 않으면 "연계하지 않음"으로 판단하고 현재 과목 주제만 추천한다.
10. 추천 3개 중 최소 1개는 같은 과목 이전 수행이 있으면 같은 과목 심화 주제로 구성한다.
11. 다른 과목 선배 데이터는 선택 사항이며, 억지로 반드시 반영하지 않는다.
""".strip()


# ───────────────────────────────────────
# 단계별 지식 데이터 읽기
# ───────────────────────────────────────

def normalize_subject(subject: str) -> str:
    s = (subject or "").replace(" ", "").strip()

    if "국어" in s or "문학" in s or "독서" in s or "화작" in s or "언매" in s:
        return "국어"
    if "수학" in s or "미적분" in s or "기하" in s or "확률" in s or "통계" in s:
        return "수학"
    if "영어" in s:
        return "영어"
    if "사회" in s or "역사" in s or "윤리" in s or "지리" in s or "경제" in s or "정치" in s:
        return "사회역사"
    if "과학" in s or "생명" in s or "화학" in s or "물리" in s or "지구" in s:
        return "과학"

    return subject.strip() or "국어"


def normalize_grade(grade: str) -> str:
    g = (grade or "").strip()

    if "2" in g:
        return "grade2"
    if "3" in g:
        return "grade3"

    return "grade1"


def load_stage_knowledge(grade: str, subject: str, stage: str) -> str:
    """
    stage:
    - topic: 주제 추천용
    - resources: 자료 추천용
    """

    grade_dir = normalize_grade(grade)
    subject_name = normalize_subject(subject)

    paths = [
        os.path.join("data", grade_dir, stage, f"{subject_name}.txt"),
        os.path.join("data", grade_dir, f"{stage}_{subject_name}.txt"),
        os.path.join("data", grade_dir, f"{subject_name}.txt"),
    ]

    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:
                return content[:12000]

    return "해당 단계의 내부 지식 데이터 없음"


def score_text(text: str, keywords: list[str]) -> int:
    t = (text or "").lower()
    score = 0

    for kw in keywords:
        k = (kw or "").lower().strip()
        if not k:
            continue
        if k in t:
            score += 3

    return score


def unique_list(values: list[str]) -> list[str]:
    seen = set()
    result = []

    for value in values:
        v = (value or "").strip()
        if not v or v in seen:
            continue
        seen.add(v)
        result.append(v)

    return result


def load_winning_dynamic_knowledge(
    grade: str,
    subject: str,
    career: str,
    selected_topic: str = "",
    assessment_info: str = "",
    purpose: str = "topic",
    max_items: int = 6,
    max_chars: int = 4500,
    include_other_subjects: bool = False
) -> str:
    """
    홈페이지 Supabase의 winning_assessment_knowledge_items에서
    위닝 수행 주제 DB / 자료 DB를 읽는다.

    purpose='topic'    → knowledge_type='topic_pattern'
    purpose='resource' → knowledge_type='verified_resource'

    include_other_subjects=True이면
    같은/비슷한 진로의 다른 과목 데이터도 후보로 2~3개 읽어온다.
    단, 최종 반영 여부는 AI가 판단한다.
    """

    if not winning_supabase:
        return "홈페이지 Supabase 위닝DB 연결 없음"

    knowledge_type = "verified_resource" if purpose == "resource" else "topic_pattern"

    normalized_subject = normalize_subject(subject)

    grade_candidates = unique_list([
        grade,
        "공통",
        "전체",
        "고등학생"
    ])

    subject_candidates = unique_list([
        normalized_subject,
        subject,
        "공통",
        "전체"
    ])

    try:
        # 1차: 현재 과목 데이터
        query = (
            winning_supabase
            .table(WINNING_KNOWLEDGE_TABLE)
            .select("id, grade, subject, knowledge_type, career_field, title, content, source, memo, created_at")
            .eq("is_active", True)
            .eq("knowledge_type", knowledge_type)
            .in_("grade", grade_candidates)
            .in_("subject", subject_candidates)
            .order("created_at", desc=True)
            .limit(80)
        )

        res = query.execute()
        rows = res.data or []

        keywords = [
            career or "",
            selected_topic or "",
            assessment_info[:400] if assessment_info else "",
            subject or "",
            normalized_subject or "",
        ]

        scored = []
        for row in rows:
            blob = " ".join([
                row.get("career_field") or "",
                row.get("title") or "",
                row.get("content") or "",
                row.get("source") or "",
                row.get("memo") or "",
            ])
            scored.append((score_text(blob, keywords), row))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [row for score, row in scored[:max_items]]

        pieces = []
        total = 0

        if selected:
            pieces.append("[현재 과목 위닝DB 후보]\n")
            for row in selected:
                piece = f"""
[위닝DB 항목]
- 학년: {row.get('grade') or ''}
- 과목: {row.get('subject') or ''}
- 진로분야: {row.get('career_field') or ''}
- 자료명: {row.get('title') or ''}
- 출처: {row.get('source') or ''}
- 내용:
{row.get('content') or ''}
""".strip()

                if total + len(piece) > max_chars:
                    break

                pieces.append(piece)
                total += len(piece)

        # 2차: 같은/비슷한 진로의 다른 과목 데이터
        if include_other_subjects:
            other_query = (
                winning_supabase
                .table(WINNING_KNOWLEDGE_TABLE)
                .select("id, grade, subject, knowledge_type, career_field, title, content, source, memo, created_at")
                .eq("is_active", True)
                .eq("knowledge_type", knowledge_type)
                .in_("grade", grade_candidates)
                .not_.in_("subject", subject_candidates)
                .order("created_at", desc=True)
                .limit(120)
            )

            other_res = other_query.execute()
            other_rows = other_res.data or []

            career_keywords = [
                career or "",
                selected_topic or "",
            ]

            other_scored = []
            for row in other_rows:
                blob = " ".join([
                    row.get("career_field") or "",
                    row.get("title") or "",
                    row.get("content") or "",
                    row.get("source") or "",
                    row.get("memo") or "",
                ])
                other_scored.append((score_text(blob, career_keywords), row))

            other_scored.sort(key=lambda x: x[0], reverse=True)

            other_selected = []
            used_subjects = set()

            for score, row in other_scored:
                if score <= 0:
                    continue

                row_subject = row.get("subject") or ""
                if len(other_selected) >= 3:
                    break

                # 같은 과목 중복 방지
                if row_subject in used_subjects and len(other_selected) >= 2:
                    continue

                other_selected.append(row)
                used_subjects.add(row_subject)

            if other_selected:
                pieces.append("\n\n[다른 과목 연계 후보 - AI가 연계 가능할 때만 사용]\n")
                for row in other_selected:
                    piece = f"""
[다른 과목 후보]
- 학년: {row.get('grade') or ''}
- 과목: {row.get('subject') or ''}
- 진로분야: {row.get('career_field') or ''}
- 자료명: {row.get('title') or ''}
- 출처: {row.get('source') or ''}
- 내용:
{row.get('content') or ''}
""".strip()

                    if total + len(piece) > max_chars:
                        break

                    pieces.append(piece)
                    total += len(piece)

        return "\n\n".join(pieces).strip() if pieces else "관련 위닝DB 항목 없음"

    except Exception as e:
        print(f"홈페이지 Supabase 위닝DB 조회 오류: {e}")
        return "홈페이지 Supabase 위닝DB 조회 실패"


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
    previous_topic: Optional[str] = None
    step: str = "init"
    selected_topic: Optional[str] = None
    selected_topic_detail: Optional[str] = None
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
    try:
        student = db_get_student(code)
        if not student:
            return False, 0, 0

        limit = student.get("call_limit") or 0
        count = student.get("call_count") or 0

        if limit == 0:
            return True, count, limit

        return count < limit, count, limit

    except Exception as e:
        print(f"호출 제한 확인 오류: {e}")
        return True, 0, 0


def db_increment_call_count(code: str):
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
        res = (
            supabase
            .table("conversations")
            .select("*")
            .eq("student_code", code.upper())
            .order("updated_at", desc=True)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None

    except Exception as e:
        print(f"대화 조회 오류: {e}")
        return None


def db_save_conversation(session: StudentSession):
    try:
        if not session.student_code:
            return

        now = datetime.utcnow().isoformat()

        topic_combined = session.selected_topic or ""
        if session.selected_topic_detail:
            topic_combined = (session.selected_topic or "") + "|||" + session.selected_topic_detail

        data = {
            "student_code": session.student_code,
            "student_name": session.student_name,
            "subject": session.subject or "",
            "grade": session.grade or "",
            "career": session.desired_career or "",
            "assessment_info": session.assessment_info or "",
            "selected_topic": topic_combined,
            "topics": session.recommended_topics or "",
            "resources": session.recommended_resources or "",
            "evaluation": session.evaluation_result or "",
            "updated_at": now,
        }

        existing = db_get_conversation(session.student_code)

        if existing:
            supabase.table("conversations").update(data).eq("student_code", session.student_code.upper()).execute()
        else:
            supabase.table("conversations").insert(data).execute()

    except Exception as e:
        print(f"대화 저장 오류: {e}")


# ───────────────────────────────────────
# 재시도 헬퍼
# ───────────────────────────────────────

def call_with_retry(fn, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return fn()

        except HTTPException:
            raise

        except Exception as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                    continue
            raise e

    raise HTTPException(status_code=503, detail="AI 서버가 일시적으로 혼잡합니다. 잠시 후 다시 시도해주세요.")


# ───────────────────────────────────────
# Gemini 호출
# ───────────────────────────────────────

def call_text(system: str, user_msg: str, history: list = None, student_code: str = None) -> str:
    if student_code:
        allowed, count, limit = db_check_call_limit(student_code)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"이용 횟수를 모두 사용했어요. (사용: {count}/{limit}회)")
        db_increment_call_count(student_code)

    contents = []

    for h in (history or []):
        role = "user" if h.get("role") == "user" else "model"
        parts = h.get("parts", [])
        text = parts[0] if parts and isinstance(parts[0], str) else (
            parts[0].get("text", "") if parts else ""
        )
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

    contents.append(types.Content(role="user", parts=[types.Part(text=user_msg)]))

    def _call():
        return client_genai.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system)
        )

    response = call_with_retry(_call)
    return response.text


def call_vision(system: str, image_bytes: bytes, mime_type: str, prompt: str, student_code: str = None) -> str:
    if student_code:
        allowed, count, limit = db_check_call_limit(student_code)
        if not allowed:
            raise HTTPException(status_code=429, detail=f"이용 횟수를 모두 사용했어요. (사용: {count}/{limit}회)")
        db_increment_call_count(student_code)

    def _call():
        return client_genai.models.generate_content(
            model=MODEL,
            contents=[
                types.Part(inline_data=types.Blob(mime_type=mime_type, data=image_bytes)),
                types.Part(text=prompt)
            ],
            config=types.GenerateContentConfig(system_instruction=system)
        )

    response = call_with_retry(_call)
    return response.text


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
    return {
        "status": "ok",
        "message": "수행평가 AI 코치 실행 중",
        "model": MODEL,
        "winning_db_connected": winning_supabase is not None
    }


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
        session.subject = prev.get("subject", "")
        session.grade = prev.get("grade", "")
        session.desired_career = prev.get("career", "")
        session.assessment_info = prev.get("assessment_info", "")

        topic_raw = prev.get("selected_topic", "")
        if "|||" in topic_raw:
            parts = topic_raw.split("|||", 1)
            session.selected_topic = parts[0]
            session.selected_topic_detail = parts[1]
        else:
            session.selected_topic = topic_raw
            session.selected_topic_detail = ""

        session.recommended_topics = prev.get("topics", "")
        session.recommended_resources = prev.get("resources", "")
        session.evaluation_result = prev.get("evaluation", "")

    _sessions[session_id] = session

    if prev:
        topic_raw = prev.get("selected_topic", "")
        prev = dict(prev)

        if "|||" in topic_raw:
            parts = topic_raw.split("|||", 1)
            prev["selected_topic"] = parts[0]
            prev["selected_topic_detail"] = parts[1]
        else:
            prev["selected_topic_detail"] = ""

    return {
        "status": "success",
        "session_id": session_id,
        "name": name,
        "code": code,
        "previous": prev,
        "call_limit": student.get("call_limit") or 0,
        "call_count": student.get("call_count") or 0,
        "message": f"안녕하세요, {name}님!"
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
        "session_id": session_id,
        "step": s.step,
        "student_name": s.student_name,
        "student_code": s.student_code,
        "has_assessment": s.assessment_info is not None,
        "grade": s.grade,
        "subject": s.subject,
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
    subject: str = Form(default=""),
    career: str = Form(default=""),
):
    session = get_or_create_session(session_id)

    if subject:
        session.subject = subject

    if career:
        session.desired_career = career

    if not session.grade:
        session.grade = "고등학생"

    image_bytes = await image.read()

    ext = image.filename.split(".")[-1].lower()
    mime_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp"
    }.get(ext, "image/jpeg")

    system = f"""
{CORE_PRINCIPLES}

당신은 고등학교 수행평가 안내문 분석 전문가입니다.
이 단계에서는 내부 위닝DB, 선배 사례, 자료 추천 데이터를 절대 사용하지 않습니다.
오직 학생이 업로드한 수행평가 안내문 이미지에 보이는 정보만 분석합니다.

반드시 아래 형식으로 정확히 추출하세요.

[수행평가 기본 정보]
- 교과/과목:
- 학년:
- 수행평가 유형:
- 주제/제목:
- 제출 형식:
- 제출 기한:

[평가 기준 및 배점]
항목별 배점 정리

[세부 요구사항]
- 필수 포함 내용:
- 특이사항:

[질문 목록 - 절대 누락 금지]
안내문에 번호가 매겨진 질문 목록이 있으면 반드시 전부 추출하세요.
질문이 없으면 "질문 목록 없음"으로 표시하세요.

[자료/도서/영상 목록 - 절대 누락 금지]
안내문에 학생이 선택해야 할 자료/도서/영상 목록이 있으면 전부 추출하세요.
없으면 "없음"으로 표시하세요.

확인 안 되는 항목은 "정보 없음"으로 표시하세요.
불확실한 정보는 절대 임의로 추가하지 마세요.
""".strip()

    result = call_vision(
        system,
        image_bytes,
        mime_type,
        "수행평가 안내문의 모든 정보를 추출해주세요.",
        student_code=session.student_code
    )

    session.assessment_info = ((session.assessment_info or "") + "\n\n" + result).strip()
    session.step = "assessed"

    add_history(session, "user", "수행평가 안내문 분석 요청")
    add_history(session, "model", result)

    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}

    return {
        "status": "success",
        "assessment_info": result,
        "call_count": student.get("call_count", 0) if student else 0,
        "call_limit": student.get("call_limit", 0) if student else 0
    }


# ───────────────────────────────────────
# 2번: 주제 추천
# ───────────────────────────────────────

class StudentInfoRequest(BaseModel):
    session_id: str
    grade: str
    subject: str
    desired_career: str
    assessment_info: Optional[str] = None
    previous_topic: Optional[str] = None


@app.post("/recommend-topics")
def recommend_topics(req: StudentInfoRequest):
    session = get_or_create_session(req.session_id)

    session.grade = req.grade
    session.subject = req.subject
    session.desired_career = req.desired_career

    if req.previous_topic:
        session.previous_topic = req.previous_topic

    if req.assessment_info:
        session.assessment_info = req.assessment_info

    _sessions[req.session_id] = session

    assessment_text = session.assessment_info or "수행평가 안내문 정보 없음"
    previous_topic = session.previous_topic or req.previous_topic or "없음"

    topic_knowledge = load_stage_knowledge(req.grade, req.subject, "topic")

    dynamic_topic_knowledge = load_winning_dynamic_knowledge(
        grade=req.grade,
        subject=req.subject,
        career=req.desired_career,
        selected_topic=previous_topic,
        assessment_info=assessment_text,
        purpose="topic",
        max_items=6,
        max_chars=4500,
        include_other_subjects=True
    )

    grade_guidance_map = {
        "고1": "1학년이므로 과목 내용 심화와 다양한 진로 탐색 가능성을 열어주는 주제를 선정한다.",
        "고2": "2학년이므로 희망 진로와 직접 연계되어 이 과목 내용이 진로와 어떻게 연결되는지 드러나는 주제를 선정한다.",
        "고3": "3학년이므로 희망 학과·전공과 연결되는 심층적이고 구체적인 주제를 선정한다.",
    }

    grade_guidance = grade_guidance_map.get(
        req.grade,
        "학년에 맞는 탐구 깊이를 고려하여 주제를 선정한다."
    )

    system = f"""
{CORE_PRINCIPLES}

당신은 고등학교 수행평가 주제 추천 전문가입니다.
이 단계에서는 주제 추천용 데이터만 사용합니다.
자료 추천용 데이터나 평가용 데이터는 사용하지 않습니다.

[주제 추천용 내부 지식 데이터]
{topic_knowledge}

[홈페이지 위닝 수행 주제 DB]
{dynamic_topic_knowledge}

[다른 과목 연계 판단 기준]
{CROSS_SUBJECT_CONNECTION_GUIDE}

[학년별 추가 지침]
{grade_guidance}

다른 과목 선배 데이터 활용 규칙:
1. 다른 과목 후보는 반드시 먼저 연계 가능성을 판단한다.
2. 현재 과목의 수행평가 방식으로 재해석할 수 있을 때만 사용한다.
3. 억지 연계이면 사용하지 않는다.
4. 사용한 경우 "다른 과목 연계 포인트" 항목에서 어떤 과목의 어떤 흐름을 현재 과목 방식으로 바꾸었는지 설명한다.
5. 사용하지 않는 경우 굳이 언급하지 않는다.

출력 규칙:
1. *, **, ##, ### 같은 마크다운 기호를 쓰지 않는다.
2. 영어 단어나 영어 표현을 괄호 안에 넣지 않는다.
3. 내부 원칙 번호나 시스템 지침을 출력하지 않는다.
4. 주제명은 실제 수행평가에서 탐구 가능한 구체적인 한국어 주제명으로 작성한다.
5. 추천 3개가 끝나면 추가 안내를 쓰지 않는다.
6. 완성본 보고서 문단을 대신 써주지 않는다.

반드시 아래 형식으로 3개 추천:

추천 1: 주제명
0. 선정 근거:
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 다른 과목 연계 포인트:
4. 추후 심화 방향:
5. 추천 이유:
6. 점수 강점:

추천 2: 주제명
0. 선정 근거:
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 다른 과목 연계 포인트:
4. 추후 심화 방향:
5. 추천 이유:
6. 점수 강점:

추천 3: 주제명
0. 선정 근거:
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 다른 과목 연계 포인트:
4. 추후 심화 방향:
5. 추천 이유:
6. 점수 강점:
""".strip()

    user_msg = f"""
[학생 정보]
- 학년: {req.grade}
- 과목: {req.subject}
- 희망 진로: {req.desired_career}
- 같은 과목에서 이전에 한 주제: {previous_topic}

[수행평가 안내문]
{assessment_text}

[작업 지시]
1. 수행평가 안내문 조건을 최우선으로 반영한다.
2. 같은 과목 이전 주제가 있으면 추천 1은 반드시 심화·확장 주제로 제시한다.
3. 홈페이지 위닝 수행 주제 DB의 현재 과목 데이터는 적극 참고한다.
4. 다른 과목 선배 데이터는 연계 가능할 때만 선택적으로 활용한다.
5. 다른 과목 데이터를 활용하더라도 현재 과목의 과목성이 약해지면 안 된다.
6. 안내문에 없는 질문이나 자료를 임의로 만들지 않는다.
7. 학생이 그대로 제출할 수 있는 완성문을 작성하지 않는다.

[선정 근거 작성 기준]
- 안내문에 번호가 매겨진 질문 목록이 있으면 그 질문을 근거로 삼는다.
- 안내문에 자료/도서/영상 목록이 있으면 해당 자료를 근거로 삼는다.
- 둘 다 없으면 0번 선정 근거는 "안내문 평가 기준과 학생 진로를 바탕으로 선정" 정도로만 간단히 쓴다.
""".strip()

    result = call_text(system, user_msg, session.history, student_code=session.student_code)

    session.recommended_topics = result
    session.step = "topic_recommended"

    add_history(session, "user", user_msg)
    add_history(session, "model", result)

    _sessions[req.session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}

    return {
        "status": "success",
        "topics": result,
        "call_count": student.get("call_count", 0) if student else 0,
        "call_limit": student.get("call_limit", 0) if student else 0
    }


# ───────────────────────────────────────
# 3번: 자료 추천
# ───────────────────────────────────────

class ResourceRequest(BaseModel):
    session_id: str
    selected_topic: str
    selected_topic_detail: Optional[str] = None


@app.post("/find-resources")
def find_resources(req: ResourceRequest):
    session = get_or_create_session(req.session_id)

    session.selected_topic = req.selected_topic
    session.selected_topic_detail = req.selected_topic_detail or ""

    _sessions[req.session_id] = session

    resource_knowledge = load_stage_knowledge(
        session.grade or "고1",
        session.subject or "국어",
        "resources"
    )

    dynamic_resource_knowledge = load_winning_dynamic_knowledge(
        grade=session.grade or "고1",
        subject=session.subject or "국어",
        career=session.desired_career or "",
        selected_topic=req.selected_topic,
        assessment_info=session.assessment_info or "",
        purpose="resource",
        max_items=8,
        max_chars=6500,
        include_other_subjects=False
    )

    system = f"""
당신은 수행평가 자료 추천 전문가입니다.
이 단계에서는 자료 추천용 데이터만 사용합니다.
주제 추천용 데이터는 참고하지 않습니다.

[자료 추천용 내부 지식 데이터]
{resource_knowledge}

[홈페이지 위닝 수행 자료 DB]
{dynamic_resource_knowledge}

[반드시 지켜야 할 규칙]
1. 위닝 수행 자료 DB에 있는 검증 자료를 우선 사용한다.
2. 선택 주제와 직접 연결되는 자료만 추천한다.
3. 내부 지식 데이터나 위닝DB에 없는 자료명, 저자, 링크를 지어내지 않는다.
4. 자료가 부족하면 자료명 대신 검색 키워드를 제시한다.
5. 링크가 내부 데이터에 없으면 링크 항목을 생략한다.
6. 자료는 최대 3개만 추천한다.
7. 자료 내용은 그대로 베끼지 말고 학생의 언어로 재구성해야 한다고 안내한다.

출력 규칙:
1. *, **, ##, ### 같은 마크다운 기호를 쓰지 않는다.
2. 항목은 숫자 번호로 구분한다.
3. 3개가 끝나면 추가 내용을 쓰지 않는다.

반드시 아래 형식으로 최대 3개 추천:

자료 1
1. 추천 구분: 선배 검증 자료 / 검색 키워드 / 자료 유형 중 하나
2. 제목 또는 검색 키워드:
3. 출처 정보:
4. 유형:
5. 핵심 내용:
6. 현재 주제와의 연결:
7. 수행평가 활용법:
8. 진로 연계:
9. 주의:

자료 2
1. 추천 구분:
2. 제목 또는 검색 키워드:
3. 출처 정보:
4. 유형:
5. 핵심 내용:
6. 현재 주제와의 연결:
7. 수행평가 활용법:
8. 진로 연계:
9. 주의:

자료 3
1. 추천 구분:
2. 제목 또는 검색 키워드:
3. 출처 정보:
4. 유형:
5. 핵심 내용:
6. 현재 주제와의 연결:
7. 수행평가 활용법:
8. 진로 연계:
9. 주의:
""".strip()

    user_msg = f"""
[선택 주제]
{req.selected_topic}

[선택 주제 상세]
{req.selected_topic_detail or '없음'}

[학생 정보]
- 학년: {session.grade or '미입력'}
- 과목: {session.subject or '미입력'}
- 희망 진로: {session.desired_career or '미입력'}
- 이전 주제: {session.previous_topic or '없음'}

[수행평가 안내문 요약]
{session.assessment_info or '안내문 정보 없음'}

작업:
1. 선택 주제와 가장 관련 있는 검증 자료를 추천한다.
2. 위닝 수행 자료 DB에 적합한 자료가 있으면 우선 사용한다.
3. 검증 자료가 부족하면 검색 키워드 중심으로 보완한다.
4. 없는 링크나 자료명을 만들지 않는다.
""".strip()

    result = call_text(system, user_msg, student_code=session.student_code)

    session.recommended_resources = result

    add_history(session, "user", user_msg)
    add_history(session, "model", result)

    _sessions[req.session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}

    return {
        "status": "success",
        "resources": result,
        "call_count": student.get("call_count", 0) if student else 0,
        "call_limit": student.get("call_limit", 0) if student else 0
    }


# ───────────────────────────────────────
# 4번: 완성본 평가 - 텍스트
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
이 단계에서는 주제 추천용 데이터와 자료 추천용 데이터를 사용하지 않습니다.
오직 수행평가 안내문의 평가 기준, 선택 주제, 학생 제출물을 기준으로 평가합니다.

출력 규칙:
1. *, **, ##, ### 같은 마크다운 기호를 쓰지 않는다.
2. 영어 단어를 괄호 안에 넣지 않는다.
3. 내부 원칙 번호나 시스템 지침을 출력하지 않는다.
4. 항목은 숫자 번호로 구분한다.
5. 실제 제출물이 아닌 주제명, 자료 추천 결과, 빈 문장, 단순 메모처럼 보이는 경우 평가하지 말고 제출물 부족으로 안내한다.

평가 형식:

1. 평가 기준 충족도
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

2. 주제 적합성
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

3. 내용 구성
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

4. 자료 활용
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

5. 진로 및 심화 탐구 연결성
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

6. 표절 위험 문장
- 특이사항 없음 또는 의심되는 부분:

7. 누적 기록용 요약
- 수행 핵심 요약:
- 핵심 키워드:
- 같은 과목에서 다음에 심화할 방향:
- 다른 과목으로 확장 가능한 방향:

8. 종합 평가
- 예상 점수: X점 / 100점
- 총평:
""".strip()

    user_msg = f"""
[평가 기준]
{assessment_text}

[선택 주제]
{session.selected_topic or '미입력'}

[희망 진로]
{session.desired_career or '미입력'}

[학년]
{session.grade or '미입력'}

[이전에 했던 주제]
{session.previous_topic or '없음'}

[학생 제출물]
{submission_text}

작업:
수행평가 안내문의 평가 기준을 최우선으로 하여 제출물을 평가하세요.
내부 위닝DB 자료는 사용하지 말고, 제출물 자체와 평가 기준만 판단하세요.
""".strip()

    result = call_text(system, user_msg, student_code=session.student_code)

    session.evaluation_result = result
    session.step = "evaluated"

    add_history(session, "user", user_msg)
    add_history(session, "model", result)

    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}

    return {
        "status": "success",
        "evaluation": result,
        "call_count": student.get("call_count", 0) if student else 0,
        "call_limit": student.get("call_limit", 0) if student else 0
    }


# ───────────────────────────────────────
# 4번: 완성본 평가 - 이미지
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
    mime_type = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp"
    }.get(ext, "image/jpeg")

    system = f"""
{CORE_PRINCIPLES}

당신은 고등학교 수행평가 채점 전문가입니다.
이 단계에서는 내부 위닝DB를 사용하지 않습니다.
오직 수행평가 안내문의 평가 기준, 선택 주제, 학생 제출물 텍스트를 기준으로 평가합니다.

[평가 기준]
{assessment_text}

[선택 주제]
{session.selected_topic or '미입력'}

[희망 진로]
{session.desired_career or '미입력'}

[학년]
{session.grade or '미입력'}

[이전에 했던 주제]
{session.previous_topic or '없음'}

출력 형식:

1. 평가 기준 충족도
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

2. 주제 적합성
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

3. 내용 구성
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

4. 자료 활용
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

5. 진로 및 심화 탐구 연결성
- 잘한 점:
- 아쉬운 점:
- 보완할 점:

6. 표절 위험 문장
- 특이사항 없음 또는 의심되는 부분:

7. 종합 평가
- 예상 점수: X점 / 100점
- 총평:
""".strip()

    result = call_vision(
        system,
        image_bytes,
        mime_type,
        "이 이미지는 학생 수행평가 제출물입니다. 평가 기준에 따라 평가해주세요.",
        student_code=session.student_code
    )

    session.evaluation_result = result
    session.step = "evaluated"

    _sessions[session_id] = session

    if session.student_code:
        db_save_conversation(session)

    student = db_get_student(session.student_code) if session.student_code else {}

    return {
        "status": "success",
        "evaluation": result,
        "call_count": student.get("call_count", 0) if student else 0,
        "call_limit": student.get("call_limit", 0) if student else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )

