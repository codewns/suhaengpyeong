const API = '/api';

// 반드시 Supabase Project Settings → API에서 가져온 값을 입력하세요.
// SERVICE_ROLE_KEY가 아니라 anon public key만 여기에 넣습니다.
const SUPABASE_URL = 'https://orwngbyiylchpzufwvej.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9yd25nYnlpeWxjaHB6dWZ3dmVqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzI4NzkyMzIsImV4cCI6MjA4ODQ1NTIzMn0.LnKlvHTo9Q_teQd-MTNpdVnGcTT27szXcO-GzGxwfmg';
const STORAGE_BUCKET = 'assessment-images';

const CURRICULUM_2022 = {
  '고1': {
    '1학기': {
      '국어': ['공통국어 1'],
      '수학': ['공통수학 1', '기본수학 1'],
      '영어': ['공통영어 1', '기본영어 1'],
      '한국사': ['한국사 1'],
      '사회': ['통합사회 1'],
      '과학': ['통합과학 1', '과학탐구실험 1']
    },
    '2학기': {
      '국어': ['공통국어 2'],
      '수학': ['공통수학 2', '기본수학 2'],
      '영어': ['공통영어 2', '기본영어 2'],
      '한국사': ['한국사 2'],
      '사회': ['통합사회 2'],
      '과학': ['통합과학 2', '과학탐구실험 2']
    }
  },

  '고2': {
    '1학기': {
      '국어': ['화법과 언어', '독서와 작문', '문학'],
      '수학': ['대수', '미적분Ⅰ', '확률과 통계'],
      '영어': ['영어Ⅰ', '영어Ⅱ', '영어 독해와 작문'],
      '사회': ['세계시민과 지리', '세계사', '사회와 문화', '현대사회와 윤리'],
      '과학': ['물리학', '화학', '생명과학', '지구과학'],
      '체육': ['체육1', '체육2'],
      '예술': ['음악', '미술', '연극'],
      '기술·가정': ['기술·가정'],
      '정보': ['정보'],
      '제2외국어': ['독일어', '프랑스어', '스페인어', '중국어', '일본어', '러시아어', '아랍어', '베트남어'],
      '한문': ['한문'],
      '교양': ['진로와 직업', '생태와 환경']
    },
    '2학기': {
      '국어': ['화법과 언어', '독서와 작문', '문학'],
      '수학': ['대수', '미적분Ⅰ', '확률과 통계'],
      '영어': ['영어Ⅰ', '영어Ⅱ', '영어 독해와 작문'],
      '사회': ['세계시민과 지리', '세계사', '사회와 문화', '현대사회와 윤리'],
      '과학': ['물리학', '화학', '생명과학', '지구과학'],
      '체육': ['체육1', '체육2'],
      '예술': ['음악', '미술', '연극'],
      '기술·가정': ['기술·가정'],
      '정보': ['정보'],
      '제2외국어': ['독일어', '프랑스어', '스페인어', '중국어', '일본어', '러시아어', '아랍어', '베트남어'],
      '한문': ['한문'],
      '교양': ['진로와 직업', '생태와 환경']
    }
  },

  '고3': {
    '1학기': {
      '국어': ['주제 탐구 독서', '문학과 영상', '직무 의사소통', '독서 토론과 글쓰기', '매체 의사소통', '언어생활 탐구'],
      '수학': ['기하', '미적분Ⅱ', '경제 수학', '인공지능 수학', '직무 수학', '수학과 문화', '실용 통계', '수학과제 탐구'],
      '영어': ['영미 문학 읽기', '영어 발표와 토론', '심화 영어', '심화 영어 독해와 작문', '직무 영어', '실생활 영어 회화', '미디어 영어', '세계 문화와 영어'],
      '사회': ['한국지리 탐구', '도시의 미래 탐구', '동아시아 역사 기행', '정치', '법과 사회', '경제', '윤리와 사상', '인문학과 윤리', '국제 관계의 이해', '여행지리', '역사로 탐구하는 현대 세계', '사회문제 탐구', '금융과 경제생활', '윤리문제 탐구', '기후변화와 지속가능한 세계'],
      '과학': ['역학과 에너지', '전자기와 양자', '물질과 에너지', '화학 반응의 세계', '세포와 물질대사', '생물의 유전', '지구시스템과학', '행성우주과학', '과학의 역사와 문화', '기후변화와 환경생태', '융합과학 탐구'],
      '체육': ['운동과 건강', '스포츠 문화', '스포츠 과학', '스포츠 생활1', '스포츠 생활2'],
      '예술': ['음악 연주와 창작', '음악 감상과 비평', '미술 창작', '미술 감상과 비평', '음악과 미디어', '미술과 매체'],
      '기술·가정': ['로봇과 공학세계', '생활과학 탐구', '창의 공학 설계', '지식 재산 일반', '생애 설계와 자립', '아동발달과 부모'],
      '정보': ['인공지능 기초', '데이터 과학', '소프트웨어와 생활'],
      '제2외국어': ['독일어 회화', '프랑스어 회화', '스페인어 회화', '중국어 회화', '일본어 회화', '러시아어 회화', '아랍어 회화', '베트남어 회화', '심화 독일어', '심화 프랑스어', '심화 스페인어', '심화 중국어', '심화 일본어', '심화 러시아어', '심화 아랍어', '심화 베트남어', '독일어권 문화', '프랑스어권 문화', '스페인어권 문화', '중국 문화', '일본 문화', '러시아 문화', '아랍 문화', '베트남 문화'],
      '한문': ['한문 고전 읽기', '언어생활과 한자'],
      '교양': ['인간과 철학', '논리와 사고', '인간과 심리', '교육의 이해', '삶과 종교', '보건', '인간과 경제활동', '논술']
    },
    '2학기': {
      '국어': ['주제 탐구 독서', '문학과 영상', '직무 의사소통', '독서 토론과 글쓰기', '매체 의사소통', '언어생활 탐구'],
      '수학': ['기하', '미적분Ⅱ', '경제 수학', '인공지능 수학', '직무 수학', '수학과 문화', '실용 통계', '수학과제 탐구'],
      '영어': ['영미 문학 읽기', '영어 발표와 토론', '심화 영어', '심화 영어 독해와 작문', '직무 영어', '실생활 영어 회화', '미디어 영어', '세계 문화와 영어'],
      '사회': ['한국지리 탐구', '도시의 미래 탐구', '동아시아 역사 기행', '정치', '법과 사회', '경제', '윤리와 사상', '인문학과 윤리', '국제 관계의 이해', '여행지리', '역사로 탐구하는 현대 세계', '사회문제 탐구', '금융과 경제생활', '윤리문제 탐구', '기후변화와 지속가능한 세계'],
      '과학': ['역학과 에너지', '전자기와 양자', '물질과 에너지', '화학 반응의 세계', '세포와 물질대사', '생물의 유전', '지구시스템과학', '행성우주과학', '과학의 역사와 문화', '기후변화와 환경생태', '융합과학 탐구'],
      '체육': ['운동과 건강', '스포츠 문화', '스포츠 과학', '스포츠 생활1', '스포츠 생활2'],
      '예술': ['음악 연주와 창작', '음악 감상과 비평', '미술 창작', '미술 감상과 비평', '음악과 미디어', '미술과 매체'],
      '기술·가정': ['로봇과 공학세계', '생활과학 탐구', '창의 공학 설계', '지식 재산 일반', '생애 설계와 자립', '아동발달과 부모'],
      '정보': ['인공지능 기초', '데이터 과학', '소프트웨어와 생활'],
      '제2외국어': ['독일어 회화', '프랑스어 회화', '스페인어 회화', '중국어 회화', '일본어 회화', '러시아어 회화', '아랍어 회화', '베트남어 회화', '심화 독일어', '심화 프랑스어', '심화 스페인어', '심화 중국어', '심화 일본어', '심화 러시아어', '심화 아랍어', '심화 베트남어', '독일어권 문화', '프랑스어권 문화', '스페인어권 문화', '중국 문화', '일본 문화', '러시아 문화', '아랍 문화', '베트남 문화'],
      '한문': ['한문 고전 읽기', '언어생활과 한자'],
      '교양': ['인간과 철학', '논리와 사고', '인간과 심리', '교육의 이해', '삶과 종교', '보건', '인간과 경제활동', '논술']
    }
  }
};

const SPECIALIZED_COURSES_2022 = {
  '수학': ['전문 수학', '이산 수학', '고급 기하', '고급 대수', '고급 미적분'],
  '과학': ['고급 물리학', '고급 화학', '고급 생명과학', '고급 지구과학', '과학과제 연구', '물리학 실험', '화학 실험', '생명과학 실험', '지구과학 실험'],
  '정보': ['정보과학'],
  '체육': ['스포츠 개론', '육상', '체조', '수상 스포츠', '기초 체육 전공 실기', '심화 체육 전공 실기', '고급 체육 전공 실기', '스포츠 경기 체력', '스포츠 경기 기술', '스포츠 경기 분석', '스포츠 교육', '스포츠 생리의학', '스포츠 행정 및 경영'],
  '예술': ['음악 이론', '음악사', '시창·청음', '음악 전공 실기', '합창·합주', '음악 공연 실습', '미술 이론', '드로잉', '미술사', '미술 전공 실기', '조형 탐구', '무용의 이해', '무용과 몸', '무용 기초 실기', '무용 전공 실기', '안무', '무용 제작 실습', '무용 감상과 비평', '문예 창작의 이해', '문장론', '문학 감상과 비평', '시 창작', '소설 창작', '극 창작', '연극과 몸', '연극과 말', '연기', '무대 미술과 기술', '연극 제작 실습', '연극 감상과 비평', '영화의 이해', '촬영·조명', '편집·사운드', '영화 제작 실습', '영화 감상과 비평', '사진의 이해', '사진 촬영', '사진 표현 기법', '영상 제작의 이해', '사진 감상과 비평', '음악과 문화', '미술 매체 탐구', '미술과 사회', '무용과 매체', '문학과 매체', '연극과 삶', '영화와 삶', '사진과 삶'],
  '제2외국어': ['전공 기초 독일어', '독일어 회화Ⅰ', '독일어 회화Ⅱ', '독일어 독해와 작문Ⅰ', '독일어 독해와 작문Ⅱ', '전공 기초 프랑스어', '프랑스어 회화Ⅰ', '프랑스어 회화Ⅱ', '프랑스어 독해와 작문Ⅰ', '프랑스어 독해와 작문Ⅱ', '전공 기초 스페인어', '스페인어 회화Ⅰ', '스페인어 회화Ⅱ', '스페인어 독해와 작문Ⅰ', '스페인어 독해와 작문Ⅱ', '전공 기초 중국어', '중국어 회화Ⅰ', '중국어 회화Ⅱ', '중국어 독해와 작문Ⅰ', '중국어 독해와 작문Ⅱ', '전공 기초 일본어', '일본어 회화Ⅰ', '일본어 회화Ⅱ', '일본어 독해와 작문Ⅰ', '일본어 독해와 작문Ⅱ', '전공 기초 러시아어', '러시아어 회화Ⅰ', '러시아어 회화Ⅱ', '러시아어 독해와 작문Ⅰ', '러시아어 독해와 작문Ⅱ', '전공 기초 아랍어', '아랍어 회화Ⅰ', '아랍어 회화Ⅱ', '아랍어 독해와 작문Ⅰ', '아랍어 독해와 작문Ⅱ', '전공 기초 베트남어', '베트남어 회화Ⅰ', '베트남어 회화Ⅱ', '베트남어 독해와 작문Ⅰ', '베트남어 독해와 작문Ⅱ']
};

function shouldIncludeSpecializedCourses(schoolType) {
  return schoolType === '특수목적고' || schoolType === '자율형 사립고';
}

function getCurriculumGroups(grade, semester) {
  return Object.keys(CURRICULUM_2022?.[grade]?.[semester] || {});
}

function getCurriculumSubjects(grade, semester, group, schoolType = '일반고') {
  const base = CURRICULUM_2022?.[grade]?.[semester]?.[group] || [];
  const specialized = shouldIncludeSpecializedCourses(schoolType)
    ? (SPECIALIZED_COURSES_2022[group] || [])
    : [];

  return [...base, ...specialized];
}

function subjectLabel(group, subject) {
  if (!group && !subject) return '';
  return group ? `${group} / ${subject}` : subject;
}

function updateCurriculumSelects(prefix) {
  const gradeEl = document.getElementById(`${prefix}GradeInput`);
  const semesterEl = document.getElementById(`${prefix}SemesterInput`);
  const schoolTypeEl = document.getElementById(`${prefix}SchoolTypeInput`);
  const groupEl = document.getElementById(`${prefix}SubjectGroupInput`);
  const subjectEl = document.getElementById(`${prefix}SubjectInput`);

  if (!gradeEl || !semesterEl || !schoolTypeEl || !groupEl || !subjectEl) return;

  const grade = gradeEl.value;
  const semester = semesterEl.value;
  const schoolType = schoolTypeEl.value || '일반고';

  const groups = getCurriculumGroups(grade, semester);
  const currentGroup = groupEl.value;

  groupEl.innerHTML = groups.map(g => `<option value="${esc(g)}">${esc(g)}</option>`).join('');

  if (groups.includes(currentGroup)) {
    groupEl.value = currentGroup;
  }

  const group = groupEl.value;
  const subjects = getCurriculumSubjects(grade, semester, group, schoolType);
  const currentSubject = subjectEl.value;

  subjectEl.innerHTML = [
    ...subjects.map(s => `<option value="${esc(s)}">${esc(s)}</option>`),
    '<option value="직접 입력">직접 입력</option>'
  ].join('');

  if (subjects.includes(currentSubject)) {
    subjectEl.value = currentSubject;
  }

  const customWrap = document.getElementById(`${prefix}CustomSubjectWrap`);
  if (customWrap) {
    customWrap.style.display = subjectEl.value === '직접 입력' ? 'block' : 'none';
  }
}

function getSelectedCurriculum(prefix) {
  const grade = document.getElementById(`${prefix}GradeInput`)?.value || '고등학생';
  const semester = document.getElementById(`${prefix}SemesterInput`)?.value || '';
  const schoolType = document.getElementById(`${prefix}SchoolTypeInput`)?.value || '일반고';
  const group = document.getElementById(`${prefix}SubjectGroupInput`)?.value || '';
  const subjectChoice = document.getElementById(`${prefix}SubjectInput`)?.value || '';
  const customSubject = document.getElementById(`${prefix}CustomSubjectInput`)?.value?.trim() || '';

  const realSubject = subjectChoice === '직접 입력' ? customSubject : subjectChoice;
  const subject = subjectLabel(group, realSubject);

  return {
    grade,
    semester,
    gradeWithSemester: semester ? `${grade} ${semester}` : grade,
    school_type: schoolType,
    subject_group: group,
    subject_name: realSubject,
    subject,
    include_specialized: shouldIncludeSpecializedCourses(schoolType)
  };
}

const sb = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

let sessionId = null;
let studentInfo = null;
let selectedTopic = null;
let assessmentInfo = null;
let studentName = null;
let studentCode = null;
let prevSessionData = null;
let latestPlanReport = null;
let latestPlanReportId = null;
let latestEvaluationReportId = null;

let pendingImages = [];

// ───────────────────────────────────────
// 로그인
// ───────────────────────────────────────
async function doLogin() {
  const name = document.getElementById('loginName').value.trim();
  const code = document.getElementById('loginCode').value.trim().toUpperCase();
  const errEl = document.getElementById('loginError');
  if(!name || !code) { errEl.textContent='이름과 코드를 모두 입력해주세요.'; errEl.style.display='block'; return; }
  errEl.style.display='none';
  const btn = document.querySelector('#loginOverlay button');
  btn.textContent='확인 중...'; btn.disabled=true;
  try {
    const res = await fetch(`${API}/login`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name,code}) });
    const data = await res.json();
    if(!res.ok) { errEl.textContent=data.detail||'로그인 실패'; errEl.style.display='block'; btn.textContent='시작하기 →'; btn.disabled=false; return; }
    sessionId = data.session_id; studentName = data.name; studentCode = data.code;
    document.getElementById('headerBadge').textContent = data.name;
    document.getElementById('loginOverlay').style.display = 'none';
    updateCallUsage(data.call_count || 0, data.call_limit || 0);
    if(data.previous && (data.previous.topics || data.previous.resources || data.previous.evaluation)) {
      showPreviousHistory(data.previous);
    } else {
      showWelcome(data.name);
    }
  } catch(e) {
    errEl.textContent='서버 연결 실패. 잠시 후 다시 시도해주세요.'; errEl.style.display='block';
    btn.textContent='시작하기 →'; btn.disabled=false;
  }
}

function showPreviousHistory(prev) {
  prevSessionData = prev;
  addAiMsg(`반갑습니다, <strong>${studentName}</strong>님! 👋<br><br>
    지난번 기록을 불러왔어요.<br>
    📚 학년/과목: <strong>${prev.grade||'-'} / ${prev.subject||'-'}</strong><br>
    🎯 선택 주제: <strong>${prev.selected_topic||'아직 선택 전'}</strong><br><br>
    이어서 하시겠어요?`, true);
  const html = `
    <div class="quick-btns">
      <button class="quick-btn" onclick="resumeFromPrev()">▶ 이어서 하기</button>
      <button class="quick-btn" onclick="showWelcome('${esc(studentName)}')">🔄 새로 시작하기</button>
    </div>`;
  addAiMsg(html, true);
  if(prev.resources) renderResources(prev.resources, prev.selected_topic || '');
  if(prev.evaluation) renderEvaluation(prev.evaluation);
}

function resumeFromPrev() {
  if(!prevSessionData) return;
  const p = prevSessionData;
  let topicObj = { title: p.selected_topic || '', details: {} };

  if(p.selected_topic_detail) {
    const details = {};
    p.selected_topic_detail.split('\n').forEach(line => {
      const idx = line.indexOf(': ');
      if(idx > -1) details[line.slice(0, idx)] = line.slice(idx + 2);
    });
    topicObj.details = details;
  } else if(p.topics && p.selected_topic) {
    const parsed = parseTopics(p.topics);
    const match = parsed.find(t => t.title && t.title.includes((p.selected_topic || '').slice(0,10)));
    if(match) topicObj = match;
  }

  resumeSession(
    p.subject || '',
    p.career || '',
    p.selected_topic || '',
    p.topics || '',
    topicObj,
    p.grade || '고등학생'
  );
}

function resumeSession(subject, career, topic, topics, topicObj, grade = '고등학생') {
  studentInfo = { grade, subject, career, prevTopic: '' };
  selectedTopic = topic || null;

  document.getElementById('headerTitle').textContent = `${grade && grade !== '고등학생' ? grade + ' ' : ''}${subject || '수행평가'} 수행평가`;
  document.getElementById('headerBadge').textContent = career || studentName;

  document.querySelectorAll('.header-tab').forEach((t,i) => t.classList.toggle('active', i===0));
  document.getElementById('panelChat').style.display = 'flex';
  document.getElementById('panelResources').style.display = 'none';
  document.getElementById('panelEval').style.display = 'none';

  const rightPanel = document.querySelector('.right-panel');
  rightPanel.classList.remove('fullwidth');

  if(topic) {
    updateStep(5);
    renderTopicSummary(topicObj || { title: topic, details: {} });
    addAiMsg(`좋아요! <strong>${esc(topic)}</strong> 주제로 이어서 진행할게요 😊<br><br>완성본이 준비되면 평가받으세요!`, true);
    if(!document.querySelector('.submission-textarea')) showEvalInput();
  } else if(topics) {
    rightPanel.style.display = 'none';
    updateStep(4);
    addAiMsg(`지난번에 추천받은 주제 중에서 골라주세요! 🎯`, true);
    showTopicCards(topics);
  } else {
    rightPanel.style.display = 'none';
    updateStep(3);
    addAiMsg(`주제부터 다시 추천받을게요!`, true);
    requestTopics();
  }
}

// ───────────────────────────────────────
// 환영
// ───────────────────────────────────────
function updateCallUsage(count, limit) {
  if(!limit) return;
  const remaining = Math.max(0, limit - count);
  const pct = Math.round((remaining / limit) * 100);
  const color = pct > 50 ? '#22c55e' : pct > 20 ? '#f59e0b' : '#ef4444';
  document.getElementById('callUsageWrap').style.display = 'flex';
  document.getElementById('callUsageText').textContent = `잔여 ${remaining}회`;
  document.getElementById('callUsageBar').style.width = pct + '%';
  document.getElementById('callUsageBar').style.background = color;
  document.getElementById('callUsagePct').style.color = color;
  document.getElementById('callUsagePct').textContent = pct + '%';
}

function showWelcome(name) {
  document.getElementById('chatMessages').innerHTML = '';

  const topicSummary = document.getElementById('panelTopicSummary');
  if(topicSummary) topicSummary.style.display = 'none';

  document.querySelector('.right-panel').style.display = 'none';
  selectedTopic = null;
  assessmentInfo = null;
  pendingImages = [];

  addAiMsg(`안녕하세요, <strong>${name}</strong>님! 👋<br><br>
    수행평가를 도와드릴게요!<br>
    먼저 <strong>학년, 과목, 같은 과목에서 이전에 했던 주제, 희망 진로</strong>를 알려주세요 📝`, true);

  showPreUploadForm();
}

function showPreUploadForm() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-ai';
  wrap.id = 'preUploadFormWrap';

  wrap.innerHTML = `
    <div class="ai-avatar">🤖</div>
    <div style="width:100%">
      <div class="ai-name">위닝AI 수행평가 서포터</div>
      <div class="info-form" id="preUploadForm">
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">학년 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="preGradeInput" onchange="updateCurriculumSelects('pre')">
              <option>고1</option>
              <option>고2</option>
              <option>고3</option>
            </select>
          </div>

          <div class="form-group">
            <label class="form-label">학기 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="preSemesterInput" onchange="updateCurriculumSelects('pre')">
              <option>1학기</option>
              <option>2학기</option>
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="form-label">학교 유형 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="preSchoolTypeInput" onchange="updateCurriculumSelects('pre')">
              <option>일반고</option>
              <option>자율형 사립고</option>
              <option>특수목적고</option>
            </select>
          </div>

          <div class="form-group">
            <label class="form-label">교과군 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="preSubjectGroupInput" onchange="updateCurriculumSelects('pre')"></select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="form-label">과목 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="preSubjectInput" onchange="updateCurriculumSelects('pre')"></select>
          </div>

          <div class="form-group" id="preCustomSubjectWrap" style="display:none;">
            <label class="form-label">직접 입력 과목명</label>
            <input class="form-input" id="preCustomSubjectInput" placeholder="예: 과학과제 연구, 고급 생명과학, 학교 자체 편성 과목" />
          </div>
        </div>

        <div style="font-size:12px;color:var(--text-sub);line-height:1.6;">
          ※ 학교 유형은 과목명이 아니라 과목 목록을 조정하기 위한 선택값입니다.<br>
          ※ 자율형 사립고·특수목적고를 선택하면 전문교과 과목이 각 교과군에 함께 표시됩니다.<br>
          ※ 학교별 교육과정 편성에 따라 실제 이수 학기는 다를 수 있습니다.
        </div>

        <div class="form-group">
          <label class="form-label">같은 과목에서 이전에 한 주제</label>
          <input class="form-input" id="prePrevTopicInput" placeholder="예: 항생제 내성, 조건부확률, 소설 속 인물 심리 분석" />
        </div>

        <div class="form-group">
          <label class="form-label">희망 진로 <span style="color:#ef4444;">*</span></label>
          <input class="form-input" id="preCareerInput" placeholder="예: 의학, 경영, 반도체, 약학, 항공"
            onkeydown="if(event.key==='Enter') submitPreForm()" />
        </div>

        <button class="form-submit" onclick="submitPreForm()">다음 단계 →</button>
      </div>
    </div>
  `;

  document.getElementById('chatMessages').appendChild(wrap);
  updateCurriculumSelects('pre');
  scrollBot();
}

function submitPreForm() {
  const curriculum = getSelectedCurriculum('pre');
  const grade = curriculum.grade;
  const semester = curriculum.semester;
  const schoolType = curriculum.school_type;
  const subject = curriculum.subject;
  const prevTopic = document.getElementById('prePrevTopicInput').value.trim();
  const career = document.getElementById('preCareerInput').value.trim();

  if(!curriculum.subject_name) {
    showToast('과목을 선택하거나 직접 입력해주세요!');
    return;
  }

  if(!career) {
    showToast('희망 진로를 입력해주세요!');
    return;
  }

  document.getElementById('preUploadForm').style.display = 'none';
  updateStep(2);

  studentInfo = {
    ...(studentInfo || {}),
    grade,
    semester,
    gradeWithSemester: curriculum.gradeWithSemester,
    schoolType,
    subject,
    subjectGroup: curriculum.subject_group,
    subjectName: curriculum.subject_name,
    includeSpecialized: curriculum.include_specialized,
    prevTopic,
    career
  };

  document.getElementById('headerTitle').textContent = `${grade} ${semester} ${subject} 수행평가`;
  document.getElementById('headerBadge').textContent = career;

  addUserMsg(`학년: ${grade} / 학기: ${semester} / 학교 유형: ${schoolType} / 과목: ${subject}${prevTopic ? ' / 이전 주제: ' + prevTopic : ''} / 진로: ${career}`);

  addAiMsg(`좋아요! 이제 <strong>수행평가 안내문 사진</strong>을 올려주세요 📄<br>
    <span style="font-size:12px;color:var(--text-sub);">여러 장도 한 번에 업로드할 수 있어요!</span>`, true);

  showUploadArea();
}

// ───────────────────────────────────────
// 멀티 이미지 업로드
// ───────────────────────────────────────
function showUploadArea() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-ai';
  wrap.id = 'uploadAreaWrap';
  wrap.innerHTML = `
    <div class="ai-avatar">🤖</div>
    <div style="width:100%;max-width:520px;">
      <div class="ai-name">위닝AI 수행평가 서포터</div>
      <div id="imagePreviewGrid" class="image-preview-grid">
        <div class="image-add-btn" onclick="document.getElementById('multiFileInput').click()">
          <span>+</span>
          <span style="font-size:10px;margin-top:2px;">사진 추가</span>
        </div>
      </div>
      <input type="file" id="multiFileInput" style="display:none" accept="image/*" multiple onchange="addImages(event)">
      <div style="font-size:12px;color:var(--text-sub);margin-top:8px;">PNG, JPG · 여러 장 선택 가능</div>
      <button class="upload-confirm-btn" id="uploadConfirmBtn" onclick="uploadAllImages()" disabled>
        📤 분석 시작하기
      </button>
      <div class="quick-btns" style="margin-top:10px;">
        <button class="quick-btn" onclick="skipToFormFromUpload()">📝 안내문 없이 시작하기</button>
      </div>
    </div>`;
  document.getElementById('chatMessages').appendChild(wrap);
  scrollBot();
}

function addImages(event) {
  const files = Array.from(event.target.files || []);
  const allowedTypes = ['image/jpeg', 'image/png', 'image/webp'];
  const maxSize = 10 * 1024 * 1024;

  files.forEach(file => {
    if(!allowedTypes.includes(file.type)) {
      showToast('JPG, PNG, WEBP 이미지만 업로드할 수 있어요.');
      return;
    }

    if(file.size > maxSize) {
      showToast('이미지는 10MB 이하만 업로드할 수 있어요.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      pendingImages.push({ file, dataUrl: e.target.result });
      renderImagePreviews();
    };
    reader.readAsDataURL(file);
  });

  event.target.value = '';
}

function renderImagePreviews() {
  const grid = document.getElementById('imagePreviewGrid');
  if(!grid) return;
  grid.innerHTML = '';
  pendingImages.forEach((img, idx) => {
    const wrap = document.createElement('div');
    wrap.className = 'image-thumb-wrap';
    wrap.innerHTML = `
      <img src="${img.dataUrl}" alt="미리보기">
      <button class="image-thumb-remove" onclick="removeImage(${idx})">✕</button>`;
    grid.appendChild(wrap);
  });
  const addBtn = document.createElement('div');
  addBtn.className = 'image-add-btn';
  addBtn.innerHTML = '<span>+</span><span style="font-size:10px;margin-top:2px;">추가</span>';
  addBtn.onclick = () => document.getElementById('multiFileInput').click();
  grid.appendChild(addBtn);

  const confirmBtn = document.getElementById('uploadConfirmBtn');
  if(confirmBtn) confirmBtn.disabled = pendingImages.length === 0;
}

function removeImage(idx) {
  pendingImages.splice(idx, 1);
  renderImagePreviews();
}

async function uploadImageToStorage(file, purpose = 'assessment') {
  if (!sessionId) {
    throw new Error('세션이 없습니다. 다시 로그인해주세요.');
  }

  if (!SUPABASE_URL || SUPABASE_URL.includes('YOUR_SUPABASE_URL')) {
    throw new Error('index.html의 SUPABASE_URL 값을 실제 Supabase URL로 바꿔주세요.');
  }

  if (!SUPABASE_ANON_KEY || SUPABASE_ANON_KEY.includes('YOUR_SUPABASE_ANON_PUBLIC_KEY')) {
    throw new Error('index.html의 SUPABASE_ANON_KEY 값을 실제 anon public key로 바꿔주세요.');
  }

  const contentType = file.type || 'image/jpeg';

  const signedRes = await apiFetch(`${API}/storage-signed-upload-url`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      session_id: sessionId,
      file_name: file.name || 'image.jpg',
      content_type: contentType,
      purpose
    })
  });

  const signedData = await signedRes.json();

  if (!signedRes.ok || !signedData.token || !signedData.path) {
    throw new Error(signedData.detail || '이미지 업로드 주소 생성에 실패했습니다.');
  }

  const { error: uploadError } = await sb
    .storage
    .from(STORAGE_BUCKET)
    .uploadToSignedUrl(
      signedData.path,
      signedData.token,
      file,
      {
        contentType,
        cacheControl: '3600'
      }
    );

  if (uploadError) {
    throw new Error(uploadError.message || 'Supabase Storage 업로드에 실패했습니다.');
  }

  return {
    path: signedData.path,
    mime_type: contentType
  };
}

async function uploadAllImages() {
  if(!pendingImages.length) {
    showToast('사진을 먼저 추가해주세요!');
    return;
  }

  if(!sessionId) {
    showToast('서버 연결 중이에요. 잠시 후 다시 시도해주세요.');
    return;
  }

  const uploadWrap = document.getElementById('uploadAreaWrap');
  if(uploadWrap) uploadWrap.style.opacity = '0.5';

  addUserMsg(`📎 안내문 사진 ${pendingImages.length}장 업로드`);
  addLoading();
  updateStep(3);

  try {
    let combinedResult = '';

    for(let i = 0; i < pendingImages.length; i++) {
      const uploaded = await uploadImageToStorage(pendingImages[i].file, 'assessment');

      const res = await apiFetch(`${API}/analyze-assessment-storage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
  session_id: sessionId,
  image_path: uploaded.path,
  mime_type: uploaded.mime_type,
  subject: studentInfo?.subject || '',
  career: studentInfo?.career || '',
  grade: studentInfo?.gradeWithSemester || studentInfo?.grade || '고등학생',
  school_type: studentInfo?.schoolType || '일반고'
})
      });

      const data = await res.json();

      if(!res.ok) {
        throw new Error(data.detail || '분석 중 오류가 발생했습니다.');
      }

      if(data.call_count !== undefined) {
        updateCallUsage(data.call_count, data.call_limit || 0);
      }

      combinedResult += (pendingImages.length > 1 ? `\n\n[사진 ${i+1}]\n` : '') + data.assessment_info;
    }

    removeLoading();
    assessmentInfo = combinedResult;

    if(uploadWrap) uploadWrap.remove();
    pendingImages = [];

    addAiMsg(`✅ <strong>안내문 분석 완료!</strong> 이제 주제를 추천해드릴게요!`, true);
    await requestTopics();

  } catch(e) {
    removeLoading();
    if(uploadWrap) uploadWrap.style.opacity = '1';
    addAiMsg(`❌ ${e.message || '분석 중 오류가 발생했어요. 다시 시도해주세요.'}`);
  }
}

// ───────────────────────────────────────
// 주제 추천
// ───────────────────────────────────────
async function requestTopics() {
  addLoading();

  try {
    const res = await apiFetch(`${API}/recommend-topics`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
  session_id: sessionId,
  grade: studentInfo?.gradeWithSemester || studentInfo?.grade || '고등학생',
  subject: studentInfo?.subject || '국어',
  school_type: studentInfo?.schoolType || '일반고',
  desired_career: studentInfo?.career || '',
  previous_topic: studentInfo?.prevTopic || '',
  assessment_info: assessmentInfo || null
})
    });

    const data = await res.json();

    if(!res.ok) {
      throw new Error(data.detail || '주제 추천 중 오류가 발생했어요.');
    }

    if(data.call_count !== undefined) updateCallUsage(data.call_count, data.call_limit || 0);
    removeLoading();
    updateStep(4);
    showTopicCards(data.topics);
  } catch(e) {
    removeLoading();
    addAiMsg(`❌ ${e.message || '주제 추천 중 오류가 발생했어요.'}`);
  }
}

function showCareerForm() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-ai';
  wrap.innerHTML = `
    <div class="ai-avatar">🤖</div>
    <div style="width:100%">
      <div class="ai-name">위닝AI 수행평가 서포터</div>
      <div class="info-form" id="careerForm">
        <div class="form-group">
          <label class="form-label">희망 진로 <span style="color:#ef4444;">*</span></label>
          <input class="form-input" id="careerOnlyInput" placeholder="예: 심리상담사, 의사, AI 개발자"
            onkeydown="if(event.key==='Enter') submitCareerOnly()"/>
        </div>
        <button class="form-submit" onclick="submitCareerOnly()">✨ 주제 추천받기</button>
      </div>
    </div>`;
  document.getElementById('chatMessages').appendChild(wrap);
  scrollBot();
}

async function submitCareerOnly() {
  const career = document.getElementById('careerOnlyInput')?.value?.trim();

  if(!career) {
    showToast('희망 진로를 입력해주세요!');
    return;
  }

  document.getElementById('careerForm').style.display = 'none';

  studentInfo = {
    ...(studentInfo || {}),
    career
  };

  document.getElementById('headerBadge').textContent = career;

  addUserMsg(`희망 진로: ${career}`);
  addLoading();

  try {
    const res = await apiFetch(`${API}/recommend-topics`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        session_id: sessionId,
        grade: studentInfo?.gradeWithSemester || studentInfo?.grade || '고등학생',
        subject: studentInfo?.subject || '국어',
        school_type: studentInfo?.schoolType || '일반고',
        desired_career: career,
        previous_topic: studentInfo?.prevTopic || '',
        assessment_info: assessmentInfo || null
      })
    });

    const data = await res.json();

    if(!res.ok) {
      throw new Error(data.detail || '오류가 발생했어요. 다시 시도해주세요.');
    }

    if(data.call_count !== undefined) updateCallUsage(data.call_count, data.call_limit || 0);

    removeLoading();
    updateStep(4);
    showTopicCards(data.topics);
  } catch(e) {
    removeLoading();
    addAiMsg(`❌ ${e.message || '오류가 발생했어요. 다시 시도해주세요.'}`);
  }
}

// ───────────────────────────────────────
// 안내문 없이 시작
// ───────────────────────────────────────
function skipToFormFromUpload() {
  const uploadWrap = document.getElementById('uploadAreaWrap');
  if(uploadWrap) uploadWrap.remove();
  pendingImages = [];
  addUserMsg('안내문 없이 시작할게요');
  updateStep(3);
  addAiMsg('알겠어요! 아래 정보를 입력해주세요 😊', true);
  showSkipForm();
}

function showSkipForm() {
  const wrap = document.createElement('div');
  wrap.className = 'msg-ai';

  wrap.innerHTML = `
    <div class="ai-avatar">🤖</div>
    <div style="width:100%">
      <div class="ai-name">위닝AI 수행평가 서포터</div>
      <div class="info-form" id="skipForm">
        <div class="form-row">
          <div class="form-group">
            <label class="form-label">학년 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="skipGradeInput" onchange="updateCurriculumSelects('skip')">
              <option>고1</option>
              <option>고2</option>
              <option>고3</option>
            </select>
          </div>

          <div class="form-group">
            <label class="form-label">학기 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="skipSemesterInput" onchange="updateCurriculumSelects('skip')">
              <option>1학기</option>
              <option>2학기</option>
            </select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="form-label">학교 유형 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="skipSchoolTypeInput" onchange="updateCurriculumSelects('skip')">
              <option>일반고</option>
              <option>자율형 사립고</option>
              <option>특수목적고</option>
            </select>
          </div>

          <div class="form-group">
            <label class="form-label">교과군 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="skipSubjectGroupInput" onchange="updateCurriculumSelects('skip')"></select>
          </div>
        </div>

        <div class="form-row">
          <div class="form-group">
            <label class="form-label">과목 <span style="color:#ef4444;">*</span></label>
            <select class="form-select" id="skipSubjectInput" onchange="updateCurriculumSelects('skip')"></select>
          </div>

          <div class="form-group" id="skipCustomSubjectWrap" style="display:none;">
            <label class="form-label">직접 입력 과목명</label>
            <input class="form-input" id="skipCustomSubjectInput" placeholder="예: 과학과제 연구, 고급 생명과학, 학교 자체 편성 과목" />
          </div>
        </div>

        <div style="font-size:12px;color:var(--text-sub);line-height:1.6;">
          ※ 학교 유형은 과목명이 아니라 과목 목록을 조정하기 위한 선택값입니다.<br>
          ※ 자율형 사립고·특수목적고를 선택하면 전문교과 과목이 각 교과군에 함께 표시됩니다.<br>
          ※ 학교별 교육과정 편성에 따라 실제 이수 학기는 다를 수 있습니다.
        </div>

        <div class="form-group">
          <label class="form-label">같은 과목에서 이전에 한 주제</label>
          <input class="form-input" id="skipPrevTopicInput" placeholder="예: 항생제 내성, 조건부확률, 소설 속 인물 심리 분석"/>
        </div>

        <div class="form-group">
          <label class="form-label">대략적인 수행평가 정보 <span style="color:#ef4444;">*</span></label>
          <textarea class="form-textarea" id="skipAssessInfoInput" placeholder="수행평가 유형, 제출 형식, 평가 기준, 필수 포함 내용 등을 적어주세요."></textarea>
        </div>

        <div class="form-group">
          <label class="form-label">희망 진로 <span style="color:#ef4444;">*</span></label>
          <input class="form-input" id="skipCareerInput" placeholder="예: 의학, 경영, 반도체, 약학, 항공"/>
        </div>

        <button class="form-submit" onclick="submitSkipForm()">✨ 주제 추천받기</button>
      </div>
    </div>
  `;

  document.getElementById('chatMessages').appendChild(wrap);
  updateCurriculumSelects('skip');
  scrollBot();
}

async function submitSkipForm() {
  const curriculum = getSelectedCurriculum('skip');
  const grade      = curriculum.grade;
  const semester   = curriculum.semester;
  const schoolType = curriculum.school_type;
  const subject    = curriculum.subject;
  const prevTopic  = document.getElementById('skipPrevTopicInput').value.trim();
  const assessInfo = document.getElementById('skipAssessInfoInput').value.trim();
  const career     = document.getElementById('skipCareerInput').value.trim();

  if(!curriculum.subject_name) {
    showToast('과목을 선택하거나 직접 입력해주세요!');
    return;
  }

  if(!assessInfo) {
    showToast('수행평가 정보를 입력해주세요!');
    return;
  }

  if(!career) {
    showToast('희망 진로를 입력해주세요!');
    return;
  }

  document.getElementById('skipForm').style.display = 'none';

  studentInfo = {
    grade,
    semester,
    gradeWithSemester: curriculum.gradeWithSemester,
    schoolType,
    subject,
    subjectGroup: curriculum.subject_group,
    subjectName: curriculum.subject_name,
    includeSpecialized: curriculum.include_specialized,
    prevTopic,
    career
  };

  assessmentInfo = assessInfo;

  document.getElementById('headerTitle').textContent = `${grade} ${semester} ${subject} 수행평가`;
  document.getElementById('headerBadge').textContent = career;

  addUserMsg(`학년: ${grade} / 학기: ${semester} / 학교 유형: ${schoolType} / 과목: ${subject}${prevTopic ? ' / 이전 주제: ' + prevTopic : ''} / 진로: ${career}`);
  addLoading();

  try {
    const res = await apiFetch(`${API}/recommend-topics`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        session_id: sessionId,
        grade: curriculum.gradeWithSemester,
        subject,
        school_type: schoolType,
        desired_career: career,
        previous_topic: prevTopic,
        assessment_info: assessInfo
      })
    });

    const data = await res.json();

    if(!res.ok) {
      throw new Error(data.detail || '오류가 발생했어요. 다시 시도해주세요.');
    }

    if(data.call_count !== undefined) updateCallUsage(data.call_count, data.call_limit || 0);

    removeLoading();
    updateStep(4);
    showTopicCards(data.topics);

  } catch(e) {
    removeLoading();
    addAiMsg(`죄송해요. 주제 추천 중 오류가 발생했어요.<br><span style="color:#ef4444">${esc(e.message)}</span>`, true);
  }
}

// ───────────────────────────────────────
// 주제 카드 (3개 각각 별도 말풍선)
// ───────────────────────────────────────
let topicDataStore = [];

function parseTopics(text) {
  const topics = [];
  const blocks = text.split(/(?=추천\s*[123]\s*[:：.\-–])/);
  for(const block of blocks) {
    const titleMatch = block.match(/추천\s*\d+\s*[:：.\-–]?\s*(.+)/);
    if(!titleMatch) continue;
    const title = titleMatch[1].replace(/[\[\]\*\#`]/g,'').trim();
    if(!title) continue;

    const lines = block.split('\n').slice(1);
    const details = {};
    let curKey = '';
    for(const line of lines) {
      const numKeyMatch = line.match(/^\d+\.\s*(.+?)[:：]\s*(.*)/);
      const dashKeyMatch = line.match(/^[-•]\s*(.+?)[:：]\s*(.*)/);
      const keyMatch = numKeyMatch || dashKeyMatch;
      if(keyMatch) {
        curKey = keyMatch[1].trim();
        details[curKey] = keyMatch[2].trim();
      } else if(curKey && line.trim()) {
        details[curKey] += ' ' + line.trim();
      }
    }
    for(const k of Object.keys(details)) {
      details[k] = details[k].replace(/\s*---+\s*$/g,'').trim();
    }
    topics.push({ title, details, raw: block });
    if(topics.length >= 3) break;
  }
  return topics;
}

function showTopicCards(text) {
  topicDataStore = parseTopics(text);

  if(topicDataStore.length === 0) {
    const firstLine = text.split('\n').find(l => l.trim() && !l.startsWith('#')) || '추천 주제';
    topicDataStore = [{ title: firstLine.replace(/추천\s*\d+\s*[:：.]\s*/,'').replace(/[\[\]\*\#`]/g,'').trim() || '추천 주제', details:{}, raw: text }];
  }

  addAiMsg(`주제를 골라주세요! 카드를 클릭하면 자세한 내용을 볼 수 있어요 🎯`, true);

  topicDataStore.forEach((t, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'msg-ai';
    wrap.innerHTML = `
      <div class="ai-avatar">🤖</div>
      <div style="width:100%;max-width:520px;">
        <div class="ai-name">위닝AI 수행평가 서포터</div>
        <div class="recommend-card" id="topicCard${i}" onclick="openTopicModal(${i})" style="cursor:pointer;background:#fff;width:100%;">
          <div class="card-num">추천 ${i+1}</div>
          <div class="card-title">${esc(t.title)}</div>
          <div class="card-desc" style="font-size:12px;color:var(--text-sub);margin-top:4px;">클릭해서 자세히 보기 →</div>
        </div>
        ${i === topicDataStore.length - 1 ? `
        <div class="quick-btns" style="margin-top:8px;">
          <button class="quick-btn" onclick="retryTopics()">🔄 다시 추천받기</button>
        </div>` : ''}
      </div>`;
    document.getElementById('chatMessages').appendChild(wrap);
    scrollBot();
  });
}

function openTopicModal(idx) {
  const t = topicDataStore[idx];
  if(!t) return;

  const detailRows = Object.entries(t.details).map(([k,v]) =>
    `<div style="margin-bottom:18px;">
      <div style="font-size:11px;font-weight:700;color:var(--primary);margin-bottom:5px;">${esc(stripMarkdown(k))}</div>
      <div style="font-size:13px;line-height:1.7;color:var(--text);">${esc(stripMarkdown(v)).replace(/\n/g,'<br>')}</div>
    </div>`
  ).join('');

  const rawFallback = detailRows || `<div style="font-size:13px;line-height:1.7;color:var(--text);">${esc(stripMarkdown(t.raw)).replace(/\n/g,'<br>')}</div>`;

  document.getElementById('topicModalTitle').textContent = t.title;
  document.getElementById('topicModalBody').innerHTML = rawFallback;
  document.getElementById('topicModalConfirm').onclick = () => {
    closeTopicModal();
    confirmTopic(idx);
  };
  document.getElementById('topicModal').classList.add('show');
}

function closeTopicModal() {
  document.getElementById('topicModal').classList.remove('show');
}

async function confirmTopic(idx) {
  const t = topicDataStore[idx];
  if(!t) return;

  document.querySelectorAll('.recommend-card').forEach(c => {
    c.onclick = null;
    c.style.cursor = 'default';
    c.style.opacity = '0.5';
  });

  const card = document.getElementById(`topicCard${idx}`);
  if(card) {
    card.classList.add('selected');
    card.style.opacity = '1';
    card.style.border = '2px solid var(--primary)';
  }

  selectedTopic = t.title;
  renderTopicSummary(t);

  addUserMsg(`"${t.title}" 으로 확정할게요!`);
  addLoading();

  const detailText = Object.entries(t.details||{}).map(([k,v])=>`${k}: ${v}`).join('\n');

  try {
    const res = await apiFetch(`${API}/find-resources`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        session_id:sessionId,
        selected_topic:t.title,
        selected_topic_detail: detailText
      })
    });

    const data = await res.json();

    if(!res.ok) {
      throw new Error(data.detail || '자료 추천 중 오류가 발생했어요.');
    }

    if(data.call_count !== undefined) updateCallUsage(data.call_count, data.call_limit || 0);
    removeLoading();
    updateStep(5);
    latestPlanReport = data.plan_report || data.resources || '';
    latestPlanReportId = data.report_id || null;
    addAiMsg(`분석이 끝났어요. 팝업창에서 <strong>수행평가 설계 리포트</strong>를 확인하고, 아래 작성 폼에 직접 작성해 제출하세요.`, true, true);
    renderResources(data.resources, t.title);
    showPlanReportModal(t.title, latestPlanReport, latestPlanReportId);
    if(!document.querySelector('.writing-form')) showEvalInput();
  } catch(e) {
    removeLoading();
    addAiMsg(`❌ ${e.message || '자료 추천 중 오류가 발생했어요.'}`);
  }
}

function renderTopicSummary(t) {
  const detailRows = Object.entries(t.details||{}).map(([k,v]) => `
    <div style="margin-bottom:14px;">
      <div style="font-size:11px;font-weight:700;color:var(--primary);margin-bottom:4px;">${esc(stripMarkdown(k))}</div>
      <div style="font-size:13px;line-height:1.7;color:var(--text);">${esc(stripMarkdown(v)).replace(/\n/g,'<br>')}</div>
    </div>`).join('');

  document.getElementById('topicSummaryContent').innerHTML = `
    <div style="margin-bottom:16px;">
      <div style="font-size:13px;color:var(--text-sub);margin-bottom:6px;">선택 주제</div>
      <div style="font-size:15px;font-weight:700;color:var(--text);line-height:1.5;">${esc(t.title)}</div>
    </div>
    <div class="meta-tags" style="margin-bottom:16px;">
      <span class="meta-tag tag-grade">${studentInfo?.grade||''}</span>
      <span class="meta-tag tag-subject">${studentInfo?.subject||''}</span>
      <span class="meta-tag tag-career">${studentInfo?.career||''}</span>
    </div>
    <hr style="border:none;border-top:1px solid var(--border);margin-bottom:16px;">
    ${detailRows || `<div style="font-size:13px;color:var(--text-sub);">세부 내용이 없어요.</div>`}`;

  document.querySelector('.right-panel').style.display = 'flex';
  document.getElementById('panelTopicSummary').style.display = 'flex';
}

function stripMarkdown(text) {
  return text
    .replace(/#{1,6}\s*/g, '')
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/^\s*[-*]\s+/gm, '')
    .replace(/^---+$/gm, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
}

function renderResources(text, topic) {
  let html = `
    <div style="margin-bottom:16px;">
      <div style="font-size:13px;color:var(--text-sub);margin-bottom:6px;">선택 주제</div>
      <div style="font-size:14px;font-weight:700;margin-bottom:8px;">${esc(topic)}</div>
      <div class="meta-tags">
        <span class="meta-tag tag-grade">${studentInfo?.grade||''}</span>
        <span class="meta-tag tag-subject">${studentInfo?.subject||''}</span>
        <span class="meta-tag tag-career">${studentInfo?.career||''}</span>
      </div>
    </div>
    <hr style="border:none;border-top:1px solid var(--border);margin-bottom:16px;">`;

  const lines = text.split('\n');
  let cards = [];
  let cur = null;
  for(const line of lines) {
    const headerMatch = line.match(/^(?:자료|추천)\s*\d+\s*[:：.]?\s*(.*)$/);
    const titleLineMatch = line.trim().match(/^\d+\.\s*제목[:：]\s*(.*)/);
    if(headerMatch) {
      if(cur) cards.push(cur);
      cur = { title: headerMatch[1].trim(), items: {}, lastKey: '' };
    } else if(titleLineMatch) {
      if(cur) cards.push(cur);
      cur = { title: '', items: { '제목': titleLineMatch[1].trim() }, lastKey: '제목' };
    } else if(cur) {
      const numMatch = line.trim().match(/^\d+\.\s*(.+?)[:：]\s*(.*)/);
      if(numMatch) {
        cur.lastKey = numMatch[1].trim();
        cur.items[cur.lastKey] = numMatch[2].trim();
      } else if(cur.lastKey && line.trim() && !line.match(/^[-—]{2,}$/) && !line.match(/^[⚠️]|^주의/)) {
        cur.items[cur.lastKey] += ' ' + line.trim();
      } else if(line.match(/^[-—]{3,}$/)) {
        if(cur) { cards.push(cur); cur = null; }
      }
    }
  }
  if(cur) cards.push(cur);
  cards = cards.filter(c => Object.keys(c.items).length > 0 || c.title.trim().length > 0);

  if(cards.length > 0) {
    cards.forEach((card, i) => {
      const titleText = stripMarkdown(card.items['제목'] || card.title || '');
      const bodyItems = Object.entries(card.items).filter(([k]) => k !== '제목');
      const bodyHtml = bodyItems.map(([k, v]) =>
        `<div style="margin-bottom:10px;">
          <span style="font-size:11px;font-weight:700;color:var(--primary);">${esc(stripMarkdown(k))}</span>
          <div style="font-size:13px;color:var(--text);line-height:1.6;margin-top:2px;">${esc(stripMarkdown(v))}</div>
        </div>`
      ).join('');
      html += `<div class="resource-card">
        <span class="resource-badge">자료 ${i+1}</span>
        ${titleText ? `<div class="resource-title" style="margin:8px 0;">${esc(titleText)}</div>` : ''}
        <div style="margin-top:8px;">${bodyHtml}</div>
      </div>`;
    });
  } else {
    html += `<div style="font-size:13px;line-height:1.8;color:var(--text);">${esc(stripMarkdown(text)).replace(/\n/g,'<br>')}</div>`;
  }
  document.getElementById('resourceContent').innerHTML = html;
}

function showPlanReportModal(topic, reportText, reportId=null) {
  const title = `${topic || '수행평가'} 설계 리포트`;
  document.getElementById('reportModalTitle').textContent = '📘 수행평가 설계 리포트';
  document.getElementById('reportModalBody').innerHTML = buildReportHtml({
    title,
    type: 'plan',
    content: reportText || '리포트 내용이 없습니다.',
    reportId
  });
  document.getElementById('reportModal').classList.add('show');
}

function showEvaluationReportModal(topic, reportText, reportId=null) {
  const title = `${topic || '수행평가'} 평가 리포트`;
  document.getElementById('reportModalTitle').textContent = '📊 수행평가 평가 리포트';
  document.getElementById('reportModalBody').innerHTML = buildReportHtml({
    title,
    type: 'evaluation',
    content: reportText || '평가 리포트 내용이 없습니다.',
    reportId
  });
  document.getElementById('reportModal').classList.add('show');
}

function buildReportHtml({title, type, content, reportId}) {
  const typeLabel = type === 'evaluation' ? '평가 리포트' : '설계 리포트';
  return `
    <div class="report-layout">
      <div class="report-paper" id="activeReportPaper">
        <div class="report-paper-title">${esc(title)}</div>
        <div class="report-paper-meta">
          <span class="meta-tag tag-grade">${esc(studentInfo?.grade || '')}</span>
          <span class="meta-tag tag-subject">${esc(studentInfo?.subject || '')}</span>
          <span class="meta-tag tag-career">${esc(studentInfo?.career || '')}</span>
          <span class="meta-tag tag-subject">${typeLabel}</span>
        </div>
        <div class="report-text">${esc(stripMarkdown(content)).replace(/\n/g,'<br>')}</div>
      </div>
      <aside class="report-actions">
        <div class="report-action-card">
          <div style="font-size:13px;font-weight:800;margin-bottom:8px;">리포트 관리</div>
          <div style="font-size:12px;color:var(--text-sub);line-height:1.6;margin-bottom:12px;">현재 리포트는 서버에 저장되어 있으며, 목록에서 다시 열 수 있습니다.</div>
          <button class="report-action-btn" onclick="downloadCurrentReportPdf('${escJs(title)}')">PDF로 다운로드</button>
          <button class="report-action-btn secondary" style="margin-top:8px;" onclick="openReportListModal()">저장 리포트 목록</button>
        </div>
      </aside>
    </div>`;
}

function closeReportModal() {
  document.getElementById('reportModal').classList.remove('show');
}

function downloadCurrentReportPdf(title='수행평가 리포트') {
  const paper = document.getElementById('activeReportPaper');
  if(!paper) return;
  const win = window.open('', '_blank');
  const html = `<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>${esc(title)}</title>
    <style>
      @page { size: A4; margin: 18mm; }
      body { font-family:'Noto Sans KR','Malgun Gothic',Arial,sans-serif; color:#1a1f3c; }
      .report-paper-title { font-size:22px; font-weight:900; margin-bottom:12px; }
      .report-paper-meta { margin-bottom:18px; display:flex; gap:8px; flex-wrap:wrap; }
      .meta-tag { border:1px solid #d8def2; border-radius:999px; padding:4px 10px; font-size:11px; font-weight:700; }
      .report-text { font-size:12.5px; line-height:1.85; }
    </style></head><body>${paper.innerHTML}</body></html>`;
  win.document.write(html);
  win.document.close();
  win.focus();
  setTimeout(() => win.print(), 350);
}

function showEvalInput() {
  const uid = Date.now();
  const topicValue = selectedTopic || '';
  addAiMsg(`이제 아래 폼에 수행평가를 직접 작성해 제출하세요. 작성한 내용은 평가 후 리포트로 저장됩니다. 📋<br>
    <div class="writing-form" id="writingForm_${uid}">
      <div class="writing-form-title">수행평가 작성 폼</div>
      <div class="writing-field">
        <label class="writing-label">주제</label>
        <input class="writing-input writing-topic" value="${esc(topicValue)}" placeholder="수행평가 주제를 입력하세요">
      </div>
      <div class="writing-field">
        <label class="writing-label">서론</label>
        <textarea class="writing-textarea writing-intro" placeholder="탐구 동기, 문제 제기, 주제 선정 이유를 학생 본인의 언어로 작성하세요."></textarea>
      </div>
      <div class="writing-field">
        <label class="writing-label">본론</label>
        <textarea class="writing-textarea body writing-body" placeholder="교과 개념, 자료 근거, 분석 과정, 자신의 해석을 중심으로 작성하세요."></textarea>
      </div>
      <div class="writing-field">
        <label class="writing-label">결론</label>
        <textarea class="writing-textarea writing-conclusion" placeholder="탐구 결과, 느낀 점, 진로 또는 후속 탐구 방향을 작성하세요."></textarea>
      </div>
      <div style="display:flex;gap:8px;align-items:center;justify-content:flex-end;margin-top:10px;">
        <button class="btn-primary" onclick="evaluateSubmission('${uid}')">📊 제출하고 평가 리포트 받기</button>
      </div>
    </div>`, true);
}

async function evaluateSubmission(uid) {
  const form = document.getElementById(`writingForm_${uid}`);
  if(!form) {
    showToast('작성 폼을 찾을 수 없습니다.');
    return;
  }

  const topic = form.querySelector('.writing-topic')?.value?.trim() || selectedTopic || '';
  const intro = form.querySelector('.writing-intro')?.value?.trim() || '';
  const body = form.querySelector('.writing-body')?.value?.trim() || '';
  const conclusion = form.querySelector('.writing-conclusion')?.value?.trim() || '';

  if(!topic || !intro || !body || !conclusion) {
    showToast('주제, 서론, 본론, 결론을 모두 입력해주세요!');
    return;
  }

  const text = `주제: ${topic}

서론
${intro}

본론
${body}

결론
${conclusion}`;

  addUserMsg(`${topic} 수행평가를 제출합니다.`);
  addLoading();

  try {
    const res = await apiFetch(`${API}/evaluate-text`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        session_id: sessionId,
        submission_text: text,
        confirm_submit: true
      })
    });

    const data = await res.json();

    if(!res.ok) {
      throw new Error(data.detail || '평가 중 오류가 발생했습니다.');
    }

    if(data.call_count !== undefined) {
      updateCallUsage(data.call_count, data.call_limit || 0);
    }

    latestEvaluationReportId = data.report_id || null;
    removeLoading();
    addAiMsg('평가 완료! <strong>평가 리포트</strong>가 저장됐어요. 오른쪽 평가 결과 탭과 저장 리포트 목록에서 다시 확인할 수 있습니다 📊', true, true);
    renderEvaluation(data.evaluation);
    showEvaluationReportModal(selectedTopic || topic, data.evaluation, latestEvaluationReportId);

  } catch(e) {
    removeLoading();
    addAiMsg(`❌ ${e.message || '평가 중 오류가 발생했어요.'}`);
  }
}

function renderEvaluation(text) {
  text = stripMarkdown(text);
  document.getElementById('evalContent').innerHTML = `
    <div style="margin-bottom:16px;">
      <div style="font-size:13px;color:var(--text-sub);margin-bottom:4px;">평가 주제</div>
      <div style="font-size:14px;font-weight:700;">${esc(selectedTopic||'')}</div>
    </div>
    <hr style="border:none;border-top:1px solid var(--border);margin-bottom:16px;">
    <div class="eval-section"><div class="eval-text">${esc(text).replace(/\n/g,'<br>')}</div></div>`;
}

async function retryTopics() {
  if(!studentInfo) return;

  addUserMsg('다른 주제로 다시 추천해주세요!');
  addLoading();

  try {
    const res = await apiFetch(`${API}/recommend-topics`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
  session_id: sessionId,
  grade: studentInfo?.gradeWithSemester || studentInfo?.grade || '고등학생',
  subject: studentInfo?.subject || '국어',
  school_type: studentInfo?.schoolType || '일반고',
  desired_career: studentInfo?.career || '',
  previous_topic: studentInfo?.prevTopic || '',
  assessment_info: assessmentInfo || null
})
    });

    const data = await res.json();

    if(!res.ok) {
      throw new Error(data.detail || '오류가 발생했어요.');
    }

    if(data.call_count !== undefined) updateCallUsage(data.call_count, data.call_limit || 0);
    removeLoading();
    showTopicCards(data.topics);
  } catch(e) {
    removeLoading();
    addAiMsg(`❌ ${e.message || '오류가 발생했어요.'}`);
  }
}

async function sendMessage() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if(!text) return;
  input.value = ''; autoResize(input);
  addUserMsg(text);

  const positiveKeywords = ['좋다','좋아','맘에','마음에','완벽','최고','굿','good','ㅋㅋ','ㅎㅎ','감사','고마','잘됐','잘 됐','선택','확정','이걸로'];
  const howKeywords = ['어떻게','어떻게 써','어떻게 쓰','방법','작성','써야','쓰면','시작','뭐부터','모르겠','막막','도움'];

  const lower = text.toLowerCase();
  const isPositive = positiveKeywords.some(k => lower.includes(k));
  const isHow = howKeywords.some(k => lower.includes(k));

  if(isPositive && !isHow) {
    addAiMsg('감사해요! 선택하신 주제로 최선을 다해 작성해보세요 😊');
    return;
  }
  if(isHow) {
    addAiMsg('위에서 추천드린 자료들을 참고해서 본인의 언어로 작성해보세요!');
    return;
  }
  addAiMsg('수행평가와 관련된 질문을 해주세요 😊');
}

async function openReportListModal() {
  if(!sessionId) {
    showToast('로그인이 필요합니다.');
    return;
  }
  const body = document.getElementById('reportListModalBody');
  body.innerHTML = '<div class="scrap-empty">저장 리포트를 불러오는 중...</div>';
  document.getElementById('reportListModal').classList.add('show');

  try {
    const res = await apiFetch(`${API}/report-list`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ session_id: sessionId })
    });
    const data = await res.json();
    if(!res.ok) throw new Error(data.detail || '목록 조회 실패');

    const reports = data.reports || [];
    if(!reports.length) {
      body.innerHTML = '<div class="scrap-empty">아직 저장된 리포트가 없습니다.</div>';
      return;
    }

    body.innerHTML = reports.map(r => {
      const type = r.report_type === 'evaluation' ? '평가 리포트' : '설계 리포트';
      const date = r.created_at ? new Date(r.created_at).toLocaleString('ko-KR') : '';
      return `<div class="report-list-item" onclick="openSavedReport('${r.id}')">
        <div class="report-list-title">${esc(r.title || r.selected_topic || '수행평가 리포트')}</div>
        <div class="report-list-meta">
          <span>${esc(type)}</span>
          <span>${esc(r.grade || '')}</span>
          <span>${esc(r.subject || '')}</span>
          <span>${esc(r.career || '')}</span>
          <span>${esc(date)}</span>
        </div>
      </div>`;
    }).join('');
  } catch(e) {
    body.innerHTML = `<div class="scrap-empty">${esc(e.message || '리포트 목록을 불러오지 못했습니다.')}</div>`;
  }
}

function closeReportListModal() {
  document.getElementById('reportListModal').classList.remove('show');
  document.querySelectorAll('.sidebar-item').forEach(i=>i.classList.remove('active'));
  document.querySelector('.sidebar-item').classList.add('active');
}

async function openSavedReport(reportId) {
  if(!sessionId || !reportId) return;

  try {
    const res = await apiFetch(`${API}/report-get`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ session_id: sessionId, report_id: reportId })
    });
    const data = await res.json();
    if(!res.ok) throw new Error(data.detail || '리포트 조회 실패');

    const r = data.report;
    closeReportListModal();
    document.getElementById('reportModalTitle').textContent = r.report_type === 'evaluation' ? '📊 수행평가 평가 리포트' : '📘 수행평가 설계 리포트';
    document.getElementById('reportModalBody').innerHTML = buildReportHtml({
      title: r.title || r.selected_topic || '수행평가 리포트',
      type: r.report_type,
      content: r.report_content || r.evaluation_result || '',
      reportId: r.id
    });
    document.getElementById('reportModal').classList.add('show');
  } catch(e) {
    showToast(e.message || '리포트를 열지 못했습니다.');
  }
}

// ───────────────────────────────────────
// 유틸
// ───────────────────────────────────────
async function apiFetch(url, options={}, timeoutMs=120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timer);
    return res;
  } catch(e) {
    clearTimeout(timer);
    if(e.name==='AbortError') throw new Error('응답 시간이 너무 길어요. 잠시 후 다시 시도해주세요.');
    throw e;
  }
}

function switchTab(tab) {
  document.querySelectorAll('.header-tab').forEach((t,i) => t.classList.toggle('active', ['chat','resources','eval'][i]===tab));
  const hasTopic = !!selectedTopic;
  const rightPanel = document.querySelector('.right-panel');

  if(tab === 'chat') {
    document.getElementById('panelChat').style.display = 'flex';
    document.getElementById('panelTopicSummary').style.display = hasTopic ? 'flex' : 'none';
    document.getElementById('panelResources').style.display = 'none';
    document.getElementById('panelEval').style.display = 'none';
    rightPanel.classList.remove('fullwidth');
    rightPanel.style.display = hasTopic ? 'flex' : 'none';
  } else {
    document.getElementById('panelChat').style.display = 'none';
    document.getElementById('panelTopicSummary').style.display = 'none';
    document.getElementById('panelResources').style.display = tab==='resources' ? 'flex' : 'none';
    document.getElementById('panelEval').style.display = tab==='eval' ? 'flex' : 'none';
    rightPanel.classList.add('fullwidth');
    rightPanel.style.display = 'flex';
  }
}

function updateStep(step) {
  for(let i=1;i<=5;i++) {
    const el = document.getElementById(`step${i}`);
    el.classList.remove('done','current');
    if(i<step) { el.classList.add('done'); el.querySelector('.step-dot').textContent='✓'; }
    else if(i===step) el.classList.add('current');
  }
}

function addAiMsg(html, isHTML=false, scrapable=false) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-ai';
  const content = isHTML ? html : esc(html).replace(/\n/g,'<br>');
  wrap.innerHTML = `<div class="ai-avatar">🤖</div><div><div class="ai-name">위닝AI 수행평가 서비스</div><div class="ai-bubble">${content}</div></div>`;
  document.getElementById('chatMessages').appendChild(wrap);
  if(scrapable) addScrapBtn(wrap, wrap.querySelector('.ai-bubble').innerText);
  scrollBot();
  return wrap;
}

function addUserMsg(text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg-user';
  wrap.innerHTML = `<div class="user-bubble">${esc(text).replace(/\n/g,'<br>')}</div>`;
  document.getElementById('chatMessages').appendChild(wrap);
  scrollBot();
}

function addLoading() {
  const wrap = document.createElement('div');
  wrap.className='msg-ai'; wrap.id='loadingMsg';
  wrap.innerHTML=`<div class="ai-avatar">🤖</div><div><div class="ai-name">위닝AI 수행평가 서포터</div><div class="ai-bubble"><div class="loading-bubble"><div class="loading-dot"></div><div class="loading-dot"></div><div class="loading-dot"></div></div></div></div>`;
  document.getElementById('chatMessages').appendChild(wrap);
  scrollBot();
}

function removeLoading() { const e=document.getElementById('loadingMsg'); if(e) e.remove(); }
function scrollBot() { const e=document.getElementById('chatMessages'); e.scrollTop=e.scrollHeight; }
function esc(t) { return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function escJs(t) { return String(t).replace(/\\/g,'\\\\').replace(/'/g,"\\'").replace(/\n/g,' '); }
function showToast(msg) { const t=document.getElementById('toast'); t.textContent=msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),3000); }
function handleKey(e) { if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); sendMessage(); } }
function autoResize(el) { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,120)+'px'; }

// 스크랩
let scraps = JSON.parse(localStorage.getItem('scraps')||'[]');
function updateScrapCount() {
  const badge=document.getElementById('scrapCount');
  if(scraps.length>0){badge.textContent=scraps.length;badge.style.display='inline';}
  else badge.style.display='none';
}
function addScrapBtn(msgEl, content) {
  const btn=document.createElement('button');
  btn.innerHTML='📌 스크랩';
  btn.style.cssText='margin-top:8px;padding:5px 12px;border:1px solid var(--border);border-radius:20px;font-size:11px;cursor:pointer;background:#fff;color:var(--text-sub);font-family:"Noto Sans KR",sans-serif;transition:all 0.2s;';
  btn.onmouseover=()=>{btn.style.borderColor='var(--primary)';btn.style.color='var(--primary)';};
  btn.onmouseout=()=>{if(!btn.dataset.saved){btn.style.borderColor='var(--border)';btn.style.color='var(--text-sub)';}};
  btn.onclick=()=>{
    const scrap={id:Date.now(),content,subject:studentInfo?.subject||'',career:studentInfo?.career||'',date:new Date().toLocaleDateString('ko-KR')};
    scraps.unshift(scrap);localStorage.setItem('scraps',JSON.stringify(scraps));updateScrapCount();
    btn.innerHTML='✅ 스크랩됨';btn.style.borderColor='#22c55e';btn.style.color='#16a34a';btn.dataset.saved='true';
    showToast('📌 스크랩에 저장됐어요!');
  };
  msgEl.querySelector('.ai-bubble').after(btn);
}
function openScrapModal() {
  const body=document.getElementById('scrapModalBody');
  body.innerHTML=scraps.length===0?'<div class="scrap-empty">아직 스크랩한 내용이 없어요</div>':
    scraps.map(s=>`<div class="scrap-item"><button class="scrap-delete" onclick="deleteScrap(${s.id})">✕</button><div class="scrap-item-title">${s.subject?`[${esc(s.subject)}] `:''}${s.career?esc(s.career):'저장된 내용'}</div><div class="scrap-item-preview">${esc(s.content.slice(0,120))}${s.content.length>120?'...':''}</div><div class="scrap-item-date">📅 ${s.date}</div></div>`).join('');
  document.getElementById('scrapModal').classList.add('show');
}
function deleteScrap(id) {
  scraps=scraps.filter(s=>s.id!==id);localStorage.setItem('scraps',JSON.stringify(scraps));updateScrapCount();openScrapModal();showToast('🗑️ 삭제됐어요.');
}
function closeScrapModal(e) { if(e.target===document.getElementById('scrapModal')) closeScrapModalBtn(); }
function closeScrapModalBtn() { document.getElementById('scrapModal').classList.remove('show'); document.querySelectorAll('.sidebar-item').forEach(i=>i.classList.remove('active')); document.querySelector('.sidebar-item').classList.add('active'); }
function setActiveMenu(el) { document.querySelectorAll('.sidebar-item').forEach(i=>i.classList.remove('active')); el.classList.add('active'); }

async function init() {
  // Vercel API는 서버리스 함수라 별도 서버 깨우기 과정이 필요 없습니다.
  const banner = document.getElementById('connectingBanner');
  if(banner) banner.classList.remove('show');
}
init();
updateScrapCount();
