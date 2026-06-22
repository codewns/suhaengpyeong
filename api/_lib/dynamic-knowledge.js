import { winningSupabaseAdmin } from './supabase.js';
import { WINNING_KNOWLEDGE_TABLE } from './config.js';

function normalizeSubject(subject = '') {
  const s = String(subject || '').replace(/\s/g, '');

  if (
    s.includes('국어') ||
    s.includes('문학') ||
    s.includes('독서') ||
    s.includes('작문') ||
    s.includes('화법') ||
    s.includes('언어') ||
    s.includes('매체')
  ) return '국어';

  if (
    s.includes('수학') ||
    s.includes('대수') ||
    s.includes('미적분') ||
    s.includes('기하') ||
    s.includes('확률') ||
    s.includes('통계') ||
    s.includes('이산')
  ) return '수학';

  if (
    s.includes('영어') ||
    s.includes('영미')
  ) return '영어';

  if (s.includes('한국사')) return '한국사';

  if (
    s.includes('사회') ||
    s.includes('역사') ||
    s.includes('윤리') ||
    s.includes('지리') ||
    s.includes('경제') ||
    s.includes('정치') ||
    s.includes('법') ||
    s.includes('세계시민') ||
    s.includes('국제관계') ||
    s.includes('기후변화')
  ) return '사회';

  if (
    s.includes('과학') ||
    s.includes('생명') ||
    s.includes('화학') ||
    s.includes('물리') ||
    s.includes('지구') ||
    s.includes('세포') ||
    s.includes('유전') ||
    s.includes('물질과에너지') ||
    s.includes('화학반응') ||
    s.includes('전자기') ||
    s.includes('양자') ||
    s.includes('역학') ||
    s.includes('행성') ||
    s.includes('우주')
  ) return '과학';

  if (
    s.includes('체육') ||
    s.includes('스포츠') ||
    s.includes('운동')
  ) return '체육';

  if (
    s.includes('음악') ||
    s.includes('미술') ||
    s.includes('연극') ||
    s.includes('영화') ||
    s.includes('사진') ||
    s.includes('무용') ||
    s.includes('드로잉')
  ) return '예술';

  if (
    s.includes('기술') ||
    s.includes('가정') ||
    s.includes('공학') ||
    s.includes('로봇') ||
    s.includes('지식재산') ||
    s.includes('생활과학')
  ) return '기술·가정';

  if (
    s.includes('정보') ||
    s.includes('데이터') ||
    s.includes('소프트웨어') ||
    s.includes('인공지능')
  ) return '정보';

  if (
    s.includes('독일어') ||
    s.includes('프랑스어') ||
    s.includes('스페인어') ||
    s.includes('중국어') ||
    s.includes('일본어') ||
    s.includes('러시아어') ||
    s.includes('아랍어') ||
    s.includes('베트남어')
  ) return '제2외국어';

  if (
    s.includes('한문') ||
    s.includes('한자')
  ) return '한문';

  if (
    s.includes('진로와직업') ||
    s.includes('생태와환경') ||
    s.includes('철학') ||
    s.includes('논리') ||
    s.includes('심리') ||
    s.includes('교육의이해') ||
    s.includes('종교') ||
    s.includes('보건') ||
    s.includes('논술')
  ) return '교양';

  return subject || '국어';
}

function unique(values) {
  return [...new Set(values.map((v) => String(v || '').trim()).filter(Boolean))];
}

function scoreText(text, keywords) {
  const base = String(text || '').toLowerCase();
  let score = 0;

  for (const keyword of keywords) {
    const k = String(keyword || '').trim().toLowerCase();
    if (!k) continue;
    if (base.includes(k)) score += 3;
  }

  return score;
}

function rowToText(row, label = '위닝DB 항목') {
  return `
[${label}]
- 학년: ${row.grade || ''}
- 과목: ${row.subject || ''}
- 진로분야: ${row.career_field || ''}
- 자료명: ${row.title || ''}
- 출처: ${row.source || ''}
- 내용:
${row.content || ''}
`.trim();
}

function packRows(rows, maxChars, label) {
  const pieces = [];
  let total = 0;

  for (const row of rows) {
    const piece = rowToText(row, label);
    if (total + piece.length > maxChars) break;
    pieces.push(piece);
    total += piece.length;
  }

  return pieces;
}

export async function loadDynamicAssessmentKnowledge({
  grade = '고등학생',
  subject = '국어',
  career = '',
  selectedTopic = '',
  assessmentInfo = '',
  purpose = 'topic',
  maxItems = 6,
  maxChars = 4500,
  includeOtherSubjects = false
}) {
  if (!winningSupabaseAdmin) {
    return '홈페이지 Supabase 위닝DB 연결 환경변수 없음';
  }

  const knowledgeType = purpose === 'resource' ? 'verified_resource' : 'topic_pattern';
  const normalizedSubject = normalizeSubject(subject);

  const gradeText = String(grade || '').trim();

const baseGrade = gradeText.includes('고1') ? '고1'
  : gradeText.includes('고2') ? '고2'
  : gradeText.includes('고3') ? '고3'
  : gradeText;

const gradeAlias = baseGrade === '고1' ? '1학년'
  : baseGrade === '고2' ? '2학년'
  : baseGrade === '고3' ? '3학년'
  : '';

const legacySubject = normalizedSubject === '사회' ? '사회역사' : '';

const subjectParts = String(subject || '')
  .split('/')
  .map(v => v.trim())
  .filter(Boolean);

const gradeCandidates = unique([
  gradeText,
  baseGrade,
  gradeAlias,
  '공통',
  '전체',
  '고등학생'
]);

const subjectCandidates = unique([
  normalizedSubject,
  legacySubject,
  subject,
  ...subjectParts,
  '공통',
  '전체'
]);

  const keywords = unique([
    career,
    selectedTopic,
    normalizedSubject,
    subject,
    String(assessmentInfo || '').slice(0, 300)
  ]);

  try {
    const { data: currentRows = [], error: currentError } = await winningSupabaseAdmin
      .from(WINNING_KNOWLEDGE_TABLE)
      .select('id, grade, subject, knowledge_type, career_field, title, content, source, memo, created_at')
      .eq('is_active', true)
      .eq('knowledge_type', knowledgeType)
      .in('grade', gradeCandidates)
      .in('subject', subjectCandidates)
      .order('created_at', { ascending: false })
      .limit(80);

    if (currentError) throw currentError;

    const currentScored = currentRows
      .map((row) => {
        const blob = [row.career_field, row.title, row.content, row.source, row.memo].join(' ');
        return { row, score: scoreText(blob, keywords) };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, maxItems)
      .map((item) => item.row);

    const pieces = [];

    if (currentScored.length) {
      pieces.push('[현재 과목 위닝DB 후보]');
      pieces.push(...packRows(currentScored, maxChars, '위닝DB 항목'));
    }

    if (includeOtherSubjects) {
      const { data: allRows = [], error: otherError } = await winningSupabaseAdmin
        .from(WINNING_KNOWLEDGE_TABLE)
        .select('id, grade, subject, knowledge_type, career_field, title, content, source, memo, created_at')
        .eq('is_active', true)
        .eq('knowledge_type', knowledgeType)
        .in('grade', gradeCandidates)
        .order('created_at', { ascending: false })
        .limit(160);

      if (otherError) throw otherError;

      const currentSubjectSet = new Set(subjectCandidates);
      const otherScored = allRows
        .filter((row) => !currentSubjectSet.has(row.subject))
        .map((row) => {
          const blob = [row.career_field, row.title, row.content, row.source, row.memo].join(' ');
          return { row, score: scoreText(blob, [career, selectedTopic]) };
        })
        .filter((item) => item.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 3)
        .map((item) => item.row);

      if (otherScored.length) {
        pieces.push('\n[다른 과목 연계 후보 - AI가 연계 가능할 때만 사용]');
        pieces.push(...packRows(otherScored, Math.max(1200, Math.floor(maxChars * 0.45)), '다른 과목 후보'));
      }
    }

    return pieces.join('\n\n').trim() || '관련 위닝DB 항목 없음';
  } catch (error) {
    console.error('홈페이지 Supabase 위닝DB 지식 조회 오류:', error);
    return '홈페이지 Supabase 위닝DB 지식 조회 실패 또는 테이블 미생성';
  }
}
