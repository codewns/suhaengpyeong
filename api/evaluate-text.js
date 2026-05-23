import { CORE_PRINCIPLES } from './_lib/config.js';
import { loadKnowledgeByGradeSubject } from './_lib/knowledge.js';
import {
  getSession,
  updateSession,
  dbSaveConversation,
  incrementCallCount
} from './_lib/sessions.js';
import { callText } from './_lib/gemini.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    const { session_id, submission_text } = req.body || {};

    if (!session_id || !submission_text) {
      return res.status(400).json({ detail: '평가할 내용이 필요합니다.' });
    }

    const session = await getSession(session_id);

    if (!session?.student_code) {
      return res.status(401).json({ detail: '로그인이 필요합니다.' });
    }

    const usage = await incrementCallCount(session.student_code);

    if (!usage.allowed) {
      return res.status(429).json({
        detail: `이용 횟수를 모두 사용했어요. (사용: ${usage.count}/${usage.limit}회)`
      });
    }

    const knowledgeBase = loadKnowledgeByGradeSubject(session.grade || '고등학생', session.subject || '국어');
    const assessmentText = session.assessment_info || '평가 기준 정보 없음';

    const system = `
${CORE_PRINCIPLES}

당신은 고등학교 수행평가 채점 전문가입니다.
아래 학년·과목별 내부 지식 데이터와 수행평가 평가 기준을 바탕으로 학생 제출물을 평가하세요.

[학년·과목별 내부 지식 데이터]
${knowledgeBase}

출력 규칙:
1. 마크다운 기호를 쓰지 않는다.
2. 항목은 숫자 번호로 구분한다.
3. 구체적 근거와 보완 방향을 제시한다.
4. 표절 위험이 있으면 반드시 지적한다.
5. 평가 기준과 연결해서 판단한다.

평가 항목마다 아래 형식으로 작성:

항목명 (별점 ★로 표시)
1. 잘한 점:
2. 아쉬운 점:
3. 보완할 점:

표절 위험 문장
심화 탐구 연결성

종합 평가
예상 점수: X점 / 100점
총평:
`.trim();

    const userMsg = `
[평가 기준]
${assessmentText}

[선택 주제]
${session.selected_topic || '미입력'}

[희망 진로]
${session.career || '미입력'}

[학생 제출물]
${submission_text}
`.trim();

    const result = await callText(system, userMsg);

    const updated = await updateSession(session_id, {
      evaluation: result
    });

    await dbSaveConversation(updated);

    return res.status(200).json({
      status: 'success',
      evaluation: result,
      call_count: usage.count,
      call_limit: usage.limit
    });
  } catch (error) {
    console.error('evaluate text error:', error);
    return res.status(500).json({ detail: '평가 중 오류가 발생했습니다.' });
  }
}
