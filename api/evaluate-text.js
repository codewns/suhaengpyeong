import { CORE_PRINCIPLES } from './_lib/config.js';
import {
  getSession,
  updateSession,
  dbSaveConversation,
  incrementCallCount
} from './_lib/sessions.js';
import { callText } from './_lib/gemini.js';
import { saveAssessmentReport } from './_lib/reports.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    const {
      session_id,
      submission_text,
      confirm_submit
    } = req.body || {};

    if (!session_id) {
      return res.status(400).json({ detail: 'session_id가 필요합니다.' });
    }

    if (confirm_submit !== true) {
      return res.status(400).json({ detail: '제출물 평가 버튼을 통해서만 평가할 수 있습니다.' });
    }

    if (!submission_text || !submission_text.trim()) {
      return res.status(400).json({ detail: '평가할 제출물이 필요합니다.' });
    }

    if (submission_text.trim().length < 100) {
      return res.status(400).json({
        detail: '제출물이 너무 짧습니다. 실제 수행평가 제출물을 입력한 뒤 평가해주세요.'
      });
    }

    const session = await getSession(session_id);
if (!session?.student_code) {
  return res.status(401).json({ detail: '로그인이 필요합니다.' });
}

const schoolType = session.school_type || '일반고';
    
    const usage = await incrementCallCount(session.student_code);

    if (!usage.allowed) {
      return res.status(429).json({
        detail: `이용 횟수를 모두 사용했어요. (사용: ${usage.count}/${usage.limit}회)`
      });
    }

    const assessmentText = session.assessment_info || '평가 기준 정보 없음';

    const system = `
${CORE_PRINCIPLES}

당신은 고등학교 수행평가 채점 전문가입니다.
이 단계에서는 주제 추천용 데이터와 자료 추천용 데이터를 사용하지 않습니다.
오직 수행평가 안내문의 평가 기준, 선택 주제, 학생 제출물을 기준으로 평가합니다.

출력 규칙:
1. 마크다운 기호를 쓰지 않는다.
2. 항목은 숫자 번호로 구분한다.
3. 구체적 근거와 보완 방향을 제시한다.
4. 표절 위험이 있으면 반드시 지적한다.
5. 평가 기준과 연결해서 판단한다.
6. 실제 제출물이 아닌 주제명, 자료 추천 결과, 빈 문장, 단순 메모처럼 보이는 경우 평가하지 말고 제출물 부족으로 안내한다.
7. 학교 유형은 평가 기준이 아니라 참고 정보로만 활용한다.
8. 과목명이 구체적으로 입력되어 있으면 해당 과목의 수준과 성격을 기준으로 판단한다. 단, 수행평가 안내문의 평가 기준이 더 구체적이면 안내문을 우선한다.
9. 전문교과 과목이라도 학생이 실제 수행평가에서 다룰 수 있는 범위를 기준으로 평가한다.

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
`.trim();

    const userMsg = `
[평가 기준]
${assessmentText}

[선택 주제]
${session.selected_topic || '미입력'}

[희망 진로]
${session.career || '미입력'}

[학년/학기]
${session.grade || '미입력'}

[학교 유형]
${schoolType}

[선택 과목]
${session.subject || '미입력'}

[이전에 했던 주제]
${session.previous_topic || '없음'}

[학생 제출물]
${submission_text}

작업:
수행평가 안내문의 평가 기준을 최우선으로 하여 제출물을 평가하세요.
내부 위닝DB 자료는 사용하지 말고, 제출물 자체와 평가 기준만 판단하세요.
`.trim();

    const result = await callText(system, userMsg);

    const updated = await updateSession(session_id, {
  evaluation: result
});

await dbSaveConversation(updated);

const savedReport = await saveAssessmentReport({
  student_code: session.student_code,
  student_name: session.student_name || '',
  report_type: 'evaluation',
  title: session.selected_topic || '수행평가 평가 리포트',
  grade: session.grade || '',
  subject: session.subject || '',
  career: session.career || '',
  selected_topic: session.selected_topic || '',
  report_content: result,
  submission_text,
  evaluation_result: result,
  meta: {
    previous_topic: session.previous_topic || '',
    assessment_info: session.assessment_info || ''
  }
});

return res.status(200).json({
  status: 'success',
  evaluation: result,
  report_id: savedReport?.id || null,
  call_count: usage.count,
  call_limit: usage.limit
});
  } catch (error) {
    console.error('evaluate text error:', error);
    return res.status(500).json({ detail: '평가 중 오류가 발생했습니다.' });
  }
}
