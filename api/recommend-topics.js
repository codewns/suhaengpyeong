import { CORE_PRINCIPLES, CROSS_SUBJECT_CONNECTION_GUIDE } from './_lib/config.js';
import { loadDynamicAssessmentKnowledge } from './_lib/dynamic-knowledge.js';
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
    const {
      session_id,
      grade = '고등학생',
      subject = '국어',
      desired_career = '',
      assessment_info = null,
      previous_topic = ''
    } = req.body || {};

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

    const assessmentText = assessment_info || session.assessment_info || '수행평가 안내문 정보 없음';
    const previousTopic = previous_topic || session.previous_topic || '없음';

  

    const dynamicTopicKnowledge = await loadDynamicAssessmentKnowledge({
      grade,
      subject,
      career: desired_career,
      selectedTopic: previousTopic,
      assessmentInfo: assessmentText,
      purpose: 'topic',
      maxItems: 6,
      maxChars: 4500,
      includeOtherSubjects: true
    });

    const system = `
${CORE_PRINCIPLES}

당신은 고등학교 수행평가 주제 추천 전문가입니다.
이 단계에서는 주제 추천용 데이터만 사용합니다.
자료 추천용 데이터나 평가용 데이터는 사용하지 않습니다.


[홈페이지 위닝 수행 주제 DB]
${dynamicTopicKnowledge}

[다른 과목 연계 판단 기준]
${CROSS_SUBJECT_CONNECTION_GUIDE}

다른 과목 선배 데이터 활용 규칙:
1. 다른 과목 후보는 반드시 먼저 연계 가능성을 판단한다.
2. 현재 과목의 수행평가 방식으로 재해석할 수 있을 때만 사용한다.
3. 억지 연계이면 사용하지 않는다.
4. 사용한 경우 "다른 과목 연계 포인트" 항목에서 어떤 과목의 어떤 흐름을 현재 과목 방식으로 바꾸었는지 설명한다.
5. 사용하지 않는 경우 굳이 언급하지 않는다.

출력 규칙:
1. *, **, ##, ### 같은 마크다운 기호를 절대 사용하지 않는다.
2. 영어 단어나 영어 표현을 괄호 안에 넣지 않는다.
3. 주제명은 반드시 수행평가에서 실제로 탐구할 수 있는 구체적인 한국어 주제명으로 작성한다.
4. 안내문에 없는 질문을 만들어내지 않는다.
5. 추천 3개가 끝나면 추가 안내를 쓰지 않는다.
6. 학생이 그대로 제출할 수 있는 완성문을 작성하지 않는다.

반드시 아래 형식으로 3개 추천:

추천 1: 구체적인 한국어 주제명
0. 선정 근거:
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 다른 과목 연계 포인트:
4. 추후 심화 방향:
5. 추천 이유:
6. 점수 강점:

추천 2: 구체적인 한국어 주제명
0. 선정 근거:
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 다른 과목 연계 포인트:
4. 추후 심화 방향:
5. 추천 이유:
6. 점수 강점:

추천 3: 구체적인 한국어 주제명
0. 선정 근거:
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 다른 과목 연계 포인트:
4. 추후 심화 방향:
5. 추천 이유:
6. 점수 강점:
`.trim();

    const userMsg = `
[학생 정보]
- 학년: ${grade}
- 과목: ${subject}
- 희망 진로: ${desired_career}
- 같은 과목에서 이전에 한 주제: ${previousTopic}

[수행평가 안내문]
${assessmentText}

작업:
1. 수행평가 안내문 조건을 최우선으로 반영한다.
2. 같은 과목 이전 주제가 있으면 추천 1은 반드시 심화·확장 주제로 제시한다.
3. 홈페이지 위닝 수행 주제 DB의 현재 과목 데이터는 적극 참고한다.
4. 다른 과목 선배 데이터는 연계 가능할 때만 선택적으로 활용한다.
5. 다른 과목 데이터를 활용하더라도 현재 과목의 과목성이 약해지면 안 된다.
6. 안내문에 없는 질문이나 자료를 임의로 만들지 않는다.
`.trim();

    const result = await callText(system, userMsg);

    const updated = await updateSession(session_id, {
      grade,
      subject,
      career: desired_career,
      previous_topic: previousTopic === '없음' ? '' : previousTopic,
      assessment_info: assessmentText,
      topics: result
    });

    await dbSaveConversation(updated);

    return res.status(200).json({
      status: 'success',
      topics: result,
      call_count: usage.count,
      call_limit: usage.limit
    });
  } catch (error) {
    console.error('recommend topics error:', error);
    return res.status(500).json({ detail: '주제 추천 중 오류가 발생했습니다.' });
  }
}
