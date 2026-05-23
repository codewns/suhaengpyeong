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
    const knowledgeBase = loadKnowledgeByGradeSubject(grade, subject);

    const system = `
${CORE_PRINCIPLES}

당신은 고등학교 수행평가 주제 추천 전문가입니다.
아래의 학년·과목별 내부 지식 데이터를 최우선으로 참고해 학생 맞춤 주제를 추천하세요.

[학년·과목별 내부 지식 데이터]
${knowledgeBase}

출력 규칙:
1. *, **, ##, ### 같은 마크다운 기호를 절대 사용하지 않는다.
2. 영어 단어나 영어 표현을 괄호 안에 넣지 않는다.
3. 주제명은 반드시 수행평가에서 실제로 탐구할 수 있는 구체적인 한국어 주제명으로 작성한다.
4. 안내문에 없는 질문을 만들어내지 않는다.
5. 추천 3개가 끝나면 추가 안내를 쓰지 않는다.

반드시 아래 형식으로 3개 추천:

추천 1: (구체적인 한국어 주제명)
1. 핵심 내용:
2. 이전 주제와의 연결:
3. 추후 심화 방향:
4. 추천 이유:
5. 점수 강점:

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
`.trim();

    const userMsg = `
[학생 정보]
- 학년: ${grade}
- 과목: ${subject}
- 희망 진로: ${desired_career}
- 같은 과목에서 이전에 한 주제: ${previousTopic}

[수행평가 안내문]
${assessmentText}

위 정보를 바탕으로 맞춤 주제 3개를 추천해주세요.
특히 이전 주제가 있다면 그것을 심화·확장하는 방향을 최우선으로 고려하세요.
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
