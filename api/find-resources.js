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
      selected_topic,
      selected_topic_detail = ''
    } = req.body || {};

    if (!session_id || !selected_topic) {
      return res.status(400).json({ detail: '선택 주제가 필요합니다.' });
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

    const system = `
당신은 수행평가 자료 추천 전문가입니다.

[내부 지식 데이터]
${knowledgeBase}

반드시 지켜야 할 규칙:
1. 내부 지식 데이터에 있는 자료와 방향을 최우선으로 활용한다.
2. 확인되지 않은 자료명이나 링크를 지어내지 않는다.
3. 실제 링크가 확실하지 않으면 링크 항목은 생략한다.
4. 자료는 참고용이며, 그대로 베끼지 말고 본인의 언어로 재구성해야 한다고 안내한다.
5. 최대 3개 자료만 추천한다.

출력 형식:

자료 1
1. 제목:
2. 유형:
3. 핵심 내용:
4. 수행평가 활용법:
5. 추후 심화 연결:
6. 진로 연계:
주의: 이 자료의 내용을 그대로 옮기지 말고, 반드시 본인의 언어로 재구성하세요.

자료 2
1. 제목:
2. 유형:
3. 핵심 내용:
4. 수행평가 활용법:
5. 추후 심화 연결:
6. 진로 연계:
주의: 이 자료의 내용을 그대로 옮기지 말고, 반드시 본인의 언어로 재구성하세요.

자료 3
1. 제목:
2. 유형:
3. 핵심 내용:
4. 수행평가 활용법:
5. 추후 심화 연결:
6. 진로 연계:
주의: 이 자료의 내용을 그대로 옮기지 말고, 반드시 본인의 언어로 재구성하세요.
`.trim();

    const userMsg = `
[선택 주제]
${selected_topic}

[선택 주제 상세]
${selected_topic_detail || '없음'}

[학생 정보]
- 학년: ${session.grade || '미입력'}
- 과목: ${session.subject || '미입력'}
- 희망 진로: ${session.career || '미입력'}
- 이전 주제: ${session.previous_topic || '없음'}

위 주제에 맞는 자료를 추천해주세요.
`.trim();

    const result = await callText(system, userMsg);

    const updated = await updateSession(session_id, {
      selected_topic,
      selected_topic_detail,
      resources: result
    });

    await dbSaveConversation(updated);

    return res.status(200).json({
      status: 'success',
      resources: result,
      call_count: usage.count,
      call_limit: usage.limit
    });
  } catch (error) {
    console.error('find resources error:', error);
    return res.status(500).json({ detail: '자료 추천 중 오류가 발생했습니다.' });
  }
}
