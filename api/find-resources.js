import { loadDynamicAssessmentKnowledge } from './_lib/dynamic-knowledge.js';
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

    const grade = session.grade || '고등학생';
    const subject = session.subject || '국어';
    const career = session.career || '';


    const dynamicResourceKnowledge = await loadDynamicAssessmentKnowledge({
      grade,
      subject,
      career,
      selectedTopic: selected_topic,
      assessmentInfo: session.assessment_info || '',
      purpose: 'resource',
      maxItems: 8,
      maxChars: 6500,
      includeOtherSubjects: false
    });

    const system = `
당신은 수행평가 자료 추천 및 설계 리포트 작성 전문가입니다.
이 단계에서는 자료 추천용 데이터만 사용합니다.
주제 추천용 데이터는 참고하지 않습니다.

[홈페이지 위닝 수행 자료 DB]
${dynamicResourceKnowledge}

반드시 지켜야 할 규칙:
1. 위닝 수행 자료 DB에 있는 검증 자료를 우선 사용한다.
2. 선택 주제와 직접 연결되는 자료만 추천한다.
3. 내부 지식 데이터나 위닝DB에 없는 자료명, 저자, 링크를 지어내지 않는다.
4. 자료가 부족하면 자료명 대신 검색 키워드를 제시한다.
5. 링크가 내부 데이터에 없으면 링크 항목을 생략한다.
6. 자료는 최대 3개만 추천한다.
7. 자료 내용은 그대로 베끼지 말고 학생의 언어로 재구성해야 한다고 안내한다.
8. Google Search를 사용하지 않았으므로 최신성 확인이 필요한 자료는 확인 필요라고 표시한다.

출력 형식:

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
`.trim();

    const userMsg = `
[선택 주제]
${selected_topic}

[선택 주제 상세]
${selected_topic_detail || '없음'}

[학생 정보]
- 학년: ${grade}
- 과목: ${subject}
- 희망 진로: ${career || '미입력'}
- 이전 주제: ${session.previous_topic || '없음'}

[수행평가 안내문 요약]
${session.assessment_info || '안내문 정보 없음'}

작업:
1. 선택 주제와 가장 관련 있는 검증 자료를 추천한다.
2. 위닝 수행 자료 DB에 적합한 자료가 있으면 우선 사용한다.
3. 검증 자료가 부족하면 검색 키워드 중심으로 보완한다.
4. 없는 링크나 자료명을 만들지 않는다.
`.trim();

    const result = await callText(system, userMsg);

    const updated = await updateSession(session_id, {
  selected_topic,
  selected_topic_detail,
  resources: result
});

await dbSaveConversation(updated);

const savedReport = await saveAssessmentReport({
  student_code: session.student_code,
  student_name: session.student_name || '',
  report_type: 'plan',
  title: selected_topic,
  grade,
  subject,
  career,
  selected_topic,
  report_content: result,
  meta: {
    selected_topic_detail
  }
});

return res.status(200).json({
  status: 'success',
  resources: result,
  plan_report: result,
  report_id: savedReport?.id || null,
  call_count: usage.count,
  call_limit: usage.limit
});
  } catch (error) {
    console.error('find resources error:', error);
    return res.status(500).json({ detail: '자료 추천 중 오류가 발생했습니다.' });
  }
}
