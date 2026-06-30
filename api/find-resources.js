import { loadDynamicAssessmentKnowledge } from './_lib/dynamic-knowledge.js';
import {
  getSession,
  updateSession,
  dbSaveConversation,
  incrementCallCount
} from './_lib/sessions.js';
import { callText } from './_lib/gemini.js';
import { saveAssessmentReport } from './_lib/reports.js';

function ensureConclusionSection(text = '') {
  const result = String(text || '').trim();

  if (/결론\s*(작성|구성)\s*방향/.test(result)) {
    return result;
  }

  return `${result}\n\n7. 결론 구성 방향\n- 결론의 역할: 본론에서 분석한 내용을 단순 반복하지 말고, 선택 주제를 통해 확인한 핵심 의미를 정리한다.\n- 탐구 결과 정리 방식: 사용한 자료와 교과 개념을 바탕으로 무엇을 알게 되었는지 2~3가지로 압축한다.\n- 새롭게 알게 된 점: 기존에 막연히 알고 있던 내용과 탐구 후 달라진 이해를 비교한다.\n- 진로 또는 후속 탐구와 연결하는 방식: 희망 진로와 연결하되 과장하지 말고, 이후 더 확인해 볼 질문을 제시한다.\n- 피해야 할 점: 느낀 점만 길게 쓰거나, 본론에 없던 새로운 주장과 자료를 결론에서 갑자기 추가하지 않는다.`.trim();
}

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

    if (!session?.main_id) {
      return res.status(401).json({ detail: '로그인이 필요합니다.' });
    }

    const selectedSavedSession = await updateSession(session_id, {
      selected_topic,
      selected_topic_detail
    });

    await dbSaveConversation(selectedSavedSession);

    const usage = await incrementCallCount(selectedSavedSession.main_id);

    if (!usage.allowed) {
      return res.status(429).json({
        detail: `이용 횟수를 모두 사용했어요. (사용: ${usage.count}/${usage.limit}회)`
      });
    }

    const grade = selectedSavedSession.grade || '고등학생';
    const subject = selectedSavedSession.subject || '국어';
    const schoolType = selectedSavedSession.school_type || '일반고';
    const career = selectedSavedSession.career || '';

    const dynamicResourceKnowledge = await loadDynamicAssessmentKnowledge({
      grade,
      subject,
      career,
      selectedTopic: selected_topic,
      assessmentInfo: selectedSavedSession.assessment_info || '',
      purpose: 'resource',
      maxItems: 8,
      maxChars: 8000,
      includeOtherSubjects: false
    });

    const system = `
당신은 고등학생 수행평가 설계 리포트 작성 전문가입니다.

목표:
학생이 선택한 주제에 대해 "통합 수행평가 설계 리포트"를 작성한다.

중요:
1. 반드시 아래 1번부터 8번까지 모든 번호 항목을 빠짐없이 출력한다.
2. 특히 "7. 결론 구성 방향"은 절대 생략하지 않는다.
3. 전체 분량은 너무 길게 쓰지 말고, 핵심만 압축한다.
4. 본론 항목은 2~3개까지만 제시한다.
5. 서론/본론/결론은 완성문이 아니라 "구성 방향, 포함 요소, 전개 순서"로 쓴다.
6. 학생이 그대로 제출할 수 있는 완성문을 쓰지 않는다.
7. 추천 자료는 아래 위닝 수행 자료 DB에서 불러온 자료를 우선 사용한다.
8. 위닝 수행 자료 DB에 "출처 링크"가 있으면 반드시 그대로 출력한다.
9. 출처 링크가 없으면 "출처 링크: 확인 필요"라고 쓴다.
10. DB에 없는 자료명, 링크, 저자, 기관명을 지어내지 않는다.
11. 자료가 부족하면 자료명 대신 검색 키워드를 제시한다.
12. 출력에 *, **, ## 같은 마크다운 기호를 쓰지 않는다.

[홈페이지 위닝 수행 자료 DB]
${dynamicResourceKnowledge || '사용 가능한 내부 자료 없음'}

반드시 아래 형식대로 출력하라.

1. 최종 주제
- 주제명:
- 주제의 핵심 의미:
- 선택 과목과 연결되는 지점:
- 희망 진로와 연결되는 지점:

2. 추천 자료 및 활용 포인트
자료 1
- 자료명 또는 검색 키워드:
- 출처 정보:
- 출처 링크:
- 핵심 내용:
- 이 자료를 보고서에서 활용하는 방법:

자료 2
- 자료명 또는 검색 키워드:
- 출처 정보:
- 출처 링크:
- 핵심 내용:
- 이 자료를 보고서에서 활용하는 방법:

자료 3
- 자료명 또는 검색 키워드:
- 출처 정보:
- 출처 링크:
- 핵심 내용:
- 이 자료를 보고서에서 활용하는 방법:

3. 탐구 핵심 질문
- 핵심 질문 1:
- 핵심 질문 2:
- 핵심 질문 3:

4. 수행평가 전체 방향
- 보고서의 중심 주장:
- 분석 포인트:
- 교과 개념을 드러내는 방식:
- 학생의 해석이 꼭 들어가야 하는 부분:

5. 서론 구성 방향
- 서론의 역할:
- 반드시 포함할 내용:
- 주제 선정 동기 구성 방식:
- 도입부에서 다루면 좋은 문제의식:
- 피해야 할 점:

6. 본론 구성 방향
본론 항목 A. [역할이 드러나는 제목]
- 중심 내용:
- 연결할 교과 개념:
- 사용할 자료:
- 전개 순서:
- 학생이 직접 해석해야 하는 부분:
- 피해야 할 점:

본론 항목 B. [역할이 드러나는 제목]
- 중심 내용:
- 연결할 교과 개념:
- 사용할 자료:
- 전개 순서:
- 학생이 직접 해석해야 하는 부분:
- 피해야 할 점:

필요할 때만 본론 항목 C를 추가한다.

7. 결론 구성 방향
- 결론의 역할:
- 탐구 결과 정리 방식:
- 새롭게 알게 된 점:
- 진로 또는 후속 탐구와 연결하는 방식:
- 피해야 할 점:

8. 학생 작성 체크리스트
- 체크 1:
- 체크 2:
- 체크 3:
- 체크 4:
- 체크 5:
`.trim();

    const userMsg = `
[선택 주제]
${selected_topic}

[선택 주제 상세]
${selected_topic_detail || '없음'}

[학생 정보]
- 학년/학기: ${grade}
- 학교 유형: ${schoolType}
- 선택 과목: ${subject}
- 희망 진로: ${career || '미입력'}
- 이전 주제: ${selectedSavedSession.previous_topic || '없음'}

[수행평가 안내문 요약]
${selectedSavedSession.assessment_info || '안내문 정보 없음'}

작업:
- 선택 주제를 기준으로 통합 수행평가 설계 리포트를 작성하라.
- 추천 자료, 출처 링크, 주제 관련 정보, 서론/본론/결론 구조를 모두 하나의 리포트에 넣어라.
- 결론 구성 방향을 반드시 포함하라.
`.trim();

    const rawResult = await callText(system, userMsg, {
      maxOutputTokens: 5200,
      temperature: 0.25
    });

    const result = ensureConclusionSection(rawResult);

    const updated = await updateSession(session_id, {
      selected_topic,
      selected_topic_detail,
      resources: result
    });

    await dbSaveConversation(updated);

    const savedReport = await saveAssessmentReport({
      main_id: selectedSavedSession.main_id,
      student_name: selectedSavedSession.student_name || '',
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

    return res.status(500).json({
      detail: '설계 리포트 생성 중 오류가 발생했습니다.',
      error_message: error?.message || String(error)
    });
  }
}
