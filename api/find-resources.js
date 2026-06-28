import { loadDynamicAssessmentKnowledge } from './_lib/dynamic-knowledge.js';
import {
  getSession,
  updateSession,
  dbSaveConversation,
  incrementCallCount
} from './_lib/sessions.js';
import { callText } from './_lib/gemini.js';
import { saveAssessmentReport } from './_lib/reports.js';

function splitCombinedResult(text = '') {
  const raw = String(text || '').trim();

  const resourceMarker = '[자료 추천]';
  const planMarker = '[수행평가 설계 리포트]';

  const resourceIdx = raw.indexOf(resourceMarker);
  const planIdx = raw.indexOf(planMarker);

  let resources = raw;
  let planReport = raw;

  if (resourceIdx !== -1 && planIdx !== -1 && resourceIdx < planIdx) {
    resources = raw.slice(resourceIdx + resourceMarker.length, planIdx).trim();
    planReport = raw.slice(planIdx + planMarker.length).trim();
  } else if (planIdx !== -1) {
    resources = raw.slice(0, planIdx).replace(resourceMarker, '').trim();
    planReport = raw.slice(planIdx + planMarker.length).trim();
  } else if (resourceIdx !== -1) {
    resources = raw.slice(resourceIdx + resourceMarker.length).trim();
    planReport = raw;
  }

  return {
    resources: resources || raw,
    planReport: planReport || raw
  };
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
    const schoolType = session.school_type || '일반고';
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
당신은 고등학생 수행평가 자료 추천 및 설계 리포트 작성 전문가입니다.
이 단계의 목표는 단순 자료 추천이 아니라, 학생이 직접 보고서를 작성할 수 있도록 "선택 주제-자료-서론-본론-결론-평가 기준 대응"을 하나의 설계 리포트로 정리하는 것입니다.

[홈페이지 위닝 수행 자료 DB]
${dynamicResourceKnowledge || '사용 가능한 내부 자료 없음'}

절대 원칙:
1. 자료 추천은 학교 유형이 아니라 실제 선택 과목명을 기준으로 한다.
2. 과목명이 "과학 / 생물의 유전"처럼 입력되면 앞의 "과학"은 교과군, 뒤의 "생물의 유전"은 실제 과목명으로 본다.
3. 학교 유형이 자율형 사립고 또는 특수목적고이고 선택 과목이 전문교과라면, 자료 수준은 높이되 고등학생 수행평가에서 실제 활용 가능한 방식으로 낮춰 설명한다.
4. 위닝 수행 자료 DB에 있는 검증 자료를 우선 사용한다.
5. 내부 DB에 없는 자료명, 저자, 기관명, 링크를 지어내지 않는다.
6. 자료가 부족하면 자료명 대신 검색 키워드를 제시한다.
7. 링크가 내부 데이터에 없으면 링크 항목은 쓰지 않는다.
8. Google Search를 사용하지 않았으므로 최신성 확인이 필요한 내용은 "확인 필요"라고 표시한다.
9. 학생이 그대로 제출 가능한 완성문을 작성하지 않는다.
10. 서론·본론·결론은 완성문이 아니라 "작성 방향, 포함할 내용, 전개 순서, 주의점" 중심으로 제시한다.
11. 학생의 진로와 연결하되 억지 연결은 피한다.
12. 수행평가 안내문 조건이 있으면 그것을 최우선으로 반영한다.
13. 자료 추천과 설계 리포트는 반드시 분리해서 출력한다.
14. 본론을 의미 없이 3개로 고정하지 않는다. 선택 주제와 수행평가 조건에 맞춰 필요한 만큼 2~5개의 본론 항목으로 유동 구성한다.
15. 본론 항목명은 "본론 1, 본론 2"처럼 기계적으로 끝내지 말고, 각 항목의 역할이 드러나게 작성한다.

반드시 아래 출력 형식을 지켜라.

[자료 추천]

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

[수행평가 설계 리포트]

1. 최종 주제
- 주제명:
- 주제의 핵심 의미:
- 선택 과목과 연결되는 지점:
- 희망 진로와 연결되는 지점:

2. 탐구 핵심 질문
- 핵심 질문 1:
- 핵심 질문 2:
- 핵심 질문 3:
※ 질문은 보고서의 본론 전개가 가능하도록 구체적으로 제시한다.

3. 수행평가 전체 방향
- 보고서의 중심 주장:
- 단순 조사와 구별되는 분석 포인트:
- 교과 개념을 드러내는 방식:
- 자료를 활용하는 방식:
- 학생 개인의 해석이 들어가야 하는 부분:

4. 서론 작성 방향
- 서론의 역할:
- 반드시 포함할 내용:
- 주제 선정 동기 구성 방식:
- 수행평가 안내문 조건과 연결할 부분:
- 피해야 할 점:
- 학생이 직접 작성할 때 사용할 수 있는 표현 방향:
※ 완성 문장을 쓰지 말고, 어떤 흐름으로 써야 하는지만 제시한다.

5. 본론 구성 방향
※ 아래 본론 항목은 선택 주제에 맞춰 필요한 만큼만 구성한다. 반드시 3개로 고정하지 않는다.

본론 항목 A. [항목 역할이 드러나는 제목]
- 중심 내용:
- 연결할 교과 개념:
- 사용할 자료 또는 사례:
- 분석 순서:
- 학생이 직접 해석해야 할 부분:
- 피해야 할 점:

본론 항목 B. [항목 역할이 드러나는 제목]
- 중심 내용:
- 연결할 교과 개념:
- 사용할 자료 또는 사례:
- 분석 순서:
- 학생이 직접 해석해야 할 부분:
- 피해야 할 점:

필요한 경우에만 본론 항목 C, D, E를 추가한다.
불필요하면 추가하지 않는다.

6. 결론 작성 방향
- 결론의 역할:
- 탐구 결과 정리 방식:
- 새롭게 알게 된 점:
- 진로 또는 후속 탐구와 연결하는 방식:
- 피해야 할 점:
- 학생이 직접 작성할 때 사용할 수 있는 표현 방향:
※ 완성 문장을 쓰지 말고, 결론의 논리 흐름만 제시한다.

7. 자료 활용 계획
- 자료 1 활용 위치:
- 자료 2 활용 위치:
- 자료 3 활용 위치:
- 자료를 그대로 베끼지 않고 재구성하는 방법:
- 출처 확인이 필요한 부분:

8. 평가 기준 대응 전략
- 과목 적합성:
- 자료 신뢰성:
- 분석력:
- 진로 연계성:
- 자기 생각:
- 형식 완성도:

9. 학생 작성 체크리스트
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
- 이전 주제: ${session.previous_topic || '없음'}

[수행평가 안내문 요약]
${session.assessment_info || '안내문 정보 없음'}

작업:
1. 선택 주제와 직접 관련 있는 자료를 최대 3개 추천한다.
2. 내부 DB에 적합한 자료가 있으면 우선 사용한다.
3. 내부 DB에 자료가 부족하면 구체적인 검색 키워드로 보완한다.
4. 없는 링크, 없는 자료명, 확인되지 않은 출처는 만들지 않는다.
5. 자료 추천 뒤에는 반드시 수행평가 설계 리포트를 작성한다.
6. 설계 리포트에는 서론, 선택 주제에 맞춘 유동적인 본론 구성, 결론 작성 방향이 들어가야 한다.
7. 본론은 무조건 3개로 나누지 말고 선택 주제의 논리 흐름에 맞게 필요한 만큼만 구성한다.
8. 학생이 그대로 제출 가능한 완성문은 쓰지 않는다.
9. 학생이 직접 작성할 수 있도록 구조, 흐름, 포함 요소, 피해야 할 점 중심으로 안내한다.
`.trim();

    const result = await callText(system, userMsg);
    const { resources, planReport } = splitCombinedResult(result);

    const updated = await updateSession(session_id, {
      selected_topic,
      selected_topic_detail,
      resources
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
      report_content: planReport,
      meta: {
        selected_topic_detail,
        resources
      }
    });

    return res.status(200).json({
      status: 'success',
      resources,
      plan_report: planReport,
      report_id: savedReport?.id || null,
      call_count: usage.count,
      call_limit: usage.limit
    });
  } catch (error) {
    console.error('find resources error:', error);
    return res.status(500).json({ detail: '자료 추천 및 설계 리포트 생성 중 오류가 발생했습니다.' });
  }
}
