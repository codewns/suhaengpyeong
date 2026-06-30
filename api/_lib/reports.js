import { supabaseAdmin } from './supabase.js';

export async function saveAssessmentReport(report) {
  try {
    const payload = {
      main_id: report.main_id,
      student_name: report.student_name || '',
      report_type: report.report_type || 'plan',
      title: report.title || '',
      grade: report.grade || '',
      subject: report.subject || '',
      career: report.career || '',
      selected_topic: report.selected_topic || '',
      report_content: report.report_content || '',
      submission_text: report.submission_text || '',
      evaluation_result: report.evaluation_result || '',
      meta: report.meta || {}
    };

    const { data, error } = await supabaseAdmin
      .from('assessment_reports')
      .insert(payload)
      .select()
      .single();

    if (error) {
      console.error('assessment report 저장 오류:', error);
      return null;
    }

    return data;
  } catch (error) {
    console.error('assessment report 저장 예외:', error);
    return null;
  }
}

export async function listAssessmentReports(mainId, limit = 30) {
  const { data, error } = await supabaseAdmin
    .from('assessment_reports')
    .select('id, report_type, title, grade, subject, career, selected_topic, created_at')
    .eq('main_id', mainId)
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) {
    console.error('assessment report 목록 조회 오류:', error);
    return [];
  }

  return data || [];
}

export async function getAssessmentReport(mainId, reportId) {
  const { data, error } = await supabaseAdmin
    .from('assessment_reports')
    .select('*')
    .eq('main_id', mainId)
    .eq('id', reportId)
    .maybeSingle();

  if (error) {
    console.error('assessment report 단건 조회 오류:', error);
    return null;
  }

  return data;
}

export async function getRecentEvaluationReportSummaries(mainId, limit = 8) {
  const { data, error } = await supabaseAdmin
    .from('assessment_reports')
    .select('id, title, grade, subject, career, selected_topic, report_content, created_at')
    .eq('main_id', mainId)
    .eq('report_type', 'evaluation')
    .order('created_at', { ascending: false })
    .limit(limit);

  if (error) {
    console.error('최근 평가 리포트 조회 오류:', error);
    return [];
  }

  return data || [];
}

export function formatPreviousReportsForRecommendation(reports, currentSubject) {
  if (!reports?.length) return '이전 평가 리포트 없음';

  const same = reports.filter((r) => r.subject === currentSubject);
  const other = reports.filter((r) => r.subject !== currentSubject);

  const compact = (text, max = 900) => {
    const v = String(text || '').replace(/\s+/g, ' ').trim();
    return v.length > max ? `${v.slice(0, max)}...` : v;
  };

  const render = (r, i) => `
[이전 수행 ${i + 1}]
- 과목: ${r.subject || '미입력'}
- 주제: ${r.selected_topic || r.title || '미입력'}
- 평가 리포트 요약: ${compact(r.report_content)}
`.trim();

  return `
[같은 과목 이전 수행 기록 - 반드시 심화·확장 우선]
${same.length ? same.map(render).join('\n\n') : '같은 과목 이전 수행 기록 없음'}

[다른 과목 이전 수행 기록 - 현재 과목 방식으로 재해석 가능할 때만 활용]
${other.length ? other.map(render).join('\n\n') : '다른 과목 이전 수행 기록 없음'}
`.trim();
}
