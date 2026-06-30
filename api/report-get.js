import { getSession } from './_lib/sessions.js';
import { getAssessmentReport } from './_lib/reports.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    const { session_id, report_id } = req.body || {};

    if (!session_id || !report_id) {
      return res.status(400).json({ detail: 'session_id와 report_id가 필요합니다.' });
    }

    const session = await getSession(session_id);

    if (!session?.main_id) {
      return res.status(401).json({ detail: '로그인이 필요합니다.' });
    }

    const report = await getAssessmentReport(session.main_id, report_id);

    if (!report) {
      return res.status(404).json({ detail: '리포트를 찾을 수 없습니다.' });
    }

    return res.status(200).json({ status: 'success', report });
  } catch (error) {
    console.error('report get error:', error);
    return res.status(500).json({ detail: '리포트를 불러오는 중 오류가 발생했습니다.' });
  }
}
