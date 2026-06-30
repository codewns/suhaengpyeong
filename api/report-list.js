import { getSession } from './_lib/sessions.js';
import { listAssessmentReports } from './_lib/reports.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    const { session_id } = req.body || {};

    if (!session_id) {
      return res.status(400).json({ detail: 'session_id가 필요합니다.' });
    }

    const session = await getSession(session_id);

    if (!session?.main_id) {
      return res.status(401).json({ detail: '로그인이 필요합니다.' });
    }

    const reports = await listAssessmentReports(session.main_id, 50);

    return res.status(200).json({ status: 'success', reports });
  } catch (error) {
    console.error('report list error:', error);
    return res.status(500).json({ detail: '리포트 목록을 불러오는 중 오류가 발생했습니다.' });
  }
}

