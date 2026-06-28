import {
  getSession,
  updateSession,
  dbSaveConversation
} from './_lib/sessions.js';

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

    const updated = await updateSession(session_id, {
      selected_topic,
      selected_topic_detail
    });

    await dbSaveConversation(updated);

    return res.status(200).json({
      status: 'success',
      selected_topic,
      saved: true
    });
  } catch (error) {
    console.error('save topic selection error:', error);

    return res.status(500).json({
      detail: '선택 주제 저장 중 오류가 발생했습니다.',
      error_message: error?.message || String(error)
    });
  }
}
