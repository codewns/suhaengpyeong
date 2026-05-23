import crypto from 'crypto';
import {
  dbGetStudent,
  dbGetConversation,
  createSession
} from './_lib/sessions.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    const { name, code } = req.body || {};

    if (!name || !code) {
      return res.status(400).json({ detail: '이름과 코드를 모두 입력해주세요.' });
    }

    const cleanCode = String(code).trim().toUpperCase();
    const cleanName = String(name).trim();

    const student = await dbGetStudent(cleanCode);

    if (!student) {
      return res.status(401).json({ detail: '등록되지 않은 코드예요. 선생님께 문의하세요.' });
    }

    if (String(student.name).trim() !== cleanName) {
      return res.status(401).json({ detail: '이름이 일치하지 않아요. 다시 확인해주세요.' });
    }

    const prev = await dbGetConversation(cleanCode);
    const sessionId = crypto.randomUUID();

    let selectedTopic = '';
    let selectedTopicDetail = '';

    if (prev?.selected_topic) {
      if (prev.selected_topic.includes('|||')) {
        const parts = prev.selected_topic.split('|||');
        selectedTopic = parts[0] || '';
        selectedTopicDetail = parts[1] || '';
      } else {
        selectedTopic = prev.selected_topic;
      }
    }

    await createSession({
      session_id: sessionId,
      student_code: cleanCode,
      student_name: cleanName,
      subject: prev?.subject || '',
      grade: prev?.grade || '',
      career: prev?.career || '',
      assessment_info: prev?.assessment_info || '',
      selected_topic: selectedTopic,
      selected_topic_detail: selectedTopicDetail,
      topics: prev?.topics || '',
      resources: prev?.resources || '',
      evaluation: prev?.evaluation || ''
    });

    let previous = prev ? { ...prev } : null;

    if (previous) {
      previous.selected_topic = selectedTopic;
      previous.selected_topic_detail = selectedTopicDetail;
    }

    return res.status(200).json({
      status: 'success',
      session_id: sessionId,
      name: cleanName,
      code: cleanCode,
      previous,
      call_limit: student.call_limit || 0,
      call_count: student.call_count || 0,
      message: `안녕하세요, ${cleanName}님! 👋`
    });
  } catch (error) {
    console.error('login error:', error);
    return res.status(500).json({ detail: '로그인 처리 중 오류가 발생했습니다.' });
  }
}
