import crypto from 'crypto';
import { supabaseAdmin } from './_lib/supabase.js';
import { requireProgramAccess } from './_lib/requireProgramAccess.js';
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
    const auth = await requireProgramAccess(req, 'suhaeng');

console.log('suhaeng login auth result:', auth);

if (!auth.ok) {
  return res.status(auth.status).json({
    detail: auth.message || '결제 후 이용해주세요.'
  });
}
    const mainId = auth.mainId;
    const fallbackName = auth.user?.user_metadata?.name || auth.user?.email || '';

    let student = await dbGetStudent(mainId);

    if (!student) {
      const { data: createdStudent, error: createError } = await supabaseAdmin
        .from('students')
        .insert({
          main_id: mainId,
          name: fallbackName,
          call_limit: 0,
          call_count: 0
        })
        .select('*')
        .single();

      if (createError) throw createError;
      student = createdStudent;
    }

    const cleanName = student.name || fallbackName;
    const prev = await dbGetConversation(mainId);
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
      main_id: mainId,
      student_name: cleanName,
      subject: prev?.subject || '',
      grade: prev?.grade || '',
      career: prev?.career || '',
      assessment_info: prev?.assessment_info || '',
      selected_topic: selectedTopic,
      selected_topic_detail: selectedTopicDetail,
      topics: prev?.topics || '',
      resources: prev?.resources || '',
      evaluation: prev?.evaluation || '',
      school_type: prev?.school_type || '일반고'
    });

    const previous = prev
      ? {
          ...prev,
          selected_topic: selectedTopic,
          selected_topic_detail: selectedTopicDetail
        }
      : null;

    return res.status(200).json({
      status: 'success',
      session_id: sessionId,
      name: cleanName,
      main_id: mainId,
      previous,
      call_limit: student.call_limit || 0,
      call_count: student.call_count || 0,
      message: `안녕하세요, ${cleanName}님! 👋`
    });
  } catch (error) {
    console.error('login error:', error);
    return res.status(500).json({
      detail: '로그인 처리 중 오류가 발생했습니다.',
      error_message: error?.message || String(error)
    });
  }
}
