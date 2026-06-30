import crypto from 'crypto';
import { supabaseAdmin } from './_lib/supabase.js';
import { requireProgramAccess } from './_lib/requireProgramAccess.js';
import {
  dbGetStudent,
  createSession
} from './_lib/sessions.js';

export default async function handler(req, res) {
  // 콜드스타트 방지용 ping
  if (req.method === 'GET') {
    return res.status(200).json({
      ok: true,
      message: 'suhaeng login api warm',
      time: new Date().toISOString()
    });
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    console.time('login-total');

    console.time('access-check');
    const auth = await requireProgramAccess(req, 'suhaeng');
    console.timeEnd('access-check');

    if (!auth.ok) {
      console.timeEnd('login-total');
      return res.status(auth.status).json({
        detail: auth.message || '결제 후 이용해주세요.'
      });
    }

    const mainId = auth.mainId;
    const fallbackName = auth.user?.user_metadata?.name || auth.user?.email || '';

    console.time('student-check');
    let student = await dbGetStudent(mainId);
    console.timeEnd('student-check');

    if (!student) {
      console.time('student-create');
      const { data: createdStudent, error: createError } = await supabaseAdmin
        .from('students')
        .insert({
          main_id: mainId,
          name: fallbackName,
          call_limit: 0,
          call_count: 0
        })
        .select('id, main_id, name, call_limit, call_count')
        .single();

      console.timeEnd('student-create');

      if (createError) throw createError;
      student = createdStudent;
    }

    const cleanName = student.name || fallbackName;
    const sessionId = crypto.randomUUID();

    console.time('session-create');
    await createSession({
      session_id: sessionId,
      main_id: mainId,
      student_name: cleanName,
      subject: '',
      grade: '',
      career: '',
      assessment_info: '',
      selected_topic: '',
      selected_topic_detail: '',
      topics: '',
      resources: '',
      evaluation: '',
      school_type: '일반고'
    });
    console.timeEnd('session-create');

    console.timeEnd('login-total');

    return res.status(200).json({
      status: 'success',
      session_id: sessionId,
      name: cleanName,
      main_id: mainId,
      previous: null,
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
