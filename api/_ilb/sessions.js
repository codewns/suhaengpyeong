import { supabaseAdmin } from './supabase.js';

export async function getSession(sessionId) {
  const { data, error } = await supabaseAdmin
    .from('api_sessions')
    .select('*')
    .eq('session_id', sessionId)
    .maybeSingle();

  if (error) {
    console.error('세션 조회 오류:', error);
    return null;
  }

  return data;
}

export async function createSession(session) {
  const { data, error } = await supabaseAdmin
    .from('api_sessions')
    .insert({
      ...session,
      updated_at: new Date().toISOString()
    })
    .select()
    .single();

  if (error) throw error;
  return data;
}

export async function updateSession(sessionId, updates) {
  const { data, error } = await supabaseAdmin
    .from('api_sessions')
    .update({
      ...updates,
      updated_at: new Date().toISOString()
    })
    .eq('session_id', sessionId)
    .select()
    .single();

  if (error) throw error;
  return data;
}

export async function dbGetStudent(code) {
  const { data, error } = await supabaseAdmin
    .from('students')
    .select('*')
    .eq('code', String(code).toUpperCase())
    .maybeSingle();

  if (error) {
    console.error('학생 조회 오류:', error);
    return null;
  }

  return data;
}

export async function dbGetConversation(code) {
  const { data, error } = await supabaseAdmin
    .from('conversations')
    .select('*')
    .eq('student_code', String(code).toUpperCase())
    .order('updated_at', { ascending: false })
    .limit(1)
    .maybeSingle();

  if (error) {
    console.error('대화 조회 오류:', error);
    return null;
  }

  return data;
}

export async function dbSaveConversation(session) {
  if (!session?.student_code) return;

  const now = new Date().toISOString();

  const selectedTopicCombined = session.selected_topic_detail
    ? `${session.selected_topic || ''}|||${session.selected_topic_detail}`
    : session.selected_topic || '';

  const dataToSave = {
    student_code: session.student_code,
    student_name: session.student_name || '',
    subject: session.subject || '',
    grade: session.grade || '',
    career: session.career || '',
    assessment_info: session.assessment_info || '',
    selected_topic: selectedTopicCombined,
    topics: session.topics || '',
    resources: session.resources || '',
    evaluation: session.evaluation || '',
    updated_at: now
  };

  const existing = await dbGetConversation(session.student_code);

  if (existing) {
    const { error } = await supabaseAdmin
      .from('conversations')
      .update(dataToSave)
      .eq('student_code', String(session.student_code).toUpperCase());

    if (error) throw error;
  } else {
    const { error } = await supabaseAdmin
      .from('conversations')
      .insert(dataToSave);

    if (error) throw error;
  }
}

export async function incrementCallCount(code) {
  const student = await dbGetStudent(code);

  if (!student) {
    return { allowed: false, count: 0, limit: 0 };
  }

  const limit = student.call_limit || 0;
  const count = student.call_count || 0;

  if (limit !== 0 && count >= limit) {
    return { allowed: false, count, limit };
  }

  const nextCount = count + 1;

  const { error } = await supabaseAdmin
    .from('students')
    .update({ call_count: nextCount })
    .eq('code', String(code).toUpperCase());

  if (error) {
    console.error('호출 횟수 업데이트 오류:', error);
  }

  return { allowed: true, count: nextCount, limit };
}
