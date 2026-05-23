import { supabaseAdmin } from './_lib/supabase.js';
import { SUPABASE_IMAGE_BUCKET, CORE_PRINCIPLES } from './_lib/config.js';
import {
  getSession,
  updateSession,
  dbSaveConversation,
  incrementCallCount
} from './_lib/sessions.js';
import { callVision } from './_lib/gemini.js';

async function downloadImage(path) {
  const { data, error } = await supabaseAdmin
    .storage
    .from(SUPABASE_IMAGE_BUCKET)
    .download(path);

  if (error) throw error;

  const arrayBuffer = await data.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

async function deleteImage(path) {
  try {
    await supabaseAdmin
      .storage
      .from(SUPABASE_IMAGE_BUCKET)
      .remove([path]);
  } catch (error) {
    console.error('이미지 삭제 오류:', error);
  }
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    const {
      session_id,
      image_path,
      mime_type = 'image/jpeg',
      subject = '',
      career = '',
      grade = '고등학생'
    } = req.body || {};

    if (!session_id || !image_path) {
      return res.status(400).json({ detail: 'session_id와 image_path가 필요합니다.' });
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

    const imageBytes = await downloadImage(image_path);

    const system = `
${CORE_PRINCIPLES}

당신은 고등학교 수행평가 안내문 분석 전문가입니다.
반드시 아래 형식으로 정확히 추출하세요:

[수행평가 기본 정보]
- 교과/과목:
- 학년:
- 수행평가 유형:
- 주제/제목:
- 제출 형식:
- 제출 기한:

[평가 기준 및 배점]
(항목별 배점 정리)

[세부 요구사항]
- 필수 포함 내용:
- 특이사항:

[질문 목록 - 절대 누락 금지]
안내문에 번호가 매겨진 질문 목록이 있으면 반드시 아래 형식으로 전부 추출하세요.
질문이 10개든 20개든 하나도 빠짐없이 전부 적으세요.
질문 1: (질문 내용 그대로)
질문 2: (질문 내용 그대로)
질문 3: (질문 내용 그대로)
...
질문 목록이 없으면 "질문 목록 없음"으로 표시하세요.

[자료/도서/영상 목록 - 절대 누락 금지]
안내문에 학생이 선택해야 할 자료/도서/영상 목록이 있으면 전부 추출하세요.
없으면 "없음"으로 표시하세요.

확인 안 되는 항목은 '정보 없음'으로 표시하세요.
불확실한 정보는 절대 임의로 추가하지 마세요.
`.trim();

    const result = await callVision(
      system,
      imageBytes,
      mime_type,
      '수행평가 안내문의 모든 정보를 추출해주세요.'
    );

    await deleteImage(image_path);

    const newAssessmentInfo = `${session.assessment_info || ''}\n\n${result}`.trim();

    const updated = await updateSession(session_id, {
      subject: subject || session.subject || '',
      career: career || session.career || '',
      grade: grade || session.grade || '고등학생',
      assessment_info: newAssessmentInfo
    });

    await dbSaveConversation(updated);

    return res.status(200).json({
      status: 'success',
      assessment_info: result,
      image_path,
      image_deleted: true,
      call_count: usage.count,
      call_limit: usage.limit
    });
  } catch (error) {
    console.error('analyze assessment storage error:', error);
    return res.status(500).json({ detail: '안내문 분석 중 오류가 발생했습니다.' });
  }
}
