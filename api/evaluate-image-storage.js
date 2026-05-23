import { supabaseAdmin } from './_lib/supabase.js';
import { SUPABASE_IMAGE_BUCKET, CORE_PRINCIPLES } from './_lib/config.js';
import { loadKnowledgeByGradeSubject } from './_lib/knowledge.js';
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

  let imagePathToDelete = null;

  try {
    const {
      session_id,
      image_path,
      mime_type = 'image/jpeg',
      confirm_submit
    } = req.body || {};

    if (!session_id) {
      return res.status(400).json({
        detail: 'session_id가 필요합니다.'
      });
    }

    if (confirm_submit !== true) {
      return res.status(400).json({
        detail: '이미지 제출물 평가 버튼을 통해서만 평가할 수 있습니다.'
      });
    }

    if (!image_path) {
      return res.status(400).json({
        detail: 'image_path가 필요합니다.'
      });
    }

    imagePathToDelete = image_path;

    const session = await getSession(session_id);

    if (!session?.student_code) {
      return res.status(401).json({
        detail: '로그인이 필요합니다.'
      });
    }

    const usage = await incrementCallCount(session.student_code);

    if (!usage.allowed) {
      return res.status(429).json({
        detail: `이용 횟수를 모두 사용했어요. (사용: ${usage.count}/${usage.limit}회)`
      });
    }

    const imageBytes = await downloadImage(image_path);

    const knowledgeBase = loadKnowledgeByGradeSubject(
      session.grade || '고등학생',
      session.subject || '국어'
    );

    const system = `
${CORE_PRINCIPLES}

당신은 고등학교 수행평가 채점 전문가입니다.

[학년·과목별 내부 지식 데이터]
${knowledgeBase}

[평가 기준]
${session.assessment_info || '평가 기준 정보 없음'}

[선택 주제] ${session.selected_topic || '미입력'}
[희망 진로] ${session.career || '미입력'}
[학년] ${session.grade || '미입력'}
[이전에 했던 주제] ${session.previous_topic || '없음'}

평가 시 반드시 확인:
- 평가 기준 모든 항목 포함 여부
- 표절 의심 문장 경고
- 이전 주제와의 심화·연결
- 추후 탐구 가능성
- 학년에 맞는 탐구 수준
- 이미지가 실제 학생 수행평가 제출물인지 여부
- 이미지가 제출물이 아니라 안내문, 주제 목록, 자료 추천 결과, 빈 화면, 단순 메모처럼 보이면 평가하지 말고 제출물 부족으로 안내

각 항목마다:
[항목명] ★★★★☆
1. 잘한 점:
2. 아쉬운 점:
3. 보완할 점:

표절 위험 문장
심화 탐구 연결성
종합 평가
예상 점수: X점 / 100점
총평:
`.trim();

    const result = await callVision(
      system,
      imageBytes,
      mime_type,
      '이 이미지는 학생이 제출물 평가 버튼을 눌러 업로드한 수행평가 제출물입니다. 실제 제출물인지 먼저 확인한 뒤, 핵심 원칙을 최우선으로 지키며 평가해주세요.'
    );

    await deleteImage(image_path);
    imagePathToDelete = null;

    const updated = await updateSession(session_id, {
      evaluation: result
    });

    await dbSaveConversation(updated);

    return res.status(200).json({
      status: 'success',
      evaluation: result,
      image_path,
      image_deleted: true,
      call_count: usage.count,
      call_limit: usage.limit
    });
  } catch (error) {
    console.error('evaluate image storage error:', error);

    if (imagePathToDelete) {
      await deleteImage(imagePathToDelete);
    }

    return res.status(500).json({
      detail: '이미지 평가 중 오류가 발생했습니다.'
    });
  }
}
