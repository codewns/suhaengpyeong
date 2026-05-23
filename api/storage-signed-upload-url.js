import crypto from 'crypto';
import { supabaseAdmin } from './_lib/supabase.js';
import { SUPABASE_IMAGE_BUCKET } from './_lib/config.js';
import { getSession } from './_lib/sessions.js';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  try {
    const { session_id, content_type, purpose = 'assessment' } = req.body || {};

    if (!session_id) {
      return res.status(400).json({ detail: 'session_id가 필요합니다.' });
    }

    const session = await getSession(session_id);

    if (!session?.student_code) {
      return res.status(401).json({ detail: '로그인이 필요합니다.' });
    }

    const allowedTypes = {
      'image/jpeg': '.jpg',
      'image/png': '.png',
      'image/webp': '.webp'
    };

    if (!allowedTypes[content_type]) {
      return res.status(400).json({ detail: 'JPG, PNG, WEBP 이미지만 업로드할 수 있습니다.' });
    }

    const safePurpose = ['assessment', 'evaluation'].includes(purpose)
      ? purpose
      : 'assessment';

    const ext = allowedTypes[content_type];
    const path = `${session.student_code}/${session_id}/${safePurpose}/${crypto.randomUUID()}${ext}`;

    const { data, error } = await supabaseAdmin
      .storage
      .from(SUPABASE_IMAGE_BUCKET)
      .createSignedUploadUrl(path);

    if (error) {
      console.error('signed upload url error:', error);
      return res.status(500).json({ detail: '업로드 URL 생성에 실패했습니다.' });
    }

    return res.status(200).json({
      status: 'success',
      bucket: SUPABASE_IMAGE_BUCKET,
      path: data.path,
      token: data.token,
      signed_url: data.signedUrl
    });
  } catch (error) {
    console.error('storage signed url error:', error);
    return res.status(500).json({ detail: '이미지 업로드 준비 중 오류가 발생했습니다.' });
  }
}
