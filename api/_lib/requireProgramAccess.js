import { createClient } from '@supabase/supabase-js';

const mainSupabase = createClient(
  process.env.WINNING_SUPABASE_URL,
  process.env.WINNING_SUPABASE_KEY,
  {
    auth: {
      persistSession: false,
      autoRefreshToken: false,
    },
  }
);

export async function requireProgramAccess(req, programKey) {
  const token = req.headers.authorization?.replace('Bearer ', '');

  if (!token) {
    return {
      ok: false,
      status: 401,
      message: '로그인이 필요합니다.',
    };
  }

  const { data: userData, error: userError } =
    await mainSupabase.auth.getUser(token);

  if (userError || !userData?.user) {
    return {
      ok: false,
      status: 401,
      message: '로그인이 필요합니다.',
    };
  }

  const mainId = userData.user.id;

  const { data: access, error: accessError } = await mainSupabase
    .from('program_access')
    .select('id')
    .eq('id', mainId)
    .eq('program_key', programKey)
    .eq('payment_status', 'paid')
    .eq('access_status', 'active')
    .maybeSingle();

  if (accessError || !access) {
    return {
      ok: false,
      status: 403,
      message: '결제 후 시도해주세요!',
    };
  }

  return {
    ok: true,
    mainId,
    user: userData.user,
  };
}
