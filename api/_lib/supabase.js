import { createClient } from '@supabase/supabase-js';
import {
  SUPABASE_URL,
  SUPABASE_SERVICE_ROLE_KEY,
  WINNING_SUPABASE_URL,
  WINNING_SUPABASE_SERVICE_ROLE_KEY
} from './config.js';

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  throw new Error('수행평가 Supabase 환경변수가 설정되지 않았습니다.');
}

export const supabaseAdmin = createClient(
  SUPABASE_URL,
  SUPABASE_SERVICE_ROLE_KEY,
  {
    auth: {
      persistSession: false,
      autoRefreshToken: false
    }
  }
);

export const winningSupabaseAdmin =
  WINNING_SUPABASE_URL && WINNING_SUPABASE_SERVICE_ROLE_KEY
    ? createClient(
        WINNING_SUPABASE_URL,
        WINNING_SUPABASE_SERVICE_ROLE_KEY,
        {
          auth: {
            persistSession: false,
            autoRefreshToken: false
          }
        }
      )
    : null;

