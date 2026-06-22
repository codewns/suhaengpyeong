import { GoogleGenAI } from '@google/genai';
import { GEMINI_API_KEY, MODEL } from './config.js';

if (!GEMINI_API_KEY) {
  throw new Error('GEMINI_API_KEY가 설정되지 않았습니다.');
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

export async function callText(system, userMsg, options = {}) {
  const response = await ai.models.generateContent({
    model: options.model || MODEL,
    contents: userMsg,
    config: {
      systemInstruction: system,
      temperature: options.temperature ?? 0.35,
      maxOutputTokens: options.maxOutputTokens ?? 2200
    }
  });

  return response.text || '';
}

export async function callVision(system, imageBytes, mimeType, prompt, options = {}) {
  const base64 = Buffer.from(imageBytes).toString('base64');

  const response = await ai.models.generateContent({
    model: options.model || MODEL,
    contents: [
      {
        inlineData: {
          mimeType,
          data: base64
        }
      },
      {
        text: prompt
      }
    ],
    config: {
      systemInstruction: system,
      temperature: options.temperature ?? 0.25,
      maxOutputTokens: options.maxOutputTokens ?? 2600
    }
  });

  return response.text || '';
}
