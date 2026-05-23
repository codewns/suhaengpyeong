import { GoogleGenAI } from '@google/genai';
import { GEMINI_API_KEY, MODEL } from './config.js';

if (!GEMINI_API_KEY) {
  throw new Error('GEMINI_API_KEY가 설정되지 않았습니다.');
}

const ai = new GoogleGenAI({ apiKey: GEMINI_API_KEY });

export async function callText(system, userMsg) {
  const response = await ai.models.generateContent({
    model: MODEL,
    contents: userMsg,
    config: {
      systemInstruction: system
    }
  });

  return response.text || '';
}

export async function callVision(system, imageBytes, mimeType, prompt) {
  const base64 = Buffer.from(imageBytes).toString('base64');

  const response = await ai.models.generateContent({
    model: MODEL,
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
      systemInstruction: system
    }
  });

  return response.text || '';
}
