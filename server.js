/**
 * Backend server for Clinical Summary Generator.
 * Keeps the Gemini API key on the server; the browser never sees it.
 */
require('dotenv').config();
const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

const API_KEY = process.env.GEMINI_API_KEY;
if (!API_KEY) {
  console.error('Missing GEMINI_API_KEY in environment. Create a .env file with GEMINI_API_KEY=your_key');
  process.exit(1);
}

app.use(cors({ origin: true }));
app.use(express.json());
app.use(express.static(__dirname));

const SYSTEM_PROMPT = `You are a professional Clinical Text Analysis engine. Your task is to perform three steps: 
1. Summarization: Create a short, coherent, and factually consistent clinical summary of the input text.
2. Entity Extraction (NER): Identify and extract critical entities from the text. The types MUST be limited to 'Disease', 'Drug', 'Procedure', 'Lab Test', or 'Patient Detail' (for names, ages, dates, etc.).
3. Hallucination Mitigation: Ensure the generated summary contains ONLY information explicitly supported by the input text.

Return the result in the mandatory JSON format. Do not include any preamble or extra text.`;

// Simplified schema (no propertyOrdering) to avoid Gemini 400 "invalid schema" errors
const RESPONSE_SCHEMA = {
  type: 'OBJECT',
  properties: {
    summary: { type: 'STRING', description: 'The concise clinical summary.' },
    entities: {
      type: 'ARRAY',
      description: 'List of entities: each has text and type (Disease, Drug, Procedure, Lab Test, Patient Detail).',
      items: {
        type: 'OBJECT',
        properties: {
          text: { type: 'STRING', description: 'Entity text.' },
          type: { type: 'STRING', description: 'One of: Disease, Drug, Procedure, Lab Test, Patient Detail.' }
        }
      }
    }
  }
};

const MODEL_CANDIDATES = ['gemini-2.5-flash', 'gemini-2.5-flash-lite'];

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function callGeminiWithResilience(userQuery) {
  const basePayload = {
    contents: [{ parts: [{ text: userQuery }] }],
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: RESPONSE_SCHEMA
    }
  };

  const retryableStatuses = new Set([429, 500, 502, 503, 504]);
  const maxAttemptsPerModel = 3;

  for (const modelName of MODEL_CANDIDATES) {
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${encodeURIComponent(API_KEY)}`;
    let delayMs = 1200;

    for (let attempt = 1; attempt <= maxAttemptsPerModel; attempt++) {
      const payload = JSON.parse(JSON.stringify(basePayload));
      let response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      let result = await response.json().catch(() => ({}));

      // If 400 (e.g. schema rejected), retry without responseSchema - JSON only
      if (response.status === 400 && payload.generationConfig.responseSchema) {
        delete payload.generationConfig.responseSchema;
        response = await fetch(apiUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        result = await response.json().catch(() => ({}));
      }

      if (response.ok) {
        return result;
      }

      const message = result.error?.message || `API returned ${response.status}`;
      const canRetry = retryableStatuses.has(response.status);
      const isLastAttempt = attempt === maxAttemptsPerModel;

      if (canRetry && !isLastAttempt) {
        await sleep(delayMs);
        delayMs *= 2;
        continue;
      }

      if (!canRetry) {
        const error = new Error(message);
        error.status = response.status;
        throw error;
      }

      // Last retry for this model failed; move on to next model.
      if (modelName === MODEL_CANDIDATES[MODEL_CANDIDATES.length - 1] && isLastAttempt) {
        const error = new Error('Model is temporarily overloaded. Please retry in 30-60 seconds.');
        error.status = 503;
        throw error;
      }
    }
  }
}

app.post('/api/analyze', async (req, res) => {
  const rawText = req.body?.text;
  if (!rawText || typeof rawText !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid "text" in request body.' });
  }

  const userQuery = `Analyze the following clinical note: \n\n---START OF NOTE---\n${rawText}\n---END OF NOTE---\n`;

  try {
    const result = await callGeminiWithResilience(userQuery);

    if (!result.candidates?.length || !result.candidates[0].content?.parts?.length) {
      return res.status(502).json({ error: 'API response missing content or candidates.' });
    }

    let jsonString = result.candidates[0].content.parts[0].text.trim();
    if (jsonString.startsWith('```')) {
      const first = jsonString.indexOf('\n');
      const last = jsonString.lastIndexOf('```');
      if (first !== -1 && last > first) {
        jsonString = jsonString.slice(first + 1, last).trim();
      }
    }

    const parsed = JSON.parse(jsonString);
    if (!parsed.summary || !Array.isArray(parsed.entities)) {
      return res.status(502).json({ error: 'Model returned incomplete data (missing summary or entities).' });
    }

    res.json(parsed);
  } catch (err) {
    console.error('Analyze error:', err);
    const status = err.status || 500;
    res.status(status).json({ error: err.message || 'Server error during analysis.' });
  }
});

app.get('/', (req, res) => {
  res.redirect('/index2.html');
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Open http://localhost:${PORT}/index2.html in your browser.`);
});
