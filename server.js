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

const RESPONSE_SCHEMA = {
  type: 'OBJECT',
  properties: {
    summary: { type: 'STRING', description: 'The concise, coherent, and factually accurate clinical summary.' },
    entities: {
      type: 'ARRAY',
      description: 'A list of critical medical and patient entities extracted from the original text.',
      items: {
        type: 'OBJECT',
        properties: {
          text: { type: 'STRING', description: 'The exact entity text (e.g., \'Jane Doe\').' },
          type: { type: 'STRING', description: 'The type of entity (must be one of: \'Disease\', \'Drug\', \'Procedure\', \'Lab Test\', \'Patient Detail\').' }
        },
        propertyOrdering: ['text', 'type']
      }
    }
  },
  propertyOrdering: ['summary', 'entities']
};

app.post('/api/analyze', async (req, res) => {
  const rawText = req.body?.text;
  if (!rawText || typeof rawText !== 'string') {
    return res.status(400).json({ error: 'Missing or invalid "text" in request body.' });
  }

  const userQuery = `Analyze the following clinical note: \n\n---START OF NOTE---\n${rawText}\n---END OF NOTE---\n`;
  const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${encodeURIComponent(API_KEY)}`;

  const payload = {
    contents: [{ parts: [{ text: userQuery }] }],
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    generationConfig: {
      responseMimeType: 'application/json',
      responseSchema: RESPONSE_SCHEMA
    }
  };

  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const result = await response.json();

    if (!response.ok) {
      const msg = result.error?.message || `API returned ${response.status}`;
      return res.status(response.status).json({ error: msg });
    }

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
    res.status(500).json({ error: err.message || 'Server error during analysis.' });
  }
});

app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Open http://localhost:${PORT}/index2.html in your browser.`);
});
