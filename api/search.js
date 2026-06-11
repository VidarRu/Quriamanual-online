import { createClient } from "@supabase/supabase-js";
import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_KEY
);

export default async function handler(req, res) {
  // CORS headers — tighten origin in production if needed
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { question } = req.body ?? {};
  if (!question || typeof question !== "string" || question.trim().length === 0) {
    return res.status(400).json({ error: "question is required" });
  }

  // 1. Embed the user's question
  const embeddingResponse = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: question.trim(),
  });
  const queryEmbedding = embeddingResponse.data[0].embedding;

  // 2. Find relevant chunks via pgvector cosine similarity
  const { data: chunks, error: matchError } = await supabase.rpc("match_chunks", {
    query_embedding: queryEmbedding,
    match_count: 5,
    similarity_threshold: 0.25,
  });

  if (matchError) {
    console.error("match_chunks error:", matchError);
    return res.status(500).json({ error: "Database search failed" });
  }

  if (!chunks || chunks.length === 0) {
    return res.json({
      answer: "Jag hittade tyvärr ingen relevant information i manualen för den frågan. Försök omformulera eller sök med andra nyckelord.",
      sources: [],
    });
  }

  // 3. Build context from retrieved chunks
  const context = chunks
    .map((c, i) => `[${i + 1}] ${c.breadcrumb}\n${c.content}`)
    .join("\n\n---\n\n");

  // 4. Ask Claude to answer based on the context
  const message = await anthropic.messages.create({
    model: "claude-haiku-4-5-20251001",
    max_tokens: 600,
    system: `Du är en hjälpsam assistent för Quria-manualen (ett molnbaserat bibliotekssystem från Axiell).
Svara på svenska baserat ENBART på den kontext som ges nedan.
Om svaret inte finns i kontexten, säg det tydligt.
Håll svaret kortfattat och praktiskt. Använd gärna punktlistor för steg-för-steg-instruktioner.`,
    messages: [
      {
        role: "user",
        content: `Kontext från manualen:\n\n${context}\n\nFråga: ${question}`,
      },
    ],
  });

  const answer = message.content[0].text;

  // 5. Return answer + sources for linking
  const sources = chunks.map((c) => ({
    section: c.breadcrumb,
    url: `https://vidarru.github.io/Quriamanual-online/${c.url_fragment}`,
    similarity: Math.round(c.similarity * 100),
  }));

  return res.json({ answer, sources });
}
