#!/usr/bin/env node
/**
 * Extracts text chunks from the manual HTML, embeds them with OpenAI,
 * and stores them in Supabase pgvector.
 *
 * Usage:
 *   OPENAI_API_KEY=... SUPABASE_URL=... SUPABASE_SERVICE_KEY=... node scripts/embed.js
 */

import { readFileSync } from "fs";
import { createClient } from "@supabase/supabase-js";
import OpenAI from "openai";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY;

if (!OPENAI_API_KEY || !SUPABASE_URL || !SUPABASE_SERVICE_KEY) {
  console.error("Missing required env vars: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY");
  process.exit(1);
}

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

/** Parse HTML and extract chunks per H3 section (with H2 breadcrumb). */
function extractChunks(html) {
  const chunks = [];

  // Strip script/style/nav blocks to avoid embedding them
  const cleaned = html
    .replace(/<style[\s\S]*?<\/style>/gi, "")
    .replace(/<nav[\s\S]*?<\/nav>/gi, "")
    .replace(/<script[\s\S]*?<\/script>/gi, "");

  // Split on h2 and h3 headings
  const sectionRegex = /<(h2|h3)[^>]*id="([^"]*)"[^>]*>([\s\S]*?)<\/h[23]>([\s\S]*?)(?=<h[123]|$)/gi;

  let currentH2 = "";
  let currentH2Id = "";
  let match;

  while ((match = sectionRegex.exec(cleaned)) !== null) {
    const [, level, id, headingHtml, bodyHtml] = match;

    const headingText = headingHtml.replace(/<[^>]+>/g, "").trim();
    const bodyText = bodyHtml
      .replace(/<[^>]+>/g, " ")
      .replace(/\s+/g, " ")
      .trim();

    if (level === "h2") {
      currentH2 = headingText;
      currentH2Id = id;
      // Include H2 intro text as its own chunk if substantial
      if (bodyText.length > 100) {
        chunks.push({
          id,
          section: headingText,
          breadcrumb: headingText,
          content: `${headingText}\n\n${bodyText}`,
          url_fragment: `#${id}`,
        });
      }
    } else {
      // H3 chunk: include breadcrumb for context
      const fullContent = currentH2
        ? `${currentH2} > ${headingText}\n\n${bodyText}`
        : `${headingText}\n\n${bodyText}`;

      if (bodyText.length > 50) {
        chunks.push({
          id,
          section: headingText,
          breadcrumb: currentH2 ? `${currentH2} > ${headingText}` : headingText,
          content: fullContent,
          url_fragment: `#${id}`,
        });
      }
    }
  }

  return chunks;
}

async function embedChunks(chunks) {
  const texts = chunks.map((c) => c.content);
  // OpenAI supports up to 2048 inputs per request; batch to be safe
  const batchSize = 100;
  const embeddings = [];

  for (let i = 0; i < texts.length; i += batchSize) {
    const batch = texts.slice(i, i + batchSize);
    console.log(`Embedding batch ${i / batchSize + 1}/${Math.ceil(texts.length / batchSize)}...`);
    const response = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: batch,
    });
    embeddings.push(...response.data.map((d) => d.embedding));
  }

  return embeddings;
}

async function upsertToSupabase(chunks, embeddings) {
  const rows = chunks.map((chunk, i) => ({
    chunk_id: chunk.id,
    section: chunk.section,
    breadcrumb: chunk.breadcrumb,
    content: chunk.content,
    url_fragment: chunk.url_fragment,
    embedding: embeddings[i],
  }));

  // Delete existing rows and re-insert (simple full refresh)
  console.log("Deleting old rows...");
  const { error: delError } = await supabase.from("manual_chunks").delete().neq("chunk_id", "__never__");
  if (delError) console.warn("Delete warning:", delError.message);

  console.log(`Inserting ${rows.length} chunks...`);
  const { error } = await supabase.from("manual_chunks").insert(rows);
  if (error) throw new Error(`Supabase insert failed: ${error.message}`);
}

async function main() {
  const html = readFileSync(new URL("../index.html", import.meta.url), "utf-8");
  console.log("Extracting chunks...");
  const chunks = extractChunks(html);
  console.log(`Found ${chunks.length} chunks`);

  const embeddings = await embedChunks(chunks);
  await upsertToSupabase(chunks, embeddings);
  console.log("Done! All chunks embedded and stored.");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
