-- Enable the pgvector extension
create extension if not exists vector;

-- Table for manual chunks with embeddings
create table if not exists manual_chunks (
  id           bigserial primary key,
  chunk_id     text unique not null,        -- matches the HTML anchor id
  section      text not null,               -- h3 heading text
  breadcrumb   text not null,               -- "H2 > H3" for display
  content      text not null,               -- plain text sent to embedding
  url_fragment text not null,               -- "#anchor-id" for linking
  embedding    vector(1536),                -- text-embedding-3-small dimension
  created_at   timestamptz default now()
);

-- IVFFlat index for fast approximate nearest-neighbor search
-- Run AFTER inserting data (needs at least a few hundred rows to be useful)
create index if not exists manual_chunks_embedding_idx
  on manual_chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 50);

-- RPC function used by the API route to find similar chunks
create or replace function match_chunks(
  query_embedding vector(1536),
  match_count     int default 5,
  similarity_threshold float default 0.3
)
returns table (
  chunk_id     text,
  section      text,
  breadcrumb   text,
  content      text,
  url_fragment text,
  similarity   float
)
language sql stable
as $$
  select
    chunk_id,
    section,
    breadcrumb,
    content,
    url_fragment,
    1 - (embedding <=> query_embedding) as similarity
  from manual_chunks
  where 1 - (embedding <=> query_embedding) > similarity_threshold
  order by embedding <=> query_embedding
  limit match_count;
$$;
