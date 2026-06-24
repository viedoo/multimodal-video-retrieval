# YouTube Processing Pipeline

Pipeline for downloading YouTube videos, creating low-resolution proxies, detecting scenes, extracting keyframes, running OCR, translating subtitles, and optionally creating Gemini summaries.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install -e ".[ocr]"
```

Paddle GPU is installed separately when GPU OCR is needed.

## Full Pipeline

```powershell
.\.venv\Scripts\stage-pipeline.exe --urls-file urls.txt --db dataset\stage_pipeline.sqlite3 --reset-db --force --download-workers 4 --proxy-workers 1 --scene-workers 2 --keyframe-workers 4 --ocr-workers 1
```

Add subtitle translation:

```powershell
--translate-subtitles
```

## Gemini Summary

Set the API key in the current PowerShell session:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Run summaries after the DB has video URLs:

```powershell
.\.venv\Scripts\stage-pipeline.exe --db dataset\stage_pipeline.sqlite3 --start-at gemini --gemini-workers 1 --force
```

Outputs are written to:

```text
gemini_summary/<video_id>/summary.json
dataset/gemini_summaries.jsonl
```

## Outputs

Generated runtime outputs are ignored by Git:

```text
videos/
subtitles/
proxies/
scenes/
keyframes/
dataset/
logs/
gemini_summary/
```

## Normalize Retrieval Documents

Build one retrieval document per scene by aligning subtitles and OCR with keyframe timestamps:

```powershell
.\.venv\Scripts\document-pipeline.exe --output dataset\documents.jsonl
```

Outputs:

```text
dataset/documents.jsonl
dataset/documents_manifest.json
```

Each document includes proxy/source frame indexes, timestamps, keyframe paths, cleaned OCR, aligned subtitle tracks, combined searchable text, and direct YouTube timestamp URLs.

## Build And Search Retrieval Indexes

Install FAISS support:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[retrieval]"
```

Build SQLite FTS5/BM25 and NER fields:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe build
```

Build a FAISS index with Gemini embeddings:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
.\.venv\Scripts\retrieval-pipeline.exe embed --provider gemini
```

For an offline functional test, use deterministic hashing vectors:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe embed --provider hashing
```

Hybrid BM25 + FAISS + entity reranking:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe search "Xuân Diệu" --mode hybrid --limit 10
```

Add the multilingual BGE cross-encoder for final query/passage reranking. On Windows the CLI loads BGE before
FAISS to avoid a native library load-order crash. Ollama query embeddings are released from GPU memory before
reranking:

```powershell
$env:HF_HOME="D:\pipeline1\.models\huggingface"
.\.venv\Scripts\retrieval-pipeline.exe search "AI nào có khả năng tự tìm lỗ hổng bảo mật?" `
  --mode hybrid --reranker bge --reranker-device cuda --limit 10
```

By default BGE scores the first 50 candidates, keeps only candidates that passed reranking, drops results below a
`0.20` final score, and removes results whose BGE logit falls more than `6.0` behind the best final result. Adjust
these controls with `--rerank-top`, `--reranker-min-final-score`, and `--reranker-max-score-drop`. Returning fewer
than `--limit` results is intentional when the remaining candidates are weak.

Retrieval candidates are scene-level documents containing the scene subtitles and deduplicated OCR from all of
its keyframes. The final passage, timestamp, frame index, and YouTube link are selected from the best matching
keyframe in that scene.

### Keyframe-Level Index

`retrieval-pipeline build` now creates both scene rows and keyframe rows in SQLite. Scene rows are stored in
`documents`; keyframe rows are stored in `keyframe_documents`. Each keyframe row contains local subtitle text,
OCR from that exact frame, timestamp, source frame index, and a `parent_document_id` pointing back to its scene.

Build a FAISS index for keyframes with local Nomic embeddings:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe embed --level keyframe --provider ollama --model nomic-embed-text `
  --dimension 768 --index dataset\faiss_keyframes_nomic.index `
  --metadata dataset\vector_metadata_keyframes_nomic.json
```

Search only keyframes with BM25:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe search "OPENCLAW" --mode bm25 --search-level keyframe --limit 10
```

Search both scene-level and keyframe-level indexes:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe search "OpenClaw khac tro ly AI truyen thong nhu the nao?" `
  --mode hybrid --search-level both --reranker bge --reranker-device cuda `
  --index dataset\faiss_nomic.index --metadata dataset\vector_metadata_nomic.json `
  --keyframe-index dataset\faiss_keyframes_nomic.index `
  --keyframe-metadata dataset\vector_metadata_keyframes_nomic.json --limit 10
```

Use `level=keyframe` in the output to identify results that came directly from a keyframe document.

### Adaptive Retrieval

Use `--adaptive` to let the CLI choose a retrieval strategy from the query:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe search "OPENCLAW" --adaptive `
  --index dataset\faiss_nomic.index --metadata dataset\vector_metadata_nomic.json `
  --keyframe-index dataset\faiss_keyframes_nomic.index `
  --keyframe-metadata dataset\vector_metadata_keyframes_nomic.json --limit 10
```

Adaptive routing currently uses simple, explainable rules:

- short entity/name queries: BM25 + entity search, include keyframe OCR, skip BGE for speed
- OCR/text/frame/timestamp queries: search both scene and keyframe rows
- semantic/paraphrase questions: enable hybrid search and BGE reranking
- entity-like queries: increase entity score weight

Add `--show-plan` to print the selected mode, search level, reranker, weights, and routing reasons.

TF-IDF and entity-only retrieval:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe search "AI tự tìm lỗ hổng bảo mật" --mode tfidf
.\.venv\Scripts\retrieval-pipeline.exe search "Ho Chi Minh" --mode entity
```

Build the index with Vietnamese ELECTRA NER:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ner,retrieval]"
$env:HF_HOME="D:\pipeline1\.models\huggingface"
.\.venv\Scripts\retrieval-pipeline.exe build --ner-provider electra --ner-device auto
```

Entity aliases are editable in `config/entity_aliases.json`. The entity index normalizes accents and maps OCR/name
variants to a canonical entity.

Use local Nomic embeddings through Ollama:

```powershell
ollama pull nomic-embed-text
.\.venv\Scripts\retrieval-pipeline.exe embed --provider ollama --model nomic-embed-text `
  --dimension 768 --index dataset\faiss_nomic.index `
  --metadata dataset\vector_metadata_nomic.json
```

Search the Nomic index explicitly (it does not replace the default FAISS index):

```powershell
.\.venv\Scripts\retrieval-pipeline.exe search "AI tự tìm lỗ hổng bảo mật" --mode semantic `
  --index dataset\faiss_nomic.index --metadata dataset\vector_metadata_nomic.json
```

Evaluate each embedding provider with its matching index/metadata files. Do not compare semantic results while
pointing at the default hashing index.

## Evaluate Retrieval

Create 60 bootstrap queries distributed across all indexed videos:

```powershell
.\.venv\Scripts\retrieval-eval.exe seed --count 60
```

Review `dataset/evaluation_queries.jsonl`: rewrite each query naturally, add every relevant scene to
`relevant_document_ids`, then set `reviewed` to `true`. Bootstrap queries are useful for plumbing checks but are
biased toward lexical retrieval and must not be treated as a final human gold set.

Compare BM25, semantic, and hybrid retrieval:

```powershell
.\.venv\Scripts\retrieval-eval.exe run
```

Evaluate hybrid retrieval with BGE reranking:

```powershell
.\.venv\Scripts\retrieval-eval.exe run --modes hybrid --reranker bge --reranker-device cuda
```

Evaluate adaptive routing:

```powershell
.\.venv\Scripts\retrieval-eval.exe run --modes hybrid --adaptive `
  --index dataset\faiss_nomic.index --metadata dataset\vector_metadata_nomic.json `
  --keyframe-index dataset\faiss_keyframes_nomic.index `
  --keyframe-metadata dataset\vector_metadata_keyframes_nomic.json
```

## Incremental Indexing

Use incremental indexing when `documents.jsonl` contains only new or changed videos. The command deletes old rows for
those `video_id`s, inserts the new scene/keyframe rows, rebuilds FTS5, and recomputes TF-IDF stats without clearing the
rest of the database:

```powershell
.\.venv\Scripts\retrieval-pipeline.exe build --documents dataset\documents_new.jsonl `
  --db dataset\search_index.sqlite3 --incremental --no-overwrite
```

If embedding vectors changed, rebuild the matching FAISS file for the affected level. Current FAISS files are still
written as full indexes; appending vectors incrementally is a later optimization.

## Persistent Retrieval Server

Run a long-lived HTTP server so FAISS indexes and BGE can stay cached in process:

```powershell
$env:HF_HOME="D:\pipeline1\.models\huggingface"
.\.venv\Scripts\python.exe -m youtube_pipeline.retrieval_server `
  --host 127.0.0.1 --port 8765 `
  --db dataset\search_index.sqlite3 `
  --index dataset\faiss_nomic.index --metadata dataset\vector_metadata_nomic.json `
  --keyframe-index dataset\faiss_keyframes_nomic.index `
  --keyframe-metadata dataset\vector_metadata_keyframes_nomic.json
```

Query it:

```powershell
$body = @{ query = "OPENCLAW"; adaptive = $true; limit = 5 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8765/search -Method Post -ContentType "application/json" -Body $body
```

Use `--preload` to load FAISS and BGE before serving. For short entity/OCR queries, adaptive routing can skip semantic
search and BGE, which keeps latency low.

Use only human-reviewed labels for final metrics:

```powershell
.\.venv\Scripts\retrieval-eval.exe run --reviewed-only
```

Outputs:

```text
dataset/evaluation_report.json
dataset/evaluation_details.jsonl
```
