# Agentic RAG Core

A core logic template for Agentic RAG. Provides query transformation, retrieval, document grading, and a strategy retry loop.

## Core Methods

### transform.py — Query Transformation
- `transform(question, strategy) -> list[str]`: Strategy entry point. Returns queries ready for retrieval.
- `rewrite_query(question) -> list[str]`: Rewrites colloquial questions into formal search queries.
- `multi_query(question, n) -> list[str]`: Generates n synonymous but structurally diverse queries, order-preserving and deduplicated.
- `hyde(question) -> list[str]`: Generates a hypothetical answer document, aligning with the index-side "title\ncontent" structure.
- `resolve_strategy(name) -> str`: Normalizes strategy names, falls back to default on unknown input.

### retriever.py — Retrieval & Grading
- `retrieve(query, top_k) -> list[dict]`: Single-query retrieval, returns docs with cosine similarity scores.
- `retrieve_many(queries, top_k) -> list[list[dict]]`: Batch-encodes multiple queries, loads collection once.
- `grade_documents(question, docs, threshold) -> dict`: Batch LLM relevance grading. Returns `relevant`, `irrelevant`, `grades`, `sufficient`.

### search_policy.py — Decision Loop
- `search_policy(question, role, top_k, strategy) -> dict`: Main Agentic retrieval entry. Up to 2 rounds of strategy retries. Returns final docs and round history.
- `check_permission(role) -> None`: Mandatory permission check. Executes first, raises `PermissionError` if failed.

## Design Principles
- **Permission First**: `check_permission` strictly precedes all LLM and vector retrieval calls.
- **Local Fallbacks**: `grade_documents` uses fail-open logic on model misordering or missing `index`. `sufficient` is locally determined by count.
- **Zero Redundant Calls**: Strategy switching uses deterministic local rotation, avoiding extra LLM calls.
- **Type Defense**: Explicitly guards against `bool` being treated as an `int` index (`True/False` mapping to `0/1`).
- **Multi-route Fusion**: `_fuse` deduplicates by `id` and takes max score. `_pick_best` returns the better round based on "relevant count + max similarity".