"""Vector retrieval + retrieval result grading.

retrieve() is the simplest baseline: the original question is used as-is, embedded and
queried directly for the top-k. grade_documents() is for Agentic RAG: it uses DeepSeek
to judge in one pass whether the retrieved results are on-topic.
"""

from config import require_env
from llm import chat_json
from rag.embedding import embed
from rag.vector_store import get_milvus_client

# Upper limit of body text fed into the grading prompt. Grading only cares whether the
# results are on-topic; there is no need to stuff the whole document in and burn tokens.
_CONTENT_PREVIEW = 300

_GRADE_SYSTEM = (
    "You are a retrieval result relevance reviewer. For each candidate document, judge "
    "whether it can help answer the user's question: as long as the document content is "
    "related to the topic of the question, it counts as relevant; it does not have to "
    "answer directly. Do not infer information beyond the documents, and give a reason "
    "for each judgment. Return it as json."
)


def _hit_to_doc(hit: dict) -> dict:
    return {
        "id": hit["id"],
        "title": hit["entity"]["title"],
        "content": hit["entity"]["content"],
        "category": hit["entity"]["category"],
        "source": hit["entity"]["source"],
        "score": hit["distance"],
    }


def _truncate(text: str) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= _CONTENT_PREVIEW:
        return text
    return text[:_CONTENT_PREVIEW] + "..."


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve the top_k knowledge documents most relevant to query.

    Returns [{"id","title","content","category","source","score"}, ...], where score is
    the cosine similarity; larger means more relevant.
    """
    db_name = require_env("DB_NAME")
    collection = require_env("COLLECTION_NAME")
    client = get_milvus_client(db_name=db_name)

    # Idempotent; avoids errors when the collection is not loaded
    client.load_collection(collection)

    hits = client.search(
        collection_name=collection,
        data=[embed([query])[0]],
        limit=top_k,
        output_fields=["title", "content", "category", "source"],
    )

    return [_hit_to_doc(hit) for hit in hits[0]]


def retrieve_many(queries: list[str], top_k: int = 3) -> list[list[dict]]:
    """Embed many queries at once, load the collection only once, and search per query.

    multi_query can yield several queries at once; if each were to go through
    retrieve(), a Milvus client + load_collection would be rebuilt every time, which
    would be a pure waste. The result corresponds one-to-one to queries.
    """
    if not queries:
        return []

    db_name = require_env("DB_NAME")
    collection = require_env("COLLECTION_NAME")
    client = get_milvus_client(db_name=db_name)
    client.load_collection(collection)

    hits_per_query = client.search(
        collection_name=collection,
        data=embed(queries),
        limit=top_k,
        output_fields=["title", "content", "category", "source"],
    )

    return [[_hit_to_doc(hit) for hit in hits] for hits in hits_per_query]


def grade_documents(question: str, docs: list[dict], *, threshold: int = 1) -> dict:
    """Use a single LLM call to batch-decide whether each doc is relevant to question.

    Returns:
      {"relevant": [...], "irrelevant": [...],
       "grades": [bool, ...],     # same length, same order as the input docs; the caller
                                  # can align by index
       "sufficient": bool,        # relevant count >= threshold
       "reason": str}             # the model's overall explanation, possibly empty

    Results are backfilled by index rather than by return order, so the model returning
    them out of order will still be positioned correctly. Missing indices are treated as
    relevant (fail-open): it is not worth re-running a round for a missing item.
    """
    if not docs:
        return {
            "relevant": [],
            "irrelevant": [],
            "grades": [],
            "sufficient": False,
            "reason": "Retrieval returned no results; nothing to grade",
        }

    listing = "\n\n".join(
        f"[{i}] title: {doc.get('title', '')}\nbody: {_truncate(doc.get('content', ''))}"
        for i, doc in enumerate(docs)
    )

    data = chat_json(
        [
            {"role": "system", "content": _GRADE_SYSTEM},
            {
                "role": "user",
                "content": f"User question: {question}\n\nCandidate documents:\n{listing}",
            },
        ],
        schema_hint=(
            '{"grades": [{"index": 0, "relevant": true, "reason": "why it is relevant"}],'
            ' "sufficient": true}'
        ),
    )

    raw_grades = data.get("grades") or []
    reason = raw_grades[0].get("reason") or ""

    verdicts: dict[int, bool] = {}
    for item in raw_grades:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        # bool is a subclass of int, so it must be excluded, otherwise True/False would
        # be treated as 0/1 indices
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if 0 <= index < len(docs):
            verdicts[index] = bool(item.get("relevant"))

    grades = [verdicts.get(i, True) for i in range(len(docs))]

    relevant = [doc for doc, ok in zip(docs, grades) if ok]
    irrelevant = [doc for doc, ok in zip(docs, grades) if not ok]

    return {
        "relevant": relevant,
        "irrelevant": irrelevant,
        "grades": grades,
        # The model-reported sufficient is not trustworthy (it does not know the
        # threshold value), so the local count decides
        "sufficient": len(relevant) >= threshold,
        "reason": str(reason or ""),
    }