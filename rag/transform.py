"""Query rewriting: turns the user's original question into queries that are easier to
retrieve.

All three strategies share one contract: `(question) -> list[str]`, returning the
"queries to embed" rather than retrieval results. This way the upper-loop and the eval
don't need to branch per strategy -- they always get a list of strings to hand to the
retriever for encoding. The HyDE-generated hypothetical document body also goes through
the same path: it is itself a "piece of text that looks like an answer".

Note that at ingestion the corpus is indexed as `title\\nbody` (see rag/ingest.py),
while at retrieval time we encode the bare query string. A hypothetical document is
closer to the index side than a short question, which is exactly why HyDE works.
"""

from collections.abc import Callable

from llm import chat, chat_json

DEFAULT_STRATEGY = "rewrite"

# For the "let the model pick a strategy" prompt: strategy name -> one-line description +
# applicable scenario
STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "rewrite": "Rewrite a colloquial question into a retrieval-friendly, written query. "
               "Best for questions whose wording is close to the venue handbook.",
    "multi_query": "Generate several queries with the same meaning but different wording "
                   "and merge the results. Best for vague, polysemous questions.",
    "hyde": "First write a hypothetical answer document, then use it to retrieve. Best "
            "for questions whose answer wording differs a lot and is very colloquial.",
}

_REWRITE_SYSTEM = (
    "You are a retrieval query rewriter. Rewrite the user's colloquial question into "
    "a concise, formal, vector-search-friendly query. Output only the rewritten query "
    "itself, with no explanation and no quotation marks."
)

_HYDE_SYSTEM = (
    "You are the knowledge-base writer for a fitness venue. For the user's question, "
    "write a 150-250 character explanation in the style of the venue handbook, giving "
    "the answer details that likely hold (such as times, prices, rules). Do not use "
    "markdown, do not restate the question, output only the body text."
)

_MULTI_SYSTEM = (
    "You are a retrieval query expander. For the user's question, generate several "
    "retrieval queries with the same meaning but different wording and angles, "
    "covering synonyms, formal expressions, and possible technical terms."
)


def rewrite_query(question: str) -> list[str]:
    """Rewrite the question into a retrieval-friendly written query. Returns a list of
    length 1."""
    text = chat(
        [
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": question},
        ]
    ).strip()

    # The model occasionally emits an empty string (a known DeepSeek issue); falling
    # back to the original question at least keeps retrieval running
    return [text] if text else [question]


def multi_query(question: str, n: int = 3) -> list[str]:
    """Generate n synonymous-but-differently-worded queries. Deduplicate in order, and
    fall back to [original question] when the model produces nothing."""
    data = chat_json(
        [
            {"role": "system", "content": _MULTI_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {question}\nPlease generate {n} queries and "
                           f"return them as json.",
            },
        ],
        schema_hint='{"queries": ["query 1", "query 2", "query 3"]}',
    )

    # Json is used instead of free text because this is a variable-length list --
    # plain-text parsing would be too fragile
    raw = data.get("queries") or []

    seen: set[str] = set()
    queries: list[str] = []
    for item in raw:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            queries.append(text)

    return queries[:n] if queries else [question]


def hyde(question: str) -> list[str]:
    """Generate a hypothetical answer document, returning [body] (length 1)."""
    text = chat(
        [
            {"role": "system", "content": _HYDE_SYSTEM},
            {"role": "user", "content": question},
        ]
    ).strip()

    return [text] if text else [question]


STRATEGIES: dict[str, Callable[[str], list[str]]] = {
    "rewrite": rewrite_query,
    "multi_query": multi_query,
    "hyde": hyde,
}

# For name normalization: mapping built after removing case, spaces, hyphens and
# underscores. "HyDE" / "multi-query" / "multi query" should all map to the same strategy.
_NORMALIZED = {
    "".join(ch for ch in name.lower() if ch.isalnum()): name for name in STRATEGIES
}


def available_strategies() -> list[str]:
    """Available strategy names, in stable order (dict preserves insertion order), for
    the strategy-picking prompt and for "try another strategy"."""
    return list(STRATEGIES)


def resolve_strategy(name: str | None) -> str:
    """Map the name given by the model to the canonical name in the registry.

    Unknown, misspelled or empty values all fall back to DEFAULT_STRATEGY -- a wrong
    strategy name from the model must never break the loop.
    """
    key = "".join(ch for ch in (name or "").lower() if ch.isalnum())
    return _NORMALIZED.get(key, DEFAULT_STRATEGY)


def transform(question: str, strategy: str) -> list[str]:
    """Strategy name -> list of queries to encode.

    Falls back to [original question] when the strategy produces nothing, so the caller
    always gets at least one searchable query.
    """
    question = (question or "").strip()
    queries = STRATEGIES[resolve_strategy(strategy)](question)
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    if queries:
        return queries
    return [question] if question else []