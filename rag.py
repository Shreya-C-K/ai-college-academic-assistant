"""Vector store access, retrieval with relevance threshold, helper formatters, baseline LLM."""
from functools import lru_cache
from typing import List, Tuple

from langchain_core.output_parsers import StrOutputParser

import config


@lru_cache(maxsize=1)
def get_embeddings():
       from langchain_community.embeddings import FastEmbedEmbeddings

       return FastEmbedEmbeddings(model_name=config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_community.vectorstores import FAISS

    if not (config.VECTORSTORE_DIR / "index.faiss").exists():
        raise FileNotFoundError("Vector store not found. Run:  python ingest.py")
    return FAISS.load_local(
        str(config.VECTORSTORE_DIR), get_embeddings(), allow_dangerous_deserialization=True
    )


def retrieve(query: str, k: int = config.TOP_K) -> Tuple[list, list]:
    """Return (relevant_docs_with_scores, all_results). Lower score = closer match."""
    results = get_vectorstore().similarity_search_with_score(query, k=k)
    kept = [(d, float(s)) for d, s in results if s <= config.MAX_L2_DISTANCE]
    return kept, [(d, float(s)) for d, s in results]


def format_context(docs_scores) -> str:
    parts = []
    for i, (d, _) in enumerate(docs_scores, 1):
        parts.append(f"[Excerpt {i} | {d.metadata.get('doc_name', 'Document')}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def unique_sources(docs_scores) -> List[str]:
    seen, out = set(), []
    for d, _ in docs_scores:
        label = d.metadata.get("doc_name", "Document")
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def format_history(history, n: int = 6) -> str:
    if not history:
        return "(no previous messages)"
    lines = []
    for m in history[-n:]:
        who = "Student" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content'][:500]}")
    return "\n".join(lines)


def basic_llm_answer(question: str) -> str:
    """Baseline chatbot: plain LLM, NO documents (used for the comparison)."""
    from llm import get_llm
    from prompts import BASIC_PROMPT

    chain = BASIC_PROMPT | get_llm(0.2) | StrOutputParser()
    return chain.invoke({"college": config.COLLEGE_NAME, "question": question})
