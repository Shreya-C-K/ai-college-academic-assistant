"""LLM factory + small helpers."""
import json
import re
from functools import lru_cache

import config


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.2):
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=temperature)
    from langchain_groq import ChatGroq

    return ChatGroq(model=config.GROQ_MODEL, temperature=temperature)


def parse_json(text):
    """Best-effort JSON extraction from an LLM reply (handles ``` fences and chatter)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", str(text)).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None
