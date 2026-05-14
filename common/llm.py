"""Shared LLM factory for all agents.

Uses OpenAI API directly with OPENAI_API_KEY.
Model is configurable via OPENAI_MODEL env var (default: gpt-4o-mini).
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenAI API."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1,
    )