"""
utils/llm.py
─────────────
Single place to initialise the Anthropic/LangChain LLM so every agent
uses the same model and settings.
"""

from langchain_anthropic import ChatAnthropic
from config import settings


def get_llm(temperature: float = 0.3) -> ChatAnthropic:
    """
    Return a ChatAnthropic instance.

    temperature=0.3  →  mostly deterministic, with a little creativity for
                        narrative summaries.

    All nodes call this helper so you can swap the model in one place.
    """
    return ChatAnthropic(
        model=settings.MODEL_NAME,
        api_key=settings.ANTHROPIC_API_KEY,
        max_tokens=settings.MAX_TOKENS,
        temperature=temperature,
    )
