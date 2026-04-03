from __future__ import annotations

import logging

import httpx

from scisynth.config import Settings

logger = logging.getLogger(__name__)


def generate_answer_text(
    settings: Settings,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
) -> str:
    """Call an OpenAI-compatible chat completion endpoint.

    Args:
        settings: Runtime settings with model and endpoint config.
        prompt: Prompt text including question and context.
        temperature: Sampling temperature for generation.
        max_output_tokens: Max tokens to generate in response.
    Returns:
        Assistant-generated answer text.

    Raises:
        RuntimeError: When API key is missing or response is invalid.
    """
    api_key = settings.llm_api_key or settings.openai_api_key
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY/OPENAI_API_KEY for answer generation.")
    payload = _build_payload(settings, prompt, temperature, max_output_tokens)
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = _build_headers(api_key)
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return _extract_text(response.json())
    except Exception as exc:
        logger.exception("LLM request failed for model=%s", settings.llm_model)
        raise


def _build_payload(
    settings: Settings,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> dict[str, object]:
    """Build chat completion payload from settings and prompt.

    Args:
        settings: Runtime settings with model config.
        prompt: User message content.
        temperature: Sampling temperature.
        max_output_tokens: Max generated tokens.
    Returns:
        JSON-serializable request payload.
    """
    return {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_output_tokens,
    }


def _build_headers(api_key: str) -> dict[str, str]:
    """Build HTTP headers for an OpenAI-compatible request.

    Args:
        api_key: Bearer token used for authorization.
    Returns:
        Header dictionary.
    """
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _extract_text(payload: dict[str, object]) -> str:
    """Extract assistant text from a chat completion response.

    Args:
        payload: Parsed JSON response object.
    Returns:
        Generated assistant text.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("LLM response did not include choices.")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("LLM response choice is malformed.")
    message = first.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("LLM response message is missing.")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("LLM response content is empty.")
    return content.strip()
