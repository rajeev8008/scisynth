from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

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
    """Call an OpenAI-compatible chat completion endpoint with retries.

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
        raise RuntimeError(
            "Missing LLM_API_KEY/OPENAI_API_KEY for answer generation. "
            "Set one in your environment or .env file.",
        )
    payload = _build_payload(settings, prompt, temperature, max_output_tokens)
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = _build_headers(api_key)
    timeout = httpx.Timeout(
        connect=15.0,
        read=settings.llm_timeout_seconds,
        write=30.0,
        pool=15.0,
    )
    max_retries = settings.llm_max_retries
    last_exc: BaseException | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                response.raise_for_status()
                return _extract_text(response.json())
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            code = exc.response.status_code
            if code in (429, 502, 503, 504) and attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning(
                    "LLM HTTP %s (attempt %s/%s); retrying in %ss",
                    code,
                    attempt + 1,
                    max_retries,
                    wait,
                )
                time.sleep(wait)
                continue
            logger.exception(
                "LLM request failed model=%s status=%s body=%s",
                settings.llm_model,
                code,
                exc.response.text[:500] if exc.response else "",
            )
            raise RuntimeError(
                f"LLM request failed with HTTP {code}. Check model name, quota, and base URL.",
            ) from exc
        except (httpx.TimeoutException, httpx.NetworkError, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning(
                    "LLM network/timeout (attempt %s/%s): %s; retrying in %ss",
                    attempt + 1,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue
            logger.exception("LLM request failed after retries model=%s", settings.llm_model)
            raise RuntimeError(
                "LLM request timed out or network failed after retries. "
                "Increase LLM_TIMEOUT_SECONDS or check connectivity.",
            ) from exc
        except Exception as exc:
            logger.exception("LLM request failed for model=%s", settings.llm_model)
            raise RuntimeError(f"Unexpected LLM error: {exc!s}") from exc
    raise RuntimeError(f"LLM request failed: {last_exc!s}")


def generate_answer_text_stream(
    settings: Settings,
    prompt: str,
    *,
    temperature: float,
    max_output_tokens: int,
) -> Iterator[str]:
    """Stream assistant text deltas from an OpenAI-compatible ``stream: true`` endpoint."""
    api_key = settings.llm_api_key or settings.openai_api_key
    if not api_key:
        raise RuntimeError(
            "Missing LLM_API_KEY/OPENAI_API_KEY for answer generation. "
            "Set one in your environment or .env file.",
        )
    payload = _build_payload(settings, prompt, temperature, max_output_tokens)
    payload["stream"] = True
    url = settings.llm_base_url.rstrip("/") + "/chat/completions"
    headers = _build_headers(api_key)
    timeout = httpx.Timeout(
        connect=15.0,
        read=settings.llm_timeout_seconds,
        write=30.0,
        pool=15.0,
    )
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
                    if not isinstance(delta, dict):
                        continue
                    piece = delta.get("content")
                    if isinstance(piece, str) and piece:
                        yield piece


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
