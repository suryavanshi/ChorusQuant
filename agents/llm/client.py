"""Minimal HTTP client for interacting with LLM backends."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_DEFAULT_TIMEOUT = 60


class LLMClientError(RuntimeError):
    """Raised when the LLM backend returns an error response."""


def _prepare_payload(messages: List[Mapping[str, Any]], model: str, extra: Mapping[str, Any] | None = None) -> bytes:
    payload: Dict[str, Any] = {"model": model, "messages": list(messages)}
    if extra:
        payload.update(extra)
    return json.dumps(payload).encode("utf-8")


def _call_endpoint(
    base_url: str,
    payload: bytes,
    *,
    timeout: int = _DEFAULT_TIMEOUT,
    headers: Optional[Mapping[str, str]] = None,
    retries: int = 0,
) -> Dict[str, Any]:
    attempt = 0
    last_error: Optional[Exception] = None
    while attempt <= retries:
        request = Request(base_url, data=payload, method="POST")
        request.add_header("Content-Type", "application/json")
        if headers:
            for key, value in headers.items():
                request.add_header(key, value)

        try:
            with urlopen(request, timeout=timeout) as response:
                response_data = response.read().decode("utf-8")
            return json.loads(response_data)
        except HTTPError as exc:  # pragma: no cover - passthrough for manual testing
            body = exc.read().decode("utf-8") if exc.fp else ""
            last_error = LLMClientError(f"LLM backend error {exc.code}: {body}")
        except URLError as exc:  # pragma: no cover - passthrough for manual testing
            last_error = LLMClientError(f"Failed to reach LLM backend: {exc.reason}")

        attempt += 1
        if attempt > retries and last_error is not None:
            raise last_error

    raise LLMClientError("LLM request failed for unknown reasons.")


def call_llm(
    messages: Iterable[Mapping[str, Any]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    retries: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Invoke a chat-completion style endpoint.

    Parameters
    ----------
    messages:
        Sequence of role/content dicts to send to the model.
    model:
        Optional model override. Defaults to the ``LLM_MODEL`` environment variable.
    base_url:
        Optional endpoint override. Defaults to ``LLM_BASE_URL``.
    api_key:
        Optional API key override. Defaults to ``LLM_API_KEY``.
    timeout:
        Request timeout in seconds.
    retries:
        Number of additional retry attempts upon failure.
    kwargs:
        Extra keyword arguments forwarded to the payload.
    """

    resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
    if not resolved_base_url:
        raise ValueError("LLM base URL must be provided via argument or LLM_BASE_URL environment variable.")

    resolved_model = model or os.getenv("LLM_MODEL")
    if not resolved_model:
        raise ValueError("Model name must be provided via argument or LLM_MODEL environment variable.")

    resolved_api_key = api_key or os.getenv("LLM_API_KEY")
    headers = {"Authorization": f"Bearer {resolved_api_key}"} if resolved_api_key else None

    payload = _prepare_payload(list(messages), resolved_model, kwargs)
    return _call_endpoint(resolved_base_url, payload, timeout=timeout, headers=headers, retries=retries)


def call_vision(
    image_paths: Iterable[str],
    prompt: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    retries: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Invoke a multimodal endpoint for vision-capable models.

    Parameters mirror :func:`call_llm` with image-specific inputs. The ``base_url`` and
    ``api_key`` fall back to ``VISION_*`` environment variables before reusing the LLM
    defaults.
    """

    resolved_base_url = base_url or os.getenv("VISION_BASE_URL") or os.getenv("LLM_BASE_URL")
    if not resolved_base_url:
        raise ValueError("Vision base URL must be provided via argument or environment variable.")

    resolved_model = model or os.getenv("VISION_MODEL") or os.getenv("LLM_MODEL")
    if not resolved_model:
        raise ValueError("Vision model must be provided via argument or environment variable.")

    payload_dict = {
        "model": resolved_model,
        "prompt": prompt,
        "images": list(image_paths),
    }
    if kwargs:
        payload_dict.update(kwargs)
    payload = json.dumps(payload_dict).encode("utf-8")
    resolved_api_key = api_key or os.getenv("VISION_API_KEY") or os.getenv("LLM_API_KEY")
    headers = {"Authorization": f"Bearer {resolved_api_key}"} if resolved_api_key else None
    return _call_endpoint(resolved_base_url, payload, timeout=timeout, headers=headers, retries=retries)
