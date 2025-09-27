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


def _call_endpoint(base_url: str, payload: bytes, *, timeout: int = _DEFAULT_TIMEOUT, headers: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    request = Request(base_url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    api_key = os.getenv("LLM_API_KEY")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    if headers:
        for key, value in headers.items():
            request.add_header(key, value)

    try:
        with urlopen(request, timeout=timeout) as response:
            response_data = response.read().decode("utf-8")
    except HTTPError as exc:  # pragma: no cover - passthrough for manual testing
        body = exc.read().decode("utf-8") if exc.fp else ""
        raise LLMClientError(f"LLM backend error {exc.code}: {body}") from exc
    except URLError as exc:  # pragma: no cover - passthrough for manual testing
        raise LLMClientError(f"Failed to reach LLM backend: {exc.reason}") from exc

    return json.loads(response_data)


def call_llm(
    messages: Iterable[Mapping[str, Any]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
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
    timeout:
        Request timeout in seconds.
    kwargs:
        Extra keyword arguments forwarded to the payload.
    """

    resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
    if not resolved_base_url:
        raise ValueError("LLM base URL must be provided via argument or LLM_BASE_URL environment variable.")

    resolved_model = model or os.getenv("LLM_MODEL")
    if not resolved_model:
        raise ValueError("Model name must be provided via argument or LLM_MODEL environment variable.")

    payload = _prepare_payload(list(messages), resolved_model, kwargs)
    return _call_endpoint(resolved_base_url, payload, timeout=timeout)


def call_vision(
    image_paths: Iterable[str],
    prompt: str,
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = _DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Invoke a multimodal endpoint for vision-capable models."""

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
    return _call_endpoint(resolved_base_url, payload, timeout=timeout)
