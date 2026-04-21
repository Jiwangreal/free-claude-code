"""Request builder for Ollama provider."""

from typing import Any

from loguru import logger

from core.anthropic import build_base_request_body

OLLAMA_DEFAULT_MAX_TOKENS = 4096


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request for Ollama."""
    logger.debug(
        "OLLAMA_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        include_thinking=thinking_enabled,
        default_max_tokens=OLLAMA_DEFAULT_MAX_TOKENS,
    )

    logger.debug(
        "OLLAMA_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
