"""Ollama provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import OLLAMA_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body

OLLAMA_BASE_URL = OLLAMA_DEFAULT_BASE


class OllamaProvider(OpenAIChatTransport):
    """Ollama provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OLLAMA",
            base_url=config.base_url or OLLAMA_BASE_URL,
            api_key=config.api_key or "ollama",
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build OpenAI-format request body from Anthropic request."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
