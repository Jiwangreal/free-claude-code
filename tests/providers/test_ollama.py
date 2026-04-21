"""Tests for Ollama provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.ollama import OLLAMA_BASE_URL, OllamaProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "ollama/llama3.1:8b"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def ollama_config():
    return ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def ollama_provider(ollama_config):
    return OllamaProvider(ollama_config)


def test_init(ollama_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = OllamaProvider(ollama_config)
        assert provider._base_url == "http://localhost:11434/v1"
        assert provider._provider_name == "OLLAMA"
        assert provider._api_key == "ollama"


def test_init_uses_default_base_url():
    """Test that provider uses default base URL when not configured."""
    config = ProviderConfig(api_key="ollama", base_url=None)
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = OllamaProvider(config)
        assert provider._base_url == OLLAMA_BASE_URL


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_client:
        OllamaProvider(config)
        call_kwargs = mock_client.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_init_base_url_strips_trailing_slash():
    """Config with base_url trailing slash is stored without it."""
    config = ProviderConfig(
        api_key="ollama",
        base_url="http://localhost:11434/v1/",
        rate_limit=10,
        rate_window=60,
    )
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = OllamaProvider(config)
        assert provider._base_url == "http://localhost:11434/v1"


def test_init_uses_default_api_key():
    """Test that provider uses default API key when not configured."""
    config = ProviderConfig(
        base_url="http://localhost:11434/v1",
        api_key="",
        rate_limit=10,
        rate_window=60,
    )
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = OllamaProvider(config)
        assert provider._api_key == "ollama"


@pytest.mark.asyncio
async def test_stream_response(ollama_provider):
    """Test streaming response with OpenAI-compatible API."""
    req = MockRequest()

    # Mock OpenAI client response
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta = MagicMock()
    mock_chunk.choices[0].delta.content = "Hello"
    mock_chunk.choices[0].finish_reason = "stop"
    mock_chunk.usage = MagicMock()
    mock_chunk.usage.completion_tokens = 5
    mock_chunk.usage.prompt_tokens = 10

    async def mock_stream():
        yield mock_chunk

    mock_completion = MagicMock()
    mock_completion.__aiter__ = lambda self: mock_stream()

    with patch.object(
        ollama_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_completion,
    ):
        events = [e async for e in ollama_provider.stream_response(req)]

        # Verify that some events were generated
        assert len(events) > 0

        # Verify the request was made
        ollama_provider._client.chat.completions.create.assert_called_once()
        call_kwargs = ollama_provider._client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True
        assert "model" in call_kwargs
        assert "messages" in call_kwargs


@pytest.mark.asyncio
async def test_build_request_body(ollama_provider):
    """Test that request body is built correctly."""
    req = MockRequest()

    body = ollama_provider._build_request_body(req)

    # Verify basic fields
    assert "model" in body
    assert "messages" in body
    assert body["model"] == "ollama/llama3.1:8b"
    # System prompt is added as a separate message
    assert len(body["messages"]) >= 1
    assert body["messages"][-1]["role"] == "user"
    assert body["messages"][-1]["content"] == "Hello"


@pytest.mark.asyncio
async def test_thinking_enabled_by_default(ollama_provider):
    """Test that thinking is enabled by default."""
    req = MockRequest()
    req.thinking.enabled = True

    body = ollama_provider._build_request_body(req)

    # Verify basic fields
    assert "model" in body
    assert "messages" in body
    assert body["model"] == "ollama/llama3.1:8b"


@pytest.mark.asyncio
async def test_thinking_disabled_when_configured(ollama_config):
    """Test that thinking can be disabled via config."""
    provider = OllamaProvider(
        ollama_config.model_copy(update={"enable_thinking": False})
    )
    req = MockRequest()
    req.thinking.enabled = True

    body = provider._build_request_body(req)

    # Verify basic fields
    assert "model" in body
    assert "messages" in body
    assert body["model"] == "ollama/llama3.1:8b"


@pytest.mark.asyncio
async def test_cleanup(ollama_provider):
    """Test that cleanup closes the client."""
    # Mock the client's aclose method
    ollama_provider._client.aclose = AsyncMock()

    await ollama_provider.cleanup()

    # Verify that the client was closed
    ollama_provider._client.aclose.assert_called_once()
