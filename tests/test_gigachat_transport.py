"""Офлайн-тесты сетевого слоя GigaChat на подменном транспорте (httpx.MockTransport).

Проверяют повторы, обработку обрывов и превращение любых ошибок транспорта в
GigaChatError (иначе они роняют REPL).
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from devassist.config import Config
from devassist.llm import gigachat
from devassist.llm.base import LLMError
from devassist.llm.gigachat import GigaChatError, GigaChatProvider
from devassist.llm.types import Message, Usage

AUTH_URL = "https://auth.test/oauth"
BASE_URL = "https://api.test/v1"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(gigachat.time, "sleep", lambda _s: None)


def _provider(handler) -> GigaChatProvider:
    cfg = Config(access_key="dummy", auth_url=AUTH_URL, base_url=BASE_URL, model="M-default")
    return GigaChatProvider(cfg, transport=httpx.MockTransport(handler))


def _auth_response():
    return httpx.Response(
        200, json={"access_token": "t", "expires_at": int((time.time() + 3600) * 1000)}
    )


def _sse(*chunks: dict) -> bytes:
    lines = [f"data: {json.dumps(c, ensure_ascii=False)}\n\n" for c in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


class _BrokenStream(httpx.SyncByteStream):
    """Тело ответа: сначала отдаёт кусок, потом рвёт соединение."""

    def __init__(self, first: bytes):
        self._first = first

    def __iter__(self):
        yield self._first
        raise httpx.ReadError("connection reset")


def _msgs():
    return [Message(role="user", content="привет")]


def test_gigachat_error_is_llm_error():
    assert issubclass(GigaChatError, LLMError)


def test_stream_broken_after_first_chunk_is_not_retried():
    chat_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        chat_calls.append(request)
        first = 'data: {"choices":[{"delta":{"content":"При"}}]}\n\n'.encode()
        return httpx.Response(200, stream=_BrokenStream(first))

    deltas: list[str] = []
    p = _provider(handler)
    with pytest.raises(GigaChatError):
        p.stream(_msgs(), on_delta=deltas.append)
    assert deltas == ["При"]  # текст не продублирован
    assert len(chat_calls) == 1  # повтора не было
    p.close()


def test_stream_retries_connect_error_before_output():
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("boom", request=request)
        body = _sse({"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]})
        return httpx.Response(200, content=body)

    deltas: list[str] = []
    p = _provider(handler)
    turn = p.stream(_msgs(), on_delta=deltas.append)
    assert turn.message.content == "ok"
    assert deltas == ["ok"]
    assert attempts["n"] == 2
    p.close()


@pytest.mark.parametrize("exc", [httpx.ProxyError, httpx.LocalProtocolError])
def test_non_retryable_transport_errors_wrapped(exc):
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        raise exc("nope")

    p = _provider(handler)
    with pytest.raises(GigaChatError):
        p.complete(_msgs())
    with pytest.raises(GigaChatError):
        p.stream(_msgs())
    p.close()


def test_non_json_response_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(200, text="<html>gateway</html>")

    p = _provider(handler)
    with pytest.raises(GigaChatError):
        p.complete(_msgs())
    p.close()


def test_auth_without_token_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": "no"})

    p = _provider(handler)
    with pytest.raises(GigaChatError):
        p.complete(_msgs())
    p.close()


def test_list_models_http_error_wrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        return httpx.Response(403, text="forbidden")

    p = _provider(handler)
    with pytest.raises(GigaChatError):
        p.list_models()
    p.close()


def test_model_override_per_call():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == AUTH_URL:
            return _auth_response()
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "x"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            },
        )

    p = _provider(handler)
    turn = p.complete(_msgs())
    p.complete(_msgs(), model="M-other")
    assert seen == ["M-default", "M-other"]
    assert turn.usage == Usage(prompt_tokens=7, completion_tokens=2, total_tokens=9)
    p.close()


def test_usage_from_raw_is_tolerant():
    assert Usage.from_raw(None) == Usage()
    assert Usage.from_raw("garbage") == Usage()  # type: ignore[arg-type]
    u = Usage.from_raw({"prompt_tokens": "5", "completion_tokens": None, "total_tokens": "x"})
    assert (u.prompt_tokens, u.completion_tokens, u.total_tokens) == (5, 0, 0)
