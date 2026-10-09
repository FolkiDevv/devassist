"""Провайдер GigaChat (Sber).

Реализует авторизацию OAuth (Basic access-key -> Bearer access_token) с
автоматическим обновлением токена по истечении срока, и вызов
chat/completions с поддержкой function calling.

Формат function calling в GigaChat (проверено на живом API):
  * запрос содержит массив ``functions`` (а не ``tools``) и
    ``function_call: "auto"``;
  * ответ при вызове инструмента: ``message.function_call`` с полями
    ``name`` и ``arguments`` (arguments — уже распарсенный объект, НЕ строка),
    ``finish_reason == "function_call"`` и непрозрачный ``functions_state_id``;
  * результат инструмента возвращается сообщением ``role: "function"`` с
    полями ``name`` и ``content``.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from devassist.config import Config
from devassist.llm.base import LLMError, LLMProvider
from devassist.llm.types import AssistantTurn, FunctionCall, Message, ToolSpec, Usage


class GigaChatError(LLMError):
    """Ошибка взаимодействия с API GigaChat."""


# Транзиентные сетевые ошибки, которые имеет смысл повторять. Для потокового
# запроса повтор допустим, только пока пользователю не отдано ни одного куска.
# Все прочие httpx.HTTPError (ProxyError, LocalProtocolError, DecodingError...)
# не повторяются и превращаются в GigaChatError.
_RETRYABLE_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

# HTTP-статусы, которые имеет смысл повторить (лимит/временная недоступность).
_RETRY_STATUS = {429, 500, 502, 503, 504}


class GigaChatProvider(LLMProvider):
    def __init__(self, config: Config, *, transport: httpx.BaseTransport | None = None):
        """``transport`` — подмена HTTP-транспорта (``httpx.MockTransport`` в тестах)."""
        config.require_credentials()
        self._cfg = config
        # В режиме mTLS verify — это SSL-контекст с клиентским сертификатом;
        # в режиме OAuth — булев флаг проверки серверного сертификата.
        self._client = httpx.Client(
            verify=config.build_ssl_verify(),
            timeout=config.timeout,
            transport=transport,
        )
        self._mtls = config.auth_mode == "mtls"
        self._token: str | None = None
        self._token_exp: float = 0.0  # unix-время истечения токена
        self._max_retries = 5  # повторы при транзиентных сбоях (сеть/429/5xx)

    @property
    def model(self) -> str:
        return self._cfg.model

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GigaChatProvider:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Сетевой слой с повторами
    # ------------------------------------------------------------------ #
    def _request_with_retry(self, do_request: Callable[[], httpx.Response]) -> httpx.Response:
        """Выполняет HTTP-запрос, повторяя транзиентные сбои.

        Повторяем (с экспоненциальной задержкой):
          * сетевые ошибки/таймауты — сеть до API Сбера бывает нестабильна;
          * 429 Too Many Requests и 5xx — временная недоступность/лимит.
        Прочие HTTP-ответы (включая 4xx) возвращаем как есть — их разбирает
        вызывающая сторона. Неповторяемые ошибки транспорта → GigaChatError.
        """
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = do_request()
            except _RETRYABLE_EXC as e:
                last_exc = e
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                continue
            except httpx.HTTPError as e:
                raise GigaChatError(f"Ошибка соединения с GigaChat: {e}") from e

            if resp.status_code in _RETRY_STATUS and attempt < self._max_retries - 1:
                time.sleep(self._retry_after(resp, delay))
                delay = min(delay * 2, 8.0)
                continue
            return resp

        raise GigaChatError(
            f"Сетевая ошибка при обращении к GigaChat (после "
            f"{self._max_retries} попыток): {last_exc}"
        ) from last_exc

    @staticmethod
    def _json(resp: httpx.Response) -> dict[str, Any]:
        """Тело ответа как JSON-объект; иначе GigaChatError."""
        try:
            data = resp.json()
        except ValueError as e:
            raise GigaChatError(
                f"GigaChat вернул не-JSON ответ ({resp.status_code}): {resp.text[:500]}"
            ) from e
        if not isinstance(data, dict):
            raise GigaChatError(f"Неожиданный формат ответа GigaChat: {str(data)[:500]}")
        return data

    @staticmethod
    def _retry_after(resp: httpx.Response, default: float) -> float:
        """Задержка перед повтором: уважает заголовок Retry-After, если он есть."""
        ra = resp.headers.get("Retry-After")
        if ra:
            try:
                return min(float(ra), 30.0)
            except ValueError:
                pass
        return default

    # ------------------------------------------------------------------ #
    # Авторизация
    # ------------------------------------------------------------------ #
    def _ensure_token(self, force: bool = False) -> str:
        # Обновляем заранее (за 30 с до истечения), чтобы не словить 401 на лету.
        if not force and self._token and time.time() < self._token_exp - 30:
            return self._token

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {self._cfg.access_key}",
        }
        resp = self._request_with_retry(
            lambda: self._client.post(
                self._cfg.auth_url,
                headers=headers,
                data={"scope": self._cfg.scope},
            )
        )
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GigaChatError(
                f"Не удалось получить токен GigaChat ({resp.status_code}): {resp.text}"
            ) from e

        data = self._json(resp)
        token = data.get("access_token")
        if not token:
            raise GigaChatError(f"В ответе OAuth нет access_token: {str(data)[:500]}")
        self._token = token
        # expires_at приходит в миллисекундах unix-времени; если нет — живём 25 мин.
        exp_ms = data.get("expires_at")
        try:
            self._token_exp = float(exp_ms) / 1000.0 if exp_ms else time.time() + 25 * 60
        except (TypeError, ValueError):
            self._token_exp = time.time() + 25 * 60
        return token

    def _auth_headers(self, *, force: bool = False) -> dict[str, str]:
        """Заголовок авторизации для текущей схемы.

        В режиме mTLS — пустой словарь (клиента авторизует сертификат на
        TLS-уровне). В режиме OAuth — ``Authorization: Bearer <token>``.
        """
        if self._mtls:
            return {}
        return {"Authorization": f"Bearer {self._ensure_token(force=force)}"}

    # ------------------------------------------------------------------ #
    # Сериализация / разбор
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_json_content(content: str) -> str:
        """GigaChat требует, чтобы content сообщения role=function был валидным
        JSON-строкой. Если результат инструмента — произвольный текст, заворачиваем
        его в JSON-объект {"result": ...}."""
        text = content or ""
        try:
            json.loads(text)
            return text  # уже валидный JSON
        except (json.JSONDecodeError, ValueError):
            return json.dumps({"result": text}, ensure_ascii=False)

    @classmethod
    def _message_to_payload(cls, msg: Message) -> dict[str, Any]:
        content = msg.content or ""
        if msg.role == "function":
            content = cls._as_json_content(content)
        out: dict[str, Any] = {"role": msg.role, "content": content}
        if msg.name is not None:
            out["name"] = msg.name
        if msg.function_call is not None:
            out["function_call"] = {
                "name": msg.function_call.name,
                "arguments": msg.function_call.arguments,
            }
        if msg.functions_state_id is not None:
            out["functions_state_id"] = msg.functions_state_id
        return out

    @staticmethod
    def _tool_to_payload(tool: ToolSpec) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    @staticmethod
    def _build_function_call(fc_raw: dict[str, Any] | None) -> FunctionCall | None:
        if not fc_raw:
            return None
        args = fc_raw.get("arguments", {})
        # Обычно arguments — уже объект; подстраховка на случай строки.
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        return FunctionCall(name=fc_raw["name"], arguments=args or {})

    def _parse_response(self, data: dict[str, Any]) -> AssistantTurn:
        try:
            choice = data["choices"][0]
            raw = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise GigaChatError(f"Неожиданный формат ответа GigaChat: {data}") from e

        function_call = self._build_function_call(raw.get("function_call"))

        message = Message(
            role="assistant",
            content=raw.get("content") or "",
            function_call=function_call,
            functions_state_id=raw.get("functions_state_id"),
        )
        return AssistantTurn(
            message=message,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=Usage.from_raw(data.get("usage")),
        )

    # ------------------------------------------------------------------ #
    # Основной вызов
    # ------------------------------------------------------------------ #
    def _payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None,
        *,
        model: str | None,
        temperature: float,
        stream: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self._cfg.model,
            "messages": [self._message_to_payload(m) for m in messages],
            "temperature": temperature,
        }
        if stream:
            payload["stream"] = True
        if tools:
            payload["functions"] = [self._tool_to_payload(t) for t in tools]
            payload["function_call"] = "auto"
        return payload

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
    ) -> AssistantTurn:
        payload = self._payload(messages, tools, model=model, temperature=temperature)
        url = f"{self._cfg.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", **self._auth_headers()}

        resp = self._request_with_retry(
            lambda: self._client.post(url, json=payload, headers=headers)
        )
        if resp.status_code == 401 and not self._mtls:
            # токен протух — обновляем принудительно и повторяем один раз
            headers.update(self._auth_headers(force=True))
            resp = self._request_with_retry(
                lambda: self._client.post(url, json=payload, headers=headers)
            )

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise GigaChatError(f"GigaChat вернул {resp.status_code}: {resp.text}") from e

        return self._parse_response(self._json(resp))

    # ------------------------------------------------------------------ #
    # Потоковый вызов (SSE)
    # ------------------------------------------------------------------ #
    def stream(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.2,
        on_delta: Callable[[str], None] | None = None,
    ) -> AssistantTurn:
        """Потоковая генерация. Вызывает on_delta(text) на каждый кусок текста и
        возвращает финальный AssistantTurn (идентичный complete()).

        Формат SSE GigaChat: строки ``data: {json}`` с ``choices[0].delta`` и
        терминатор ``data: [DONE]``. Текст приходит в ``delta.content``; вызов
        функции — целиком в одном чанке (``delta.function_call`` + finish_reason).

        Повтор при сетевом сбое выполняется, только пока пользователю не отдан
        ни один кусок текста — иначе он увидел бы ответ дважды.
        """
        sink = on_delta or (lambda _s: None)
        state = {"emitted": False}

        def emit(piece: str) -> None:
            state["emitted"] = True
            sink(piece)

        payload = self._payload(messages, tools, model=model, temperature=temperature, stream=True)
        url = f"{self._cfg.base_url}/chat/completions"
        delay = 1.0
        last_exc: Exception | None = None
        refreshed = False

        for attempt in range(self._max_retries):
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **self._auth_headers(),
            }
            try:
                with self._client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code == 401 and not self._mtls and not refreshed:
                        resp.read()
                        self._ensure_token(force=True)
                        refreshed = True
                        continue
                    if resp.status_code in _RETRY_STATUS and attempt < self._max_retries - 1:
                        resp.read()
                        time.sleep(self._retry_after(resp, delay))
                        delay = min(delay * 2, 8.0)
                        continue
                    if resp.status_code >= 400:
                        resp.read()
                        raise GigaChatError(f"GigaChat вернул {resp.status_code}: {resp.text}")
                    return self._consume_sse(resp, emit)
            except _RETRYABLE_EXC as e:
                last_exc = e
                if state["emitted"]:
                    # часть текста уже отдана пользователю — повтор приведёт к дублю
                    raise GigaChatError(f"Сетевой сбой во время потоковой передачи: {e}") from e
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
            except httpx.HTTPError as e:
                raise GigaChatError(f"Ошибка соединения с GigaChat: {e}") from e

        raise GigaChatError(
            f"Сетевая ошибка при потоковом обращении к GigaChat (после "
            f"{self._max_retries} попыток): {last_exc}"
        ) from last_exc

    def _consume_sse(self, resp: httpx.Response, emit: Callable[[str], None]) -> AssistantTurn:
        """Разбирает SSE-поток, отдавая текстовые дельты в ``emit``."""
        content_parts: list[str] = []
        function_call = None
        state_id: str | None = None
        finish_reason = "stop"
        usage: dict[str, Any] = {}

        for raw_line in resp.iter_lines():
            if not raw_line or not raw_line.startswith("data:"):
                continue
            data = raw_line[len("data:") :].lstrip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {}) or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                    emit(piece)
                if delta.get("function_call") and function_call is None:
                    function_call = self._build_function_call(delta["function_call"])
                if delta.get("functions_state_id"):
                    state_id = delta["functions_state_id"]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
            if chunk.get("usage"):
                usage = chunk["usage"]

        message = Message(
            role="assistant",
            content="".join(content_parts),
            function_call=function_call,
            functions_state_id=state_id,
        )
        return AssistantTurn(
            message=message, finish_reason=finish_reason, usage=Usage.from_raw(usage)
        )

    def list_models(self) -> list[str]:
        """Список доступных моделей (для диагностики/выбора)."""
        resp = self._request_with_retry(
            lambda: self._client.get(
                f"{self._cfg.base_url}/models",
                headers=self._auth_headers(),
            )
        )
        if resp.status_code >= 400:
            raise GigaChatError(
                f"Не удалось получить список моделей ({resp.status_code}): {resp.text}"
            )
        models = self._json(resp).get("data") or []
        return [m["id"] for m in models if isinstance(m, dict) and "id" in m]
