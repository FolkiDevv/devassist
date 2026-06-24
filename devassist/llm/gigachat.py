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
from typing import Any, Callable, Dict, List, Optional, Sequence

import httpx

from devassist.devassist.config import Config
from devassist.devassist.llm.base import LLMProvider
from devassist.devassist.llm.types import AssistantTurn, FunctionCall, Message, ToolSpec


class GigaChatError(RuntimeError):
    """Ошибка взаимодействия с API GigaChat."""


# Транзиентные сетевые ошибки, которые имеет смысл повторять.
_RETRYABLE_EXC = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
)

# HTTP-статусы, которые имеет смысл повторить (лимит/временная недоступность).
_RETRY_STATUS = {429, 500, 502, 503, 504}


class GigaChatProvider(LLMProvider):
    def __init__(self, config: Config):
        config.require_credentials()
        self._cfg = config
        # В режиме mTLS verify — это SSL-контекст с клиентским сертификатом;
        # в режиме OAuth — булев флаг проверки серверного сертификата.
        self._client = httpx.Client(verify=config.build_ssl_verify(), timeout=config.timeout)
        self._mtls = config.auth_mode == "mtls"
        self._token: Optional[str] = None
        self._token_exp: float = 0.0  # unix-время истечения токена
        self._max_retries = 5  # повторы при транзиентных сбоях (сеть/429/5xx)

    @property
    def model(self) -> str:
        return self._cfg.model

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GigaChatProvider":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Сетевой слой с повторами
    # ------------------------------------------------------------------ #
    def _request_with_retry(
        self, do_request: Callable[[], httpx.Response]
    ) -> httpx.Response:
        """Выполняет HTTP-запрос, повторяя транзиентные сбои.

        Повторяем (с экспоненциальной задержкой):
          * сетевые ошибки/таймауты — сеть до API Сбера бывает нестабильна;
          * 429 Too Many Requests и 5xx — временная недоступность/лимит.
        Прочие HTTP-ответы (включая 4xx) возвращаем как есть — их разбирает
        вызывающая сторона.
        """
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                resp = do_request()
            except _RETRYABLE_EXC as e:
                last_exc = e
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)
                continue

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

        data = resp.json()
        self._token = data["access_token"]
        # expires_at приходит в миллисекундах unix-времени; если нет — живём 25 мин.
        exp_ms = data.get("expires_at")
        self._token_exp = (exp_ms / 1000.0) if exp_ms else (time.time() + 25 * 60)
        return self._token

    def _auth_headers(self, *, force: bool = False) -> Dict[str, str]:
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
    def _message_to_payload(cls, msg: Message) -> Dict[str, Any]:
        content = msg.content or ""
        if msg.role == "function":
            content = cls._as_json_content(content)
        out: Dict[str, Any] = {"role": msg.role, "content": content}
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
    def _tool_to_payload(tool: ToolSpec) -> Dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }

    @staticmethod
    def _build_function_call(fc_raw: Optional[Dict[str, Any]]) -> Optional[FunctionCall]:
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

    def _parse_response(self, data: Dict[str, Any]) -> AssistantTurn:
        try:
            choice = data["choices"][0]
            raw = choice["message"]
        except (KeyError, IndexError) as e:
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
            usage=data.get("usage", {}) or {},
        )

    # ------------------------------------------------------------------ #
    # Основной вызов
    # ------------------------------------------------------------------ #
    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[ToolSpec]] = None,
        *,
        temperature: float = 0.2,
    ) -> AssistantTurn:
        payload: Dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [self._message_to_payload(m) for m in messages],
            "temperature": temperature,
        }
        if tools:
            payload["functions"] = [self._tool_to_payload(t) for t in tools]
            payload["function_call"] = "auto"

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
            raise GigaChatError(
                f"GigaChat вернул {resp.status_code}: {resp.text}"
            ) from e

        return self._parse_response(resp.json())

    # ------------------------------------------------------------------ #
    # Потоковый вызов (SSE)
    # ------------------------------------------------------------------ #
    def stream(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[ToolSpec]] = None,
        *,
        temperature: float = 0.2,
        on_delta: Optional[Callable[[str], None]] = None,
    ) -> AssistantTurn:
        """Потоковая генерация. Вызывает on_delta(text) на каждый кусок текста и
        возвращает финальный AssistantTurn (идентичный complete()).

        Формат SSE GigaChat: строки ``data: {json}`` с ``choices[0].delta`` и
        терминатор ``data: [DONE]``. Текст приходит в ``delta.content``; вызов
        функции — целиком в одном чанке (``delta.function_call`` + finish_reason).
        """
        emit = on_delta or (lambda _s: None)
        payload: Dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [self._message_to_payload(m) for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["functions"] = [self._tool_to_payload(t) for t in tools]
            payload["function_call"] = "auto"

        url = f"{self._cfg.base_url}/chat/completions"
        delay = 1.0
        last_exc: Optional[Exception] = None
        refreshed = False

        for attempt in range(self._max_retries):
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **self._auth_headers(),
            }
            emitted = False
            try:
                with self._client.stream(
                    "POST", url, json=payload, headers=headers
                ) as resp:
                    if resp.status_code == 401 and not self._mtls and not refreshed:
                        resp.read()
                        self._ensure_token(force=True)
                        refreshed = True
                        continue
                    if (
                        resp.status_code in _RETRY_STATUS
                        and attempt < self._max_retries - 1
                    ):
                        resp.read()
                        time.sleep(self._retry_after(resp, delay))
                        delay = min(delay * 2, 8.0)
                        continue
                    if resp.status_code >= 400:
                        resp.read()
                        raise GigaChatError(
                            f"GigaChat вернул {resp.status_code}: {resp.text}"
                        )
                    turn, emitted = self._consume_sse(resp, emit)
                    return turn
            except _RETRYABLE_EXC as e:
                last_exc = e
                if emitted:
                    # часть текста уже отдана пользователю — повтор приведёт к дублю
                    raise GigaChatError(
                        f"Сетевой сбой во время потоковой передачи: {e}"
                    ) from e
                if attempt < self._max_retries - 1:
                    time.sleep(delay)
                    delay = min(delay * 2, 8.0)

        raise GigaChatError(
            f"Сетевая ошибка при потоковом обращении к GigaChat (после "
            f"{self._max_retries} попыток): {last_exc}"
        ) from last_exc

    def _consume_sse(self, resp: httpx.Response, emit: Callable[[str], None]):
        """Разбирает SSE-поток, отдавая текстовые дельты в ``emit``.

        Возвращает (AssistantTurn, emitted_flag).
        """
        content_parts: List[str] = []
        function_call = None
        state_id: Optional[str] = None
        finish_reason = "stop"
        usage: Dict[str, Any] = {}
        emitted = False

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
            choices = chunk.get("choices") or []
            if choices:
                choice = choices[0]
                delta = choice.get("delta", {}) or {}
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)
                    emitted = True
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
        return (
            AssistantTurn(message=message, finish_reason=finish_reason, usage=usage),
            emitted,
        )

    def list_models(self) -> List[str]:
        """Список доступных моделей (для диагностики/выбора)."""
        resp = self._request_with_retry(
            lambda: self._client.get(
                f"{self._cfg.base_url}/models",
                headers=self._auth_headers(),
            )
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
