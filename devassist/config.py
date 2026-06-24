"""Конфигурация devassist.

Источники настроек (в порядке приоритета):
  1. Явные аргументы командной строки (передаются в Config.load()).
  2. Переменные окружения.
  3. Файл .env в корне проекта (загружается без внешних зависимостей).
  4. Значения по умолчанию.

Секреты (ключ GigaChat) НЕ хранятся в коде — только в окружении / .env.
"""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass, field
from pathlib import Path

# URL по умолчанию для каждой схемы авторизации.
DEFAULT_OAUTH_URL = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_MTLS_URL = "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1"


def load_dotenv(path: Path) -> None:
    """Минимальный парсер .env: KEY=VALUE, поддержка # комментариев и кавычек.

    Не перезаписывает уже установленные переменные окружения.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    """Сводная конфигурация приложения."""

    # --- GigaChat ---
    # Две схемы авторизации (выбирается автоматически по наличию реквизитов):
    #   * OAuth (внешний контур): задан access_key -> Basic-ключ обменивается на
    #     Bearer-токен на auth_url, запросы идут с заголовком Authorization;
    #   * mTLS (внутренний контур): заданы cert+key -> клиентский сертификат
    #     предъявляется на TLS-уровне, токен и заголовок Authorization не нужны.
    access_key: str | None = None
    cert: str | None = None  # путь к клиентскому сертификату (mTLS, PEM)
    key: str | None = None  # путь к приватному ключу (mTLS)
    scope: str = "GIGACHAT_API_PERS"
    model: str = "GigaChat-3-Ultra"
    auth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    base_url: str = DEFAULT_OAUTH_URL
    verify_ssl: bool = False
    timeout: int = 120

    # --- Агент / окружение ---
    project_root: Path = field(default_factory=Path.cwd)
    auto_approve: bool = False  # пропускать подтверждения (опасно)
    max_steps: int = 50  # предохранитель агентного цикла
    max_tool_failures: int = 4  # стоп при N неудачных вызовах подряд (анти-залипание)
    temperature: float = 0.2
    stream: bool = True  # потоковый (посимвольный) вывод ответа модели

    @classmethod
    def load(
        cls,
        *,
        project_root: Path | None = None,
        model: str | None = None,
        auto_approve: bool = False,
        stream: bool = True,
    ) -> Config:
        root = Path(project_root or Path.cwd()).resolve()
        # .env ищем в корне проекта, затем в cwd
        for candidate in {root / ".env", Path.cwd() / ".env"}:
            load_dotenv(candidate)

        access_key = os.environ.get("GIGACHAT_ACCESS_KEY") or None
        cert = os.environ.get("GIGACHAT_CERT") or None
        key = os.environ.get("GIGACHAT_KEY") or None

        # Выбор эндпоинта: явный GIGACHAT_URL имеет приоритет; иначе берётся
        # дефолт под выбранную схему — OAuth при наличии ключа, mTLS при cert+key.
        base_url = os.environ.get("GIGACHAT_URL")
        if not base_url:
            base_url = DEFAULT_MTLS_URL if (not access_key and cert and key) else DEFAULT_OAUTH_URL

        return cls(
            access_key=access_key,
            cert=cert,
            key=key,
            scope=os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
            model=model or os.environ.get("GIGACHAT_MODEL", "GigaChat-3-Ultra"),
            auth_url=os.environ.get(
                "GIGACHAT_AUTH_URL",
                "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
            ),
            base_url=base_url,
            verify_ssl=_env_bool("GIGACHAT_VERIFY_SSL", False),
            timeout=int(os.environ.get("GIGACHAT_TIMEOUT", "120")),
            project_root=root,
            auto_approve=auto_approve,
            stream=stream,
            temperature=float(os.environ.get("DEVASSIST_TEMPERATURE", "0.2")),
        )

    @property
    def auth_mode(self) -> str:
        """Схема авторизации, выбранная по наличию реквизитов.

        ``"oauth"`` — есть access_key; ``"mtls"`` — есть cert+key (и нет ключа);
        ``"none"`` — реквизитов нет.
        """
        if self.access_key:
            return "oauth"
        if self.cert and self.key:
            return "mtls"
        return "none"

    def build_ssl_verify(self) -> ssl.SSLContext | bool:
        """Значение для httpx ``verify=``.

        Для mTLS строит SSL-контекст с клиентским сертификатом (cert+key); файлы
        проверяются на существование и загружаемость. Для OAuth возвращает флаг
        проверки TLS-сертификата сервера (verify_ssl).
        """
        if self.auth_mode != "mtls":
            return self.verify_ssl

        cert_path = Path(self.cert)  # type: ignore[arg-type]
        key_path = Path(self.key)  # type: ignore[arg-type]
        if not cert_path.is_file():
            raise FileNotFoundError(f"Сертификат GigaChat не найден: {cert_path}")
        if not key_path.is_file():
            raise FileNotFoundError(f"Ключ GigaChat не найден: {key_path}")

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if not self.verify_ssl:
            # Внутренний контур: самоподписанный серверный сертификат.
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        try:
            context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
        except ssl.SSLError as e:
            raise RuntimeError(f"Не удалось загрузить сертификат GigaChat: {e}") from e
        return context

    def require_credentials(self) -> None:
        if self.auth_mode == "none":
            raise RuntimeError(
                "Не заданы реквизиты GigaChat. Укажите либо GIGACHAT_ACCESS_KEY "
                "(внешний контур, OAuth), либо пару GIGACHAT_CERT и GIGACHAT_KEY "
                "(внутренний контур, mTLS) в .env или переменных окружения. "
                "См. .env.example."
            )
