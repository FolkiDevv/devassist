"""Конфигурация devassist.

Источники настроек (в порядке приоритета):
  1. Явные аргументы командной строки (передаются в Config.load()).
  2. Переменные окружения.
  3. Файл .env в корне проекта.
  4. Файл .env в текущей директории (если она не корень проекта).
  5. Значения по умолчанию.

Файлы .env читаются в отдельный словарь и НЕ попадают в ``os.environ`` — иначе
секреты унаследовали бы все процессы, которые запускает агент.
Секреты (ключ GigaChat) НЕ хранятся в коде — только в окружении / .env.
"""

from __future__ import annotations

import os
import ssl
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from devassist.permissions import PermissionMode, parse_mode

# URL по умолчанию для каждой схемы авторизации.
DEFAULT_OAUTH_URL = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_MTLS_URL = "https://gigachat-ift.sberdevices.delta.sbrf.ru/v1"
DEFAULT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_MODEL = "GigaChat-3-Ultra"


class ConfigError(ValueError):
    """Некорректное значение настройки (сообщение пригодно для показа пользователю)."""


def read_dotenv(path: Path) -> dict[str, str]:
    """Минимальный парсер .env: KEY=VALUE, поддержка # комментариев и кавычек.

    Возвращает словарь; глобальное окружение процесса не изменяется.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise ConfigError(f"Не удалось прочитать {path}: {e}") from e
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_environment(
    root: Path,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> dict[str, str]:
    """Сводное окружение: environ > <root>/.env > <cwd>/.env.

    Порядок детерминирован; одинаковые пути читаются один раз.
    """
    environ = os.environ if environ is None else environ
    cwd = (cwd or Path.cwd()).resolve()
    root = root.resolve()
    merged: dict[str, str] = {}
    # От младшего источника к старшему: каждый следующий перекрывает предыдущий.
    sources = [cwd / ".env"] if cwd != root else []
    sources.append(root / ".env")
    for path in sources:
        merged.update(read_dotenv(path))
    merged.update(environ)
    return merged


def _env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    val = env.get(name)
    if val is None or not val.strip():
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as e:
        raise ConfigError(f"{name}: ожидается целое число, получено {raw!r}") from e
    if value < minimum:
        raise ConfigError(f"{name}: значение должно быть не меньше {minimum}, получено {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name}: значение должно быть не больше {maximum}, получено {value}")
    return value


def _env_optional_int(env: Mapping[str, str], name: str, *, minimum: int = 1) -> int | None:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return None
    return _env_int(env, name, 0, minimum=minimum)


def _env_mode(env: Mapping[str, str], name: str) -> PermissionMode:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return PermissionMode.MANUAL
    try:
        return parse_mode(raw)
    except ValueError as e:
        raise ConfigError(f"{name}: {e}") from e


def _env_float(env: Mapping[str, str], name: str, default: float, *, lo: float, hi: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as e:
        raise ConfigError(f"{name}: ожидается число, получено {raw!r}") from e
    if not lo <= value <= hi:
        raise ConfigError(
            f"{name}: значение должно быть в диапазоне [{lo}; {hi}], получено {value}"
        )
    return value


@dataclass(frozen=True)
class Config:
    """Сводная конфигурация приложения. Неизменяема после загрузки.

    Состояние, меняющееся во время работы (например, текущая модель после
    ``/model``), хранится в агенте, а не здесь.
    """

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
    model: str = DEFAULT_MODEL
    auth_url: str = DEFAULT_AUTH_URL
    base_url: str = DEFAULT_OAUTH_URL
    verify_ssl: bool = False
    timeout: int = 120

    # --- Агент / окружение ---
    project_root: Path = field(default_factory=Path.cwd)
    auto_approve: bool = False  # пропускать подтверждения (опасно)
    # Начальный режим разрешений (ручной / авто-правки / план); в работе режим
    # меняется в агенте (Shift+Tab, /mode).
    mode: PermissionMode = PermissionMode.MANUAL
    max_steps: int = 50  # предохранитель агентного цикла
    max_tool_failures: int = 4  # стоп при N неудачных вызовах подряд (анти-залипание)
    # Одинаковый вызов без изменений между повторами: на N-м — предупреждение
    # модели, следующий не выполняется и ход останавливается (анти-зацикливание).
    max_tool_repeats: int = 3
    temperature: float = 0.2
    stream: bool = True  # потоковый (посимвольный) вывод ответа модели
    # Бюджет контекста (оценка в токенах), в который укладывается история при
    # отправке модели; старые сообщения сверх бюджета отбрасываются. None — от
    # замеренного окна модели (~/.devassist/models.json), число — явное переопределение.
    context_budget_tokens: int | None = None
    # Сжимать историю в краткое содержание, когда она занимает compact_threshold
    # бюджета истории (бюджет за вычетом системного промпта и схем инструментов).
    auto_compact: bool = True
    compact_threshold: float = 0.8
    # Сохранять чаты в .devassist/chats/ (продолжение — /resume, --continue).
    save_chats: bool = True
    # Предлагать замер окна незамеренной модели (пробы оплачиваются); False — только
    # по --test-context.
    auto_measure: bool = True

    @classmethod
    def load(
        cls,
        *,
        project_root: Path | None = None,
        model: str | None = None,
        auto_approve: bool = False,
        mode: PermissionMode | None = None,
        stream: bool = True,
        save_chats: bool = True,
        environ: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> Config:
        """Собирает конфигурацию. Некорректные значения → :class:`ConfigError`.

        ``environ``/``cwd`` подменяются в тестах (по умолчанию — os.environ и Path.cwd()).
        """
        root = Path(project_root or cwd or Path.cwd()).resolve()
        env = load_environment(root, environ=environ, cwd=cwd)

        access_key = env.get("GIGACHAT_ACCESS_KEY") or None
        cert = env.get("GIGACHAT_CERT") or None
        key = env.get("GIGACHAT_KEY") or None

        # Выбор эндпоинта: явный GIGACHAT_URL имеет приоритет; иначе берётся
        # дефолт под выбранную схему — OAuth при наличии ключа, mTLS при cert+key.
        base_url = env.get("GIGACHAT_URL")
        if not base_url:
            base_url = DEFAULT_MTLS_URL if (not access_key and cert and key) else DEFAULT_OAUTH_URL

        return cls(
            access_key=access_key,
            cert=cert,
            key=key,
            scope=env.get("GIGACHAT_SCOPE") or "GIGACHAT_API_PERS",
            model=model or env.get("GIGACHAT_MODEL") or DEFAULT_MODEL,
            auth_url=env.get("GIGACHAT_AUTH_URL") or DEFAULT_AUTH_URL,
            base_url=base_url.rstrip("/"),
            verify_ssl=_env_bool(env, "GIGACHAT_VERIFY_SSL", False),
            timeout=_env_int(env, "GIGACHAT_TIMEOUT", 120),
            project_root=root,
            auto_approve=auto_approve,
            mode=mode or _env_mode(env, "DEVASSIST_MODE"),
            stream=stream,
            temperature=_env_float(env, "DEVASSIST_TEMPERATURE", 0.2, lo=0.0, hi=2.0),
            context_budget_tokens=_env_optional_int(env, "DEVASSIST_CONTEXT_TOKENS", minimum=4_000),
            auto_compact=_env_bool(env, "DEVASSIST_AUTO_COMPACT", True),
            compact_threshold=_env_int(
                env, "DEVASSIST_COMPACT_THRESHOLD", 80, minimum=10, maximum=95
            )
            / 100,
            save_chats=save_chats and _env_bool(env, "DEVASSIST_SAVE_CHATS", True),
            auto_measure=_env_bool(env, "DEVASSIST_AUTO_MEASURE", True),
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
