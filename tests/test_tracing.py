"""Тесты gate'а Phoenix-трейсинга — без тяжёлых зависимостей и сети."""

import app.observability.tracing as tracing
from app.core.config import Settings


def test_tracing_disabled_returns_false() -> None:
    assert tracing.setup_tracing(Settings(phoenix_enabled=False)) is False


def test_tracing_enabled_without_deps_returns_false(monkeypatch) -> None:
    """Флаг включён, но пакеты трейсинга не установлены — setup_tracing
    не должен падать, только предупредить и вернуть False."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        # Симулируем ImportError для пакетов opentelemetry/openinference
        if name.startswith(("opentelemetry", "openinference")):
            raise ImportError(f"mocked ImportError for {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    settings = Settings(phoenix_enabled=True)
    result = tracing.setup_tracing(settings)
    assert result is False