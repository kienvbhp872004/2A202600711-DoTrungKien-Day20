"""Tracing hooks — wired to LangSmith when key is present, falls back to local spans."""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings


def _langsmith_enabled() -> bool:
    settings = get_settings()
    return bool(settings.langsmith_api_key)


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Span context — sends to LangSmith if LANGSMITH_API_KEY is set."""

    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}

    if _langsmith_enabled():
        try:
            from langsmith import trace as ls_trace

            with ls_trace(
                name=name,
                metadata=attributes or {},
                project_name=get_settings().langsmith_project,
            ):
                started = perf_counter()
                try:
                    yield span
                finally:
                    span["duration_seconds"] = perf_counter() - started
            return
        except Exception:
            pass  # fallback to local span if LangSmith fails

    started = perf_counter()
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started
