"""Benchmark: single-agent vs multi-agent comparison."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]

# Cost per token in USD (openrouter gpt-4o-mini pricing)
_COST_PER_INPUT_TOKEN = 0.00000015   # $0.15 / 1M tokens
_COST_PER_OUTPUT_TOKEN = 0.0000006   # $0.60 / 1M tokens


def _estimate_cost(state: ResearchState) -> float | None:
    """Sum token costs recorded in trace spans."""
    total = 0.0
    found = False
    for event in state.trace:
        payload = event.get("payload", {})
        inp = payload.get("input_tokens")
        out = payload.get("output_tokens")
        if inp is not None:
            total += inp * _COST_PER_INPUT_TOKEN
            found = True
        if out is not None:
            total += out * _COST_PER_OUTPUT_TOKEN
            found = True
    return round(total, 6) if found else None


def _citation_coverage(state: ResearchState) -> float | None:
    """Fraction of sources that appear as citations in the final answer."""
    if not state.sources or not state.final_answer:
        return None
    cited = sum(
        1 for i in range(len(state.sources))
        if f"[{i + 1}]" in state.final_answer
    )
    return round(cited / len(state.sources), 2)


def _count_llm_calls(state: ResearchState) -> int:
    """Number of LLM calls recorded via trace events."""
    return sum(
        1 for e in state.trace
        if e.get("payload", {}).get("input_tokens") is not None
    )


def run_benchmark(
    run_name: str,
    query: str,
    runner: Runner,
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run a single benchmark, measure latency and derived metrics."""

    started = perf_counter()
    failed = False
    try:
        state = runner(query)
    except Exception as exc:
        # Capture failure without crashing the benchmark loop
        from multi_agent_research_lab.core.schemas import ResearchQuery
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(exc))
        failed = True
    latency = perf_counter() - started

    notes_parts: list[str] = []
    if failed:
        notes_parts.append("FAILED")
    if state.errors:
        notes_parts.append(f"errors={len(state.errors)}")

    llm_calls = _count_llm_calls(state)
    if llm_calls:
        notes_parts.append(f"llm_calls={llm_calls}")

    citation_cov = _citation_coverage(state)
    if citation_cov is not None:
        notes_parts.append(f"citation_cov={citation_cov:.0%}")

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=_estimate_cost(state),
        notes=", ".join(notes_parts),
    )
    return state, metrics
