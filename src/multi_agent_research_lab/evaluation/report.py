"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


def render_markdown_report(
    metrics: list[BenchmarkMetrics],
    states: list[ResearchState] | None = None,
) -> str:
    lines: list[str] = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.6f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )

    # Summary analysis when both runs present
    if len(metrics) == 2:
        a, b = metrics[0], metrics[1]
        lines += ["", "## Summary", ""]
        lat_diff = b.latency_seconds - a.latency_seconds
        lines.append(
            f"- **Latency delta**: {lat_diff:+.2f}s "
            f"({'slower' if lat_diff > 0 else 'faster'} for {b.run_name})"
        )
        if a.estimated_cost_usd is not None and b.estimated_cost_usd is not None:
            cost_diff = b.estimated_cost_usd - a.estimated_cost_usd
            lines.append(f"- **Cost delta**: ${cost_diff:+.6f} ({b.run_name} vs {a.run_name})")

    if states:
        lines += ["", "## Trace Summary", ""]
        for state, metric in zip(states, metrics):
            lines.append(f"### {metric.run_name}")
            lines.append(f"- Route: `{' → '.join(state.route_history)}`")
            lines.append(f"- Sources: {len(state.sources)}")
            lines.append(f"- Errors: {len(state.errors)}")
            if state.errors:
                for err in state.errors:
                    lines.append(f"  - {err}")
            lines.append("")

    return "\n".join(lines) + "\n"
