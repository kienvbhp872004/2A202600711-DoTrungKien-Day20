"""Command-line entrypoint for the lab starter."""

import datetime
import pathlib
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    client = LLMClient()
    response = client.complete(
        system_prompt="You are a research assistant. Answer the question thoroughly and clearly.",
        user_prompt=query,
    )
    state.final_answer = response.content
    console.print(Panel.fit(state.final_answer, title="Single-Agent Baseline"))
    console.print(f"[dim]tokens: {response.input_tokens} in / {response.output_tokens} out[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    print(result.model_dump_json(indent=2))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    save: Annotated[bool, typer.Option("--save", help="Save report to reports/")] = True,
) -> None:
    """Run baseline AND multi-agent on the same query, then compare."""

    _init()
    console.print(f"\n[bold]Benchmarking:[/bold] {query}\n")

    # --- Baseline runner ---
    def baseline_runner(q: str) -> ResearchState:
        client = LLMClient()
        req = ResearchQuery(query=q)
        st = ResearchState(request=req)
        resp = client.complete(
            system_prompt="You are a research assistant. Answer the question thoroughly.",
            user_prompt=q,
        )
        st.final_answer = resp.content
        # Record tokens in trace so benchmark can measure cost
        st.add_trace_event("baseline_llm", {
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
        })
        return st

    # --- Multi-agent runner ---
    def multi_agent_runner(q: str) -> ResearchState:
        st = ResearchState(request=ResearchQuery(query=q))
        return MultiAgentWorkflow().run(st)

    console.print("[cyan]Running baseline...[/cyan]")
    baseline_state, baseline_metrics = run_benchmark("baseline", query, baseline_runner)
    console.print(f"  Latency: {baseline_metrics.latency_seconds:.2f}s  |  "
                  f"Cost: ${baseline_metrics.estimated_cost_usd or 0:.6f}  |  "
                  f"{baseline_metrics.notes}")

    console.print("[cyan]Running multi-agent...[/cyan]")
    multi_state, multi_metrics = run_benchmark("multi-agent", query, multi_agent_runner)
    console.print(f"  Latency: {multi_metrics.latency_seconds:.2f}s  |  "
                  f"Cost: ${multi_metrics.estimated_cost_usd or 0:.6f}  |  "
                  f"{multi_metrics.notes}")

    # --- Report ---
    report_md = render_markdown_report(
        [baseline_metrics, multi_metrics],
        [baseline_state, multi_state],
    )
    console.print("\n")
    console.print(Markdown(report_md))

    if save:
        reports_dir = pathlib.Path("reports")
        reports_dir.mkdir(exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"benchmark_{ts}.md"
        report_path.write_text(report_md, encoding="utf-8")
        console.print(f"\n[dim]Report saved → {report_path}[/dim]")


if __name__ == "__main__":
    app()
