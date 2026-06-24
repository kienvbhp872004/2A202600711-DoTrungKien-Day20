"""Researcher agent — searches the web and summarises sources into notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self._search = SearchClient()
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        try:
            with trace_span("researcher", {"query": state.request.query}) as span:
                sources = self._search.search(
                    query=state.request.query,
                    max_results=state.request.max_sources,
                )
                state.sources = sources
                span["num_sources"] = len(sources)

                snippets = "\n\n".join(
                    f"[{i+1}] {s.title}\n{s.snippet}" for i, s in enumerate(sources)
                )
                resp = self._llm.complete(
                    system_prompt=(
                        "You are a research assistant. Summarise the following search results "
                        "into clear, concise research notes. Keep all important facts and "
                        "mention the source number (e.g. [1]) for each key claim."
                    ),
                    user_prompt=f"Query: {state.request.query}\n\nSearch results:\n{snippets}",
                )
                state.research_notes = resp.content
                span["input_tokens"] = resp.input_tokens
                span["output_tokens"] = resp.output_tokens

            state.add_trace_event("researcher_done", {
                "num_sources": len(sources),
                "notes_length": len(state.research_notes),
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
            })
        except Exception as exc:
            state.errors.append(f"researcher failed: {exc}")
            state.add_trace_event("researcher_error", {"error": str(exc)})
        return state
