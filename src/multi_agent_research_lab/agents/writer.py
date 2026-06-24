"""Writer agent — synthesises a final answer with citations."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces the final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        # Build a source reference block so the writer can cite URLs
        source_refs = "\n".join(
            f"[{i+1}] {s.title} — {s.url or 'no url'}"
            for i, s in enumerate(state.sources)
        )

        with trace_span("writer", {"audience": state.request.audience}) as span:
            resp = self._llm.complete(
                system_prompt=(
                    f"You are a technical writer. Your audience: {state.request.audience}.\n"
                    "Write a clear, well-structured answer to the query. "
                    "Cite sources using [1], [2] notation. "
                    "End with a 'Sources' section listing the references."
                ),
                user_prompt=(
                    f"Query: {state.request.query}\n\n"
                    f"Analysis:\n{state.analysis_notes}\n\n"
                    f"Available sources:\n{source_refs}"
                ),
            )
            state.final_answer = resp.content
            span["input_tokens"] = resp.input_tokens
            span["output_tokens"] = resp.output_tokens

        state.add_trace_event("writer_done", {
            "answer_length": len(state.final_answer),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
        })
        return state
