"""Analyst agent — extracts key claims and flags weak evidence."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("analyst", {"notes_length": len(state.research_notes or "")}) as span:
            resp = self._llm.complete(
                system_prompt=(
                    "You are a critical analyst. Given research notes, produce a structured analysis:\n"
                    "1. Key claims (bullet list)\n"
                    "2. Conflicting viewpoints (if any)\n"
                    "3. Evidence quality: flag any claims that lack strong sources\n"
                    "4. Gaps: what is still unknown or needs more research?\n"
                    "Be concise and objective."
                ),
                user_prompt=(
                    f"Query: {state.request.query}\n\n"
                    f"Research notes:\n{state.research_notes}"
                ),
            )
            state.analysis_notes = resp.content
            span["input_tokens"] = resp.input_tokens
            span["output_tokens"] = resp.output_tokens

        state.add_trace_event("analyst_done", {
            "analysis_length": len(state.analysis_notes),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
        })
        return state
