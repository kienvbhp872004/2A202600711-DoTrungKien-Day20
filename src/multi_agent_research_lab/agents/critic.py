"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"
    def __init__(self) -> None:
        self._llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        with trace_span("critic", {"notes_length": len(state.final_answer or "")}) as span:
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
                    f"Final answer:\n{state.final_answer}"
                ),
            )
            state.fact_check = resp.content
            span["input_tokens"] = resp.input_tokens
            span["output_tokens"] = resp.output_tokens

        state.add_trace_event("critic_done", {
            "fact_check_length": len(state.fact_check),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
        })
        return state