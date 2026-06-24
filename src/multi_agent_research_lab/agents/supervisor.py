"""Supervisor / router — decides which worker runs next."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop.

    Routing policy (rule-based — no LLM call needed here):
      researcher  → khi chưa có research_notes
      analyst     → khi đã có research_notes nhưng chưa có analysis_notes
      writer      → khi đã có analysis_notes nhưng chưa có final_answer
      done        → khi đã có final_answer, hoặc vượt max_iterations
    """

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        settings = get_settings()

        if state.iteration >= settings.max_iterations:
            # Vượt giới hạn → fallback: ghi nhận lỗi và dừng
            state.errors.append(f"Reached max iterations ({settings.max_iterations})")
            state.record_route("done")
            state.add_trace_event("supervisor", {"route": "done", "reason": "max_iterations"})
            return state

        if state.errors:
            route = "done"
        elif state.research_notes is None:
            route = "researcher"
        elif state.analysis_notes is None:
            route = "analyst"
        elif state.final_answer is None:
            route = "writer"
        elif state.fact_check is None:
            route = "fact_checker"
        else:
            route = "done"

        state.record_route(route)
        state.add_trace_event("supervisor", {"route": route, "iteration": state.iteration})
        return state
