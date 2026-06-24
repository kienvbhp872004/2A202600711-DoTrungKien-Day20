"""LangGraph multi-agent workflow."""

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState

# LangGraph nhận/trả dict, nên cần 2 hàm chuyển đổi
def _to_dict(state: ResearchState) -> dict:
    return state.model_dump()


def _from_dict(data: dict) -> ResearchState:
    return ResearchState.model_validate(data)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Graph shape:
        supervisor → researcher → supervisor
        supervisor → analyst    → supervisor
        supervisor → writer     → supervisor
        supervisor → END
    """

    def build(self) -> object:
        supervisor = SupervisorAgent()
        researcher = ResearcherAgent()
        analyst = AnalystAgent()
        writer = WriterAgent()
        critic = CriticAgent()
        def run_supervisor(data: dict) -> dict:
            return _to_dict(supervisor.run(_from_dict(data)))

        def run_researcher(data: dict) -> dict:
            return _to_dict(researcher.run(_from_dict(data)))

        def run_analyst(data: dict) -> dict:
            return _to_dict(analyst.run(_from_dict(data)))

        def run_writer(data: dict) -> dict:
            return _to_dict(writer.run(_from_dict(data)))
        def run_critic(data: dict) -> dict:
            return _to_dict(critic.run(_from_dict(data)))
        def route(data: dict) -> str:
            """Đọc route mới nhất từ route_history."""
            history = data.get("route_history", [])
            return history[-1] if history else "done"

        graph = StateGraph(dict)
        graph.add_node("supervisor", run_supervisor)
        graph.add_node("researcher", run_researcher)
        graph.add_node("analyst", run_analyst)
        graph.add_node("writer", run_writer)
        graph.add_node("fact_checker", run_critic)
        graph.set_entry_point("supervisor")

        # Conditional routing từ supervisor
        graph.add_conditional_edges(
            "supervisor",
            route,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "fact_checker": "fact_checker",
                "done": END,
            },
        )

        # Mỗi worker xong → quay về supervisor
        graph.add_edge("researcher", "supervisor")
        graph.add_edge("analyst", "supervisor")
        graph.add_edge("writer", "supervisor")
        graph.add_edge("fact_checker", "supervisor")
        return graph.compile()

    def run(self, state: ResearchState) -> ResearchState:
        graph = self.build()
        result_dict = graph.invoke(_to_dict(state))
        return _from_dict(result_dict)
