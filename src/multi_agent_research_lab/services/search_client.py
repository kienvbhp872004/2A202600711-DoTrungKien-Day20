"""Search client — currently backed by Tavily."""

from tavily import TavilyClient

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client backed by Tavily."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = TavilyClient(api_key=settings.tavily_api_key)
        self._depth = settings.tavily_search_depth

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search the web and return structured source documents."""

        response = self._client.search(
            query=query,
            search_depth=self._depth,
            max_results=max_results,
        )
        docs: list[SourceDocument] = []
        for r in response.get("results", []):
            docs.append(
                SourceDocument(
                    title=r.get("title", ""),
                    url=r.get("url"),
                    snippet=r.get("content", ""),
                )
            )
        return docs
