"""
Research API Client — Fetches papers from arXiv and Semantic Scholar.
Used for literature retrieval, paper recommendations, and related work discovery.
"""

import httpx
import asyncio
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional
import xml.etree.ElementTree as ET
import re


@dataclass
class PaperResult:
    title: str
    authors: list[str]
    abstract: str
    url: str
    source: str  # 'arxiv' or 'semantic_scholar'
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None
    citation_count: Optional[int] = None
    fields: list[str] = field(default_factory=list)


ARXIV_API = "https://export.arxiv.org/api/query"
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

# XML namespaces for arXiv
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class ResearchAPIClient:
    """Async client for arXiv and Semantic Scholar APIs."""

    def __init__(self):
        self.http = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

    async def search_arxiv(self, query: str, max_results: int = 5) -> list[PaperResult]:
        """Search arXiv for papers matching query."""
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            response = await self.http.get(ARXIV_API, params=params)
            response.raise_for_status()
            return self._parse_arxiv_response(response.text)
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

    def _parse_arxiv_response(self, xml_text: str) -> list[PaperResult]:
        """Parse arXiv Atom feed XML into PaperResult list."""
        results = []
        try:
            root = ET.fromstring(xml_text)
            entries = root.findall("atom:entry", NS)
            for entry in entries:
                title_el = entry.find("atom:title", NS)
                summary_el = entry.find("atom:summary", NS)
                id_el = entry.find("atom:id", NS)
                authors_els = entry.findall("atom:author", NS)

                title = title_el.text.strip().replace("\n", " ") if title_el is not None else "Unknown"
                abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""
                arxiv_url = id_el.text.strip() if id_el is not None else ""
                arxiv_id = arxiv_url.split("/")[-1] if arxiv_url else None
                authors = []
                for a in authors_els:
                    name_el = a.find("atom:name", NS)
                    if name_el is not None:
                        authors.append(name_el.text.strip())

                # Try to get year from published date
                published_el = entry.find("atom:published", NS)
                year = None
                if published_el is not None and published_el.text:
                    year_match = re.search(r'(\d{4})', published_el.text)
                    if year_match:
                        year = int(year_match.group(1))

                results.append(PaperResult(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=arxiv_url,
                    source="arxiv",
                    arxiv_id=arxiv_id,
                    year=year,
                ))
        except Exception as e:
            logger.error(f"arXiv XML parse error: {e}")
        return results

    async def search_semantic_scholar(self, query: str, max_results: int = 5) -> list[PaperResult]:
        """Search Semantic Scholar for papers."""
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,citationCount,externalIds,fieldsOfStudy,url",
        }
        try:
            response = await self.http.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/search",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
            return self._parse_ss_response(data)
        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")
            return []

    def _parse_ss_response(self, data: dict) -> list[PaperResult]:
        """Parse Semantic Scholar API response."""
        results = []
        papers = data.get("data", [])
        for paper in papers:
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            external_ids = paper.get("externalIds", {}) or {}
            doi = external_ids.get("DOI")
            arxiv_id = external_ids.get("ArXiv")
            url = paper.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")
            results.append(PaperResult(
                title=paper.get("title", "Unknown"),
                authors=authors,
                abstract=paper.get("abstract") or "",
                url=url,
                source="semantic_scholar",
                doi=doi,
                arxiv_id=arxiv_id,
                year=paper.get("year"),
                citation_count=paper.get("citationCount"),
                fields=paper.get("fieldsOfStudy") or [],
            ))
        return results

    async def get_arxiv_paper(self, arxiv_id: str) -> Optional[PaperResult]:
        """Fetch a specific arXiv paper by ID."""
        results = await self.search_arxiv(f"id:{arxiv_id}", max_results=1)
        return results[0] if results else None

    async def fetch_arxiv_pdf(self, arxiv_id: str) -> Optional[bytes]:
        """Download PDF bytes for an arXiv paper."""
        # Clean ID
        arxiv_id = arxiv_id.strip().replace("https://arxiv.org/abs/", "")
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            response = await self.http.get(pdf_url)
            response.raise_for_status()
            logger.info(f"Downloaded arXiv PDF: {arxiv_id} ({len(response.content)} bytes)")
            return response.content
        except Exception as e:
            logger.error(f"Failed to download arXiv PDF {arxiv_id}: {e}")
            return None

    async def combined_search(self, query: str, max_per_source: int = 3) -> list[PaperResult]:
        """
        Search both arXiv and Semantic Scholar in parallel,
        merge and deduplicate results.
        """
        arxiv_results, ss_results = await asyncio.gather(
            self.search_arxiv(query, max_per_source),
            self.search_semantic_scholar(query, max_per_source),
        )
        seen_titles = set()
        combined = []
        for paper in arxiv_results + ss_results:
            norm_title = paper.title.lower().strip()
            if norm_title not in seen_titles:
                seen_titles.add(norm_title)
                combined.append(paper)
        return combined

    async def close(self):
        await self.http.aclose()


# Singleton
research_api = ResearchAPIClient()
