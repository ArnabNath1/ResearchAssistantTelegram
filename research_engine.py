"""
Research Engine — Core scientific reasoning pipeline.
Orchestrates PDF parsing, vector store, LLM inference, and API retrieval.
"""

from loguru import logger
from pdf_parser import pdf_parser, ParsedPaper
from vector_store import vector_store
from llm_client import groq_client
from research_api import research_api, PaperResult
from prompts import (
    MASTER_SYSTEM_PROMPT,
    PAPER_ANALYSIS_PROMPT,
    RESEARCH_QA_PROMPT,
    GAP_DETECTION_PROMPT,
    EXPERIMENT_SUGGESTION_PROMPT,
    NOVELTY_GENERATION_PROMPT,
    LITERATURE_REVIEW_PROMPT,
    EQUATION_EXPLANATION_PROMPT,
    DATASET_RECOMMENDATION_PROMPT,
    SHORT_SUMMARY_PROMPT,
)
import hashlib
import asyncio
from typing import Optional


def _paper_id(title: str, content: str = "") -> str:
    """Generate stable paper ID."""
    raw = (title + content[:100]).encode()
    return hashlib.md5(raw).hexdigest()[:16]


def _format_paper_for_llm(paper: ParsedPaper) -> str:
    """Format parsed paper into structured text for LLM input."""
    parts = [f"TITLE: {paper.title}"]

    if paper.authors:
        parts.append(f"AUTHORS: {', '.join(paper.authors[:5])}")

    parts.append(f"PAGES: {paper.num_pages}")

    if paper.abstract:
        parts.append(f"\nABSTRACT:\n{paper.abstract}")

    # Add key sections
    section_priority = [
        "introduction", "methodology", "experiments",
        "results", "discussion", "conclusion", "limitations"
    ]
    for sec in section_priority:
        if sec in paper.sections:
            content = paper.sections[sec][:2000]  # Cap section length
            parts.append(f"\n{sec.upper()}:\n{content}")

    # Add equations if any
    if paper.equations:
        eq_sample = paper.equations[:10]
        parts.append(f"\nKEY EQUATIONS:\n" + "\n".join(eq_sample))

    return "\n\n".join(parts)


class ResearchEngine:
    """
    Main orchestration engine for all research intelligence tasks.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # PAPER ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    async def analyze_paper_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Full pipeline: PDF → parse → store → analyze → return insights.
        """
        # 1. Parse PDF
        logger.info("Parsing PDF...")
        paper = pdf_parser.parse(pdf_bytes)

        # 2. Store in vector DB (async, non-blocking for response time)
        paper_id = _paper_id(paper.title, paper.abstract)
        asyncio.create_task(
            vector_store.upsert_paper(
                paper_id=paper_id,
                title=paper.title,
                abstract=paper.abstract,
                full_text=paper.full_text,
                metadata={"authors": paper.authors, "pages": paper.num_pages},
            )
        )

        # 3. Format paper for LLM
        paper_text = _format_paper_for_llm(paper)

        # 4. Get LLM analysis
        logger.info(f"Analyzing paper: {paper.title}")
        result = await groq_client.complete(
            system_prompt=MASTER_SYSTEM_PROMPT + "\n\n" + PAPER_ANALYSIS_PROMPT,
            user_message=f"Analyze this research paper:\n\n{paper_text}",
            temperature=0.2,
            max_tokens=4096,
        )
        return result

    async def quick_summary(self, pdf_bytes: bytes) -> str:
        """Quick TL;DR summary of a paper."""
        paper = pdf_parser.parse(pdf_bytes)
        paper_text = _format_paper_for_llm(paper)
        result = await groq_client.complete(
            system_prompt=MASTER_SYSTEM_PROMPT + "\n\n" + SHORT_SUMMARY_PROMPT,
            user_message=f"Summarize:\n\n{paper_text}",
            temperature=0.2,
            max_tokens=1024,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # RESEARCH Q&A WITH RAG
    # ─────────────────────────────────────────────────────────────────────────

    async def answer_research_question(
        self,
        question: str,
        paper_id: Optional[str] = None,
        chat_history: Optional[list] = None,
    ) -> str:
        """
        Answer a research question using RAG (vector store + LLM).
        """
        # Retrieve relevant context from vector store
        context = await vector_store.get_paper_context(question, paper_id=paper_id, top_k=5)
        system = MASTER_SYSTEM_PROMPT + "\n\n" + RESEARCH_QA_PROMPT.format(
            context=context if context else "No prior knowledge available. Use your scientific expertise."
        )

        if chat_history:
            messages = chat_history + [{"role": "user", "content": question}]
            result = await groq_client.multi_turn(
                messages=messages,
                system_prompt=system,
                temperature=0.3,
                max_tokens=2048,
            )
        else:
            result = await groq_client.complete(
                system_prompt=system,
                user_message=question,
                temperature=0.3,
                max_tokens=2048,
            )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # GAP DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    async def detect_research_gaps(self, pdf_bytes: bytes) -> str:
        """Detect research gaps in a paper."""
        paper = pdf_parser.parse(pdf_bytes)
        paper_text = _format_paper_for_llm(paper)

        # Also pull related work context from vector store
        context = await vector_store.get_paper_context(paper.title, top_k=3)
        context_note = f"\nRELATED KNOWLEDGE:\n{context}" if context else ""

        result = await groq_client.complete(
            system_prompt=MASTER_SYSTEM_PROMPT + "\n\n" + GAP_DETECTION_PROMPT,
            user_message=f"{paper_text}{context_note}",
            temperature=0.3,
            max_tokens=3000,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # EXPERIMENT SUGGESTIONS
    # ─────────────────────────────────────────────────────────────────────────

    async def suggest_experiments(self, pdf_bytes: bytes) -> str:
        """Generate concrete experiment suggestions for a paper."""
        paper = pdf_parser.parse(pdf_bytes)
        paper_text = _format_paper_for_llm(paper)
        result = await groq_client.complete(
            system_prompt=MASTER_SYSTEM_PROMPT + "\n\n" + EXPERIMENT_SUGGESTION_PROMPT,
            user_message=f"Design experiments for:\n\n{paper_text}",
            temperature=0.35,
            max_tokens=3000,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # NOVELTY GENERATION
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_novel_ideas(self, pdf_bytes: bytes) -> str:
        """Generate novel research ideas from a paper."""
        paper = pdf_parser.parse(pdf_bytes)
        paper_text = _format_paper_for_llm(paper)
        result = await groq_client.complete(
            system_prompt=MASTER_SYSTEM_PROMPT + "\n\n" + NOVELTY_GENERATION_PROMPT,
            user_message=f"Generate novel research ideas from:\n\n{paper_text}",
            temperature=0.5,  # More creative
            max_tokens=3500,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # LITERATURE REVIEW
    # ─────────────────────────────────────────────────────────────────────────

    async def generate_literature_review(self, topic: str) -> str:
        """Generate a literature review for a given research topic."""
        # Fetch papers from APIs
        papers = await research_api.combined_search(topic, max_per_source=5)

        # Retrieve from vector store
        context = await vector_store.get_paper_context(topic, top_k=5)

        # Build paper summaries
        paper_summaries = []
        for p in papers[:8]:
            summary = f"• [{p.source.upper()}] {p.title} ({p.year or 'n/d'})"
            if p.authors:
                summary += f" — {', '.join(p.authors[:2])}"
            if p.abstract:
                summary += f"\n  Abstract: {p.abstract[:300]}"
            paper_summaries.append(summary)

        papers_text = "\n\n".join(paper_summaries)
        context_text = f"\nSTORED KNOWLEDGE:\n{context}" if context else ""

        result = await groq_client.complete(
            system_prompt=MASTER_SYSTEM_PROMPT + "\n\n" + LITERATURE_REVIEW_PROMPT.format(topic=topic),
            user_message=f"Topic: {topic}\n\nRetrieved Papers:\n{papers_text}{context_text}",
            temperature=0.3,
            max_tokens=3500,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # EQUATION EXPLANATION
    # ─────────────────────────────────────────────────────────────────────────

    async def explain_equations(self, equation_text: str, paper_context: str = "") -> str:
        """Explain mathematical equations in depth."""
        user_msg = f"Explain these equations:\n{equation_text}"
        if paper_context:
            user_msg += f"\n\nPaper context:\n{paper_context[:1000]}"
        result = await groq_client.complete(
            system_prompt=MASTER_SYSTEM_PROMPT + "\n\n" + EQUATION_EXPLANATION_PROMPT,
            user_message=user_msg,
            temperature=0.2,
            max_tokens=2048,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # DATASET RECOMMENDATION
    # ─────────────────────────────────────────────────────────────────────────

    async def recommend_datasets(self, research_task: str) -> str:
        """Recommend datasets for a research task."""
        result = await groq_client.complete(
            system_prompt=MASTER_SYSTEM_PROMPT + "\n\n" + DATASET_RECOMMENDATION_PROMPT,
            user_message=f"Research task: {research_task}",
            temperature=0.25,
            max_tokens=2048,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # ARXIV PAPER ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    async def analyze_arxiv_paper(self, arxiv_id: str) -> tuple[str, Optional[bytes]]:
        """Download and analyze a paper by arXiv ID. Returns (analysis, pdf_bytes)."""
        # Clean arXiv ID
        arxiv_id = arxiv_id.strip()
        if "arxiv.org" in arxiv_id:
            arxiv_id = arxiv_id.split("/")[-1]

        logger.info(f"Fetching arXiv paper: {arxiv_id}")
        pdf_bytes = await research_api.fetch_arxiv_pdf(arxiv_id)
        if not pdf_bytes:
            return f"❌ Could not download arXiv paper `{arxiv_id}`. Check the ID and try again.", None

        analysis = await self.analyze_paper_from_pdf(pdf_bytes)
        return analysis, pdf_bytes

    # ─────────────────────────────────────────────────────────────────────────
    # PAPER SEARCH
    # ─────────────────────────────────────────────────────────────────────────

    async def search_papers(self, query: str) -> str:
        """Search for papers and return formatted results."""
        papers = await research_api.combined_search(query, max_per_source=4)
        if not papers:
            return "❌ No papers found. Try a different query."

        lines = [f"🔍 **Papers found for:** `{query}`\n"]
        for i, p in enumerate(papers[:8], 1):
            lines.append(f"**{i}. {p.title}**")
            if p.authors:
                lines.append(f"   👤 {', '.join(p.authors[:3])}")
            if p.year:
                lines.append(f"   📅 {p.year}")
            if p.citation_count is not None:
                lines.append(f"   📊 {p.citation_count} citations")
            lines.append(f"   🔗 {p.url}")
            if p.abstract:
                lines.append(f"   > {p.abstract[:200]}...")
            lines.append("")

        return "\n".join(lines)


# Singleton
research_engine = ResearchEngine()
