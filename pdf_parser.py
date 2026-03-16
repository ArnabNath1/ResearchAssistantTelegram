"""
PDF Parser — Extracts structured text from research paper PDFs using PyMuPDF.
Detects sections (Abstract, Introduction, Methodology, Results, etc.)
and extracts equations, tables, and figures metadata.
"""

import fitz  # PyMuPDF
import re
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional


# Common section headers in research papers
SECTION_PATTERNS = [
    (r'\babstract\b', 'abstract'),
    (r'\bintroduction\b', 'introduction'),
    (r'\brelated work\b', 'related_work'),
    (r'\bliterature review\b', 'related_work'),
    (r'\bbackground\b', 'background'),
    (r'\bmethodology\b', 'methodology'),
    (r'\bmethod\b', 'methodology'),
    (r'\bapproach\b', 'methodology'),
    (r'\bproposed\b', 'methodology'),
    (r'\barchitecture\b', 'methodology'),
    (r'\bexperiment', 'experiments'),
    (r'\bevaluation\b', 'experiments'),
    (r'\bresults?\b', 'results'),
    (r'\bdiscussion\b', 'discussion'),
    (r'\bconclusion', 'conclusion'),
    (r'\breferences?\b', 'references'),
    (r'\bappendix\b', 'appendix'),
    (r'\blimitation', 'limitations'),
    (r'\bfuture work\b', 'future_work'),
    (r'\bdataset', 'datasets'),
]


@dataclass
class ParsedPaper:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    full_text: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    equations: list[str] = field(default_factory=list)
    num_pages: int = 0
    num_figures: int = 0
    num_tables: int = 0
    raw_metadata: dict = field(default_factory=dict)


class PDFParser:
    """
    Robust PDF parser for academic papers.
    Extracts title, authors, abstract, sections, and equations.
    """

    def parse(self, pdf_bytes: bytes) -> ParsedPaper:
        """Parse a PDF from bytes and return a structured ParsedPaper."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            raise ValueError(f"Invalid PDF: {e}")

        result = ParsedPaper()
        result.num_pages = len(doc)
        result.raw_metadata = doc.metadata or {}

        all_text_blocks = []
        all_text = []

        for page_num, page in enumerate(doc):
            blocks = page.get_text("blocks")
            for block in blocks:
                # block = (x0, y0, x1, y1, text, block_no, block_type)
                if len(block) >= 5 and block[6] == 0:  # text block
                    text = block[4].strip()
                    if text:
                        all_text_blocks.append({
                            "page": page_num + 1,
                            "text": text,
                            "y": block[1],
                            "size": self._estimate_font_size(block),
                        })
                        all_text.append(text)

        full_text = "\n".join(all_text)
        result.full_text = self._clean_text(full_text)

        # Extract metadata
        result.title = self._extract_title(all_text_blocks, doc.metadata)
        result.authors = self._extract_authors(all_text_blocks)
        result.abstract = self._extract_abstract(full_text)
        result.sections = self._extract_sections(full_text)
        result.equations = self._extract_equations(full_text)

        # Count figures/tables (approximate)
        result.num_figures = len(re.findall(r'\bfig(?:ure)?\.?\s*\d+', full_text, re.IGNORECASE))
        result.num_tables = len(re.findall(r'\btable\s*\d+', full_text, re.IGNORECASE))

        doc.close()
        logger.info(
            f"Parsed PDF: '{result.title}' | pages={result.num_pages} | "
            f"sections={list(result.sections.keys())} | equations={len(result.equations)}"
        )
        return result

    def _estimate_font_size(self, block) -> float:
        """Estimate relative font size from block bounding box height."""
        try:
            return block[3] - block[1]
        except Exception:
            return 0.0

    def _clean_text(self, text: str) -> str:
        """Remove excessive whitespace and non-printable chars."""
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {3,}', ' ', text)
        return text.strip()

    def _extract_title(self, blocks: list[dict], metadata: dict) -> str:
        """Extract paper title (usually largest font on page 1)."""
        # Try metadata first
        if metadata.get("title"):
            return metadata["title"].strip()

        # Fallback: largest text block on first page
        page1_blocks = [b for b in blocks if b["page"] == 1 and len(b["text"]) > 10]
        if page1_blocks:
            largest = max(page1_blocks[:10], key=lambda b: b["size"])
            candidate = largest["text"].replace("\n", " ").strip()
            if 10 < len(candidate) < 300:
                return candidate

        return "Unknown Title"

    def _extract_authors(self, blocks: list[dict]) -> list[str]:
        """Heuristic extraction of author names from page 1."""
        page1_text = " ".join(b["text"] for b in blocks if b["page"] == 1)
        # Look for author-like patterns (comma-separated names)
        patterns = [
            r'(?:by\s+)([\w\s,\.]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',
        ]
        for pat in patterns:
            match = re.search(pat, page1_text[:1000])
            if match:
                raw = match.group(1)
                authors = [a.strip() for a in re.split(r',|and', raw) if len(a.strip()) > 3]
                if authors:
                    return authors[:10]
        return []

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section."""
        patterns = [
            r'Abstract[.\-—]?\s*\n+([\s\S]{100,1500?}?)(?=\n\n|\n[A-Z]|\n\d+\.)',
            r'ABSTRACT\s*\n+([\s\S]{100,1500?}?)(?=\n\n|\n[A-Z])',
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return self._clean_text(match.group(1))

        # Fallback: first 1000 chars if nothing found
        return text[:1000].strip()

    def _extract_sections(self, text: str) -> dict[str, str]:
        """
        Split paper into named sections based on common header patterns.
        Returns dict of section_name -> section_text.
        """
        sections = {}
        lines = text.split('\n')
        current_section = "preamble"
        current_content = []

        for line in lines:
            stripped = line.strip()
            detected = self._detect_section_header(stripped)
            if detected:
                if current_content:
                    content = "\n".join(current_content).strip()
                    if content:
                        if current_section in sections:
                            sections[current_section] += "\n" + content
                        else:
                            sections[current_section] = content
                current_section = detected
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content:
            content = "\n".join(current_content).strip()
            if content:
                if current_section in sections:
                    sections[current_section] += "\n" + content
                else:
                    sections[current_section] = content

        return sections

    def _detect_section_header(self, line: str) -> Optional[str]:
        """Check if a line is a recognized section header."""
        if not line or len(line) > 80:
            return None
        # Must start with number, upper case, or keyword
        if not (line[0].isupper() or line[0].isdigit()):
            return None
        line_lower = line.lower()
        for pattern, section_name in SECTION_PATTERNS:
            if re.search(pattern, line_lower):
                return section_name
        return None

    def _extract_equations(self, text: str) -> list[str]:
        """
        Extract mathematical equations from text.
        Looks for LaTeX-like patterns and standalone formulas.
        """
        equations = []
        # LaTeX inline and display math
        latex_patterns = [
            r'\$\$[\s\S]+?\$\$',
            r'\$[^\$\n]+?\$',
            r'\\begin\{equation\}[\s\S]+?\\end\{equation\}',
            r'\\begin\{align\}[\s\S]+?\\end\{align\}',
        ]
        for pat in latex_patterns:
            found = re.findall(pat, text)
            equations.extend(found)

        # Simple formula-like lines (e.g., "y = mx + b")
        formula_pattern = r'\b[a-zA-Z]\s*=\s*[a-zA-Z0-9\+\-\*\/\^\(\)\s]{5,50}'
        found = re.findall(formula_pattern, text)
        equations.extend(found[:20])

        return list(set(equations))[:50]  # deduplicate, cap at 50


# Singleton
pdf_parser = PDFParser()
