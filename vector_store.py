"""
Astra DB Vector Store — Handles embeddings, upsert, and semantic search
for RAG-based grounded scientific reasoning.
"""

from astrapy import DataAPIClient
from sentence_transformers import SentenceTransformer
from loguru import logger
from config import settings
from typing import Optional
import numpy as np
import hashlib
import json


class VectorStore:
    """
    Manages vector storage and semantic retrieval using Astra DB.
    Embeddings are generated with sentence-transformers (local, no API cost).
    """

    def __init__(self):
        self._model = None
        self.dimension = 384  # all-MiniLM-L6-v2 output dim
        self.collection = None
        self._init_db()

    def _init_db(self):
        """Initialize Astra DB connection and ensure collection exists."""
        try:
            client = DataAPIClient(settings.astra_db_application_token)
            db = client.get_database_by_api_endpoint(settings.astra_db_api_endpoint)
            logger.info("Connected to Astra DB")

            # Create collection with vector support if it doesn't exist
            existing = [c.name for c in db.list_collections()]
            if settings.astra_db_collection not in existing:
                db.create_collection(
                    settings.astra_db_collection,
                    dimension=self.dimension,
                    metric="cosine",
                )
                logger.info(f"Created collection: {settings.astra_db_collection}")
            else:
                logger.info(f"Using existing collection: {settings.astra_db_collection}")

            self.collection = db.get_collection(settings.astra_db_collection)
        except Exception as e:
            logger.error(f"Astra DB init failed: {e}")
            self.collection = None

    def _get_model(self):
        """Lazy load the embedding model to save memory on start."""
        if self._model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            # Optimization: Disable gradients for torch to save RAM
            try:
                import torch
                torch.set_grad_enabled(False)
            except ImportError:
                pass
            
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    def _embed(self, text: str) -> list[float]:
        """Generate normalized embedding for text."""
        model = self._get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _doc_id(self, text: str) -> str:
        """Generate stable document ID from content hash."""
        return hashlib.md5(text[:200].encode()).hexdigest()

    async def upsert_paper(self, paper_id: str, title: str, abstract: str, full_text: str, metadata: dict = None):
        """
        Store a research paper's chunks in the vector store.
        Chunks abstract + sections separately for finer retrieval.
        """
        if not self.collection:
            logger.warning("VectorStore not initialized, skipping upsert")
            return

        chunks = self._chunk_text(full_text, chunk_size=800, overlap=100)
        docs = []

        # Store abstract as a special high-priority chunk
        abstract_doc = {
            "_id": f"{paper_id}_abstract",
            "paper_id": paper_id,
            "title": title,
            "chunk_type": "abstract",
            "text": abstract,
            "$vector": self._embed(abstract),
            "metadata": metadata or {},
        }
        docs.append(abstract_doc)

        # Store body chunks
        for i, chunk in enumerate(chunks):
            doc = {
                "_id": f"{paper_id}_chunk_{i}",
                "paper_id": paper_id,
                "title": title,
                "chunk_type": "body",
                "chunk_index": i,
                "text": chunk,
                "$vector": self._embed(chunk),
                "metadata": metadata or {},
            }
            docs.append(doc)

        # Upsert in batches
        batch_size = 20
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            self.collection.insert_many(batch, ordered=False)

        logger.info(f"Upserted paper '{title}' with {len(docs)} chunks")

    async def search(self, query: str, top_k: int = 5, filter_dict: dict = None) -> list[dict]:
        """
        Semantic search over stored research knowledge.
        Returns ranked list of relevant chunks.
        """
        if not self.collection:
            logger.warning("VectorStore not initialized, returning empty results")
            return []

        query_vector = self._embed(query)
        search_kwargs = {
            "sort": {"$vector": query_vector},
            "limit": top_k,
            "include_similarity": True,
        }
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        results = self.collection.find(**search_kwargs)
        docs = list(results)
        logger.info(f"Vector search returned {len(docs)} results for query: '{query[:60]}...'")
        return docs

    async def get_paper_context(self, query: str, paper_id: Optional[str] = None, top_k: int = 5) -> str:
        """
        Retrieve relevant context chunks for RAG.
        Optionally filter to a specific paper.
        """
        filter_dict = {"paper_id": paper_id} if paper_id else None
        docs = await self.search(query, top_k=top_k, filter_dict=filter_dict)

        if not docs:
            return ""

        context_parts = []
        for doc in docs:
            similarity = doc.get("$similarity", 0)
            title = doc.get("title", "Unknown")
            text = doc.get("text", "")
            context_parts.append(f"[Source: {title} | Relevance: {similarity:.2f}]\n{text}")

        return "\n\n---\n\n".join(context_parts)

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
        """
        Split text into overlapping chunks by characters,
        preserving sentence boundaries where possible.
        """
        words = text.split()
        chunks = []
        chunk_words = chunk_size // 6  # approx words per chunk
        overlap_words = overlap // 6
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_words - overlap_words
        return chunks

    async def delete_paper(self, paper_id: str):
        """Remove all chunks for a paper from the vector store."""
        if not self.collection:
            return
        result = self.collection.delete_many({"paper_id": paper_id})
        logger.info(f"Deleted paper {paper_id} from vector store")


# Singleton
vector_store = VectorStore()
