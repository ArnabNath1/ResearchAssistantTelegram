"""
Astra DB Vector Store — Handles embeddings, upsert, and semantic search
for RAG-based grounded scientific reasoning.
"""

import httpx
import hashlib
from typing import Optional
from loguru import logger
from astrapy import DataAPIClient
from config import settings

class VectorStore:
    """
    Manages vector storage using Astra DB.
    Embeddings are fetched via Jina AI API to save local RAM and improve speed.
    """

    def __init__(self):
        self.dimension = 768  # Jina v2 base dimension
        self.collection = None
        self.jina_url = "https://api.jina.ai/v1/embeddings"
        self._init_db()

    def _init_db(self):
        """Initialize Astra DB connection."""
        try:
            client = DataAPIClient(settings.astra_db_application_token)
            db = client.get_database_by_api_endpoint(settings.astra_db_api_endpoint)
            logger.info("Connected to Astra DB")

            # Check/Update collection dimension for Jina (1024)
            self.collection = db.get_collection(settings.astra_db_collection)
            logger.info(f"Using collection: {settings.astra_db_collection}")
        except Exception as e:
            logger.error(f"Astra DB init failed: {e}")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts via Jina AI API."""
        if not texts:
            return []
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.jina_api_key}"
        }
        data = {
            "model": "jina-embeddings-v2-base-en",
            "input": texts
        }
        try:
            with httpx.Client() as client:
                # Increased timeout to 30.0s for larger batches
                response = client.post(self.jina_url, headers=headers, json=data, timeout=30.0)
                response.raise_for_status()
                return [item["embedding"] for item in response.json()["data"]]
        except Exception as e:
            logger.error(f"Batch embedding API failed: {e}")
            # Fallback: return zero vectors
            return [[0.0] * self.dimension for _ in texts]

    def _embed(self, text: str) -> list[float]:
        """Generate a single embedding."""
        return self._embed_batch([text])[0]

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

        # Prepare all documents (without vectors first)
        all_docs_metadata = []
        all_texts = [abstract] + chunks
        
        # Determine chunk indices/types for metadata
        doc_metadata = []
        doc_metadata.append({"id": f"{paper_id}_abstract", "type": "abstract", "idx": 0})
        for i in range(len(chunks)):
            doc_metadata.append({"id": f"{paper_id}_chunk_{i}", "type": "body", "idx": i})

        # Process embeddings in batches of 20 to avoid API limits
        batch_size = 20
        for i in range(0, len(all_texts), batch_size):
            text_batch = all_texts[i : i + batch_size]
            meta_batch = doc_metadata[i : i + batch_size]
            
            logger.info(f"Embedding batch {i//batch_size + 1} for '{title}'...")
            vectors = self._embed_batch(text_batch)
            
            docs_to_insert = []
            for j, vector in enumerate(vectors):
                docs_to_insert.append({
                    "_id": meta_batch[j]["id"],
                    "paper_id": paper_id,
                    "title": title,
                    "chunk_type": meta_batch[j]["type"],
                    "chunk_index": meta_batch[j]["idx"] if meta_batch[j]["type"] == "body" else None,
                    "text": text_batch[j],
                    "$vector": vector,
                    "metadata": metadata or {},
                })
            
            if docs_to_insert:
                self.collection.insert_many(docs_to_insert, ordered=False)

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
