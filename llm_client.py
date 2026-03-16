"""
Groq LLM Client — Production-grade wrapper for Groq API inference.
Handles streaming, retries, and token management.
"""

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger
from config import settings
from typing import Optional, AsyncIterator
import groq


class GroqClient:
    """Async Groq API client with retry logic and structured prompting."""

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((groq.RateLimitError, groq.APIConnectionError)),
    )
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a completion request to Groq and return full response text.
        """
        logger.info(f"Groq request | model={self.model} | max_tokens={max_tokens}")
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result = response.choices[0].message.content
        logger.info(f"Groq response received | chars={len(result)}")
        return result

    async def stream(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from Groq API for real-time output.
        """
        logger.info(f"Groq stream start | model={self.model}")
        async with self.client.chat.completions.stream(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        ) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def multi_turn(
        self,
        messages: list[dict],
        system_prompt: str,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        """
        Multi-turn conversation (for research Q&A with history).
        """
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


# Singleton
groq_client = GroqClient()
