"""Async client for the model server (vLLM, Ollama, TGI, llama.cpp: anything OpenAI-compatible)."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass

import httpx

RETRYABLE = {408, 429, 500, 502, 503, 504}


class BackendError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class ChatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        max_tokens: int = 256,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.models_url = base_url.rstrip("/") + "/models"
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.AsyncClient(timeout=timeout_s, headers=headers)

    async def complete(self, messages: list[dict[str, str]]) -> Completion:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            # SQL never needs a second statement; stopping early saves tokens and latency
            "stop": [";\n\n", "\n\n\n"],
        }
        for attempt in range(self.max_retries + 1):
            try:
                resp = await self.client.post(self.url, json=body)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == self.max_retries:
                    raise BackendError(f"model server unreachable: {exc!r}") from exc
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    return Completion(
                        text=data["choices"][0]["message"].get("content") or "",
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                    )
                if resp.status_code not in RETRYABLE or attempt == self.max_retries:
                    raise BackendError(f"model server returned HTTP {resp.status_code}")
            await asyncio.sleep(min(4.0, 0.25 * 2**attempt) * random.uniform(0.5, 1.0))
        raise BackendError("unreachable")  # pragma: no cover

    async def ready(self) -> bool:
        try:
            resp = await self.client.get(self.models_url, timeout=3.0)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def aclose(self) -> None:
        await self.client.aclose()
