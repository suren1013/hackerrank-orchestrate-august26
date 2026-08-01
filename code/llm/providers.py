"""LLM provider abstraction.

Defines an ``LLMProvider`` interface and concrete implementations:

- ``OpenAIProvider``  — OpenAI-compatible chat completions API (works with
  OpenAI, Azure OpenAI, Groq, Together, etc. via base_url).
- ``GeminiProvider``  — Google Gemini generateContent API.
- ``OllamaProvider``  — Local Ollama chat API.
- ``MockProvider``    — Deterministic rule-based fallback (no network).

The router only depends on the ``LLMProvider`` interface, so switching
models is a matter of changing environment variables.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import requests

from llm.schemas import LLMResponse
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    """Interface for an LLM provider that returns structured JSON."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Send a chat completion request and return a structured response."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible chat completions provider."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not self.api_key:
            logger.warning("OpenAIProvider: no OPENAI_API_KEY set.")
        logger.info("OpenAIProvider initialized (model=%s).", self.model)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.base_url}/chat/completions" if self.base_url else (
            "https://api.openai.com/v1/chat/completions"
        )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return _parse_llm_json(content)


class GeminiProvider(LLMProvider):
    """Google Gemini generateContent provider."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        if not self.api_key:
            logger.warning("GeminiProvider: no GEMINI_API_KEY set.")
        logger.info("GeminiProvider initialized (model=%s).", self.model)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": system_prompt + "\n\n" + user_prompt},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_llm_json(content)


class OllamaProvider(LLMProvider):
    """Local Ollama chat provider."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
        logger.info("OllamaProvider initialized (model=%s).", self.model)

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"]
        return _parse_llm_json(content)


class MockProvider(LLMProvider):
    """Deterministic rule-based provider for testing without an API key.

    Produces a reasonable routing decision based on simple heuristics so
    the full pipeline can be exercised offline.
    """

    def __init__(self) -> None:
        logger.info("MockProvider initialized (deterministic fallback).")

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        # Extract key signals from the prompt text.
        text = user_prompt.lower()

        # Safety signals -> mute.
        safety_terms = [
            "otp", "verification code", "login code", "pin", "password",
            "account number", "card number", "cvv", "phishing", "scam",
            "pay now", "qr", "blocked", "suspended", "restricted",
        ]
        if any(term in text for term in safety_terms):
            return LLMResponse(
                action="mute",
                message_type="scam" if any(
                    t in text for t in ["otp", "verification code", "login code", "pin", "password", "phishing", "scam"]
                ) else "spam",
                reason="Message contains safety/risk signals (OTP, payment, or phishing cues).",
                confidence=0.85,
                raw="mock",
            )

        # Urgency signals -> notify.
        urgency_terms = [
            "urgent", "immediately", "asap", "deadline", "today", "tonight",
            "before", "now", "call me", "need to", "must",
        ]
        if any(term in text for term in urgency_terms):
            return LLMResponse(
                action="notify",
                message_type="urgent",
                reason="Message contains urgency signals requiring immediate attention.",
                confidence=0.8,
                raw="mock",
            )

        # Greeting / forward -> digest or mute.
        greeting_terms = ["good morning", "good evening", "blessing", "forward", "share"]
        if any(term in text for term in greeting_terms):
            return LLMResponse(
                action="digest",
                message_type="greeting" if "good morning" in text or "good evening" in text else "forward",
                reason="Message is a greeting or forward that can be read later.",
                confidence=0.7,
                raw="mock",
            )

        # Default: digest.
        return LLMResponse(
            action="digest",
            message_type="unknown",
            reason="No strong urgency or safety signals detected.",
            confidence=0.6,
            raw="mock",
        )


def _parse_llm_json(content: str) -> LLMResponse:
    """Parse a JSON string from an LLM into an LLMResponse."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Try to extract a JSON object from the text.
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            data = json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse LLM JSON output.")
            return LLMResponse(raw=content)

    return LLMResponse(
        action=str(data.get("action", "")).strip(),
        message_type=str(data.get("message_type", "")).strip(),
        reason=str(data.get("reason", "")).strip(),
        confidence=float(data.get("confidence", 0.0)),
        raw=content,
    )


def create_provider() -> LLMProvider:
    """Create an LLM provider based on the LLM_PROVIDER env var.

    Supported values: openai, gemini, ollama, mock (default).
    """
    provider_name = os.getenv("LLM_PROVIDER", "mock").strip().lower()

    if provider_name == "openai":
        return OpenAIProvider()
    if provider_name == "gemini":
        return GeminiProvider()
    if provider_name == "ollama":
        return OllamaProvider()
    if provider_name == "mock":
        return MockProvider()

    logger.warning(
        "Unknown LLM_PROVIDER '%s'; falling back to mock.", provider_name
    )
    return MockProvider()