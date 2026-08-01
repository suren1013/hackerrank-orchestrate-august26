"""Prompt templates for the routing LLM.

Later phases will fill in the actual system/user prompts used to make
routing decisions. This module keeps prompts centralized and versioned.
"""

from __future__ import annotations

SYSTEM_PROMPT: str = (
    "You are a message notification router for WhatsApp. "
    "Decide whether each incoming message should be notified, digested, or muted."
)

USER_PROMPT_TEMPLATE: str = (
    "Given the following message context, decide the routing action.\n\n"
    "{context}"
)