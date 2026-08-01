"""Deterministic text analysis for extracted media content.

Provides entity extraction (dates, times, money, links, phone numbers),
urgency detection, and safety/risk detection. Used by both the image and
audio processors so behavior stays consistent.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(
    r"(?:https?://|www\.)[^\s]+", re.IGNORECASE
)

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+?\d[\d\s\-()]{7,}\d)(?!\d)"
)

DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r"(?:\s+\d{1,2}(?:st|nd|rd|th)?)?"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    r")\b",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"\b(?:[01]?\d|2[0-3])(?:[:.][0-5]\d)?\s?(?:am|pm)\b"
    r"|\b(?:[01]?\d|2[0-3]):[0-5]\d\b",
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"\b(?:rs\.?|inr|usd|eur|gbp|₹|₨|\$|€|£)\s?\d[\d,]*(?:\.\d{1,2})?\b"
    r"|\b\d[\d,]*(?:\.\d{1,2})?\s?(?:rs\.?|inr|usd|eur|gbp|₹|₨|\$|€|£)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Urgency signal keywords
# ---------------------------------------------------------------------------

URGENCY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("deadline", re.compile(r"\b(?:deadline|closes?|closing)\b", re.IGNORECASE)),
    ("urgent", re.compile(r"\b(?:urgent|immediate|immediately|asap)\b", re.IGNORECASE)),
    ("time_sensitive", re.compile(r"\b(?:today|tonight|before|by\s+\d{1,2}(?::\d{2})?(?:\s?(?:am|pm))?|in\s+\d+\s*(?:min|minutes?|hrs?|hours?))\b", re.IGNORECASE)),
    ("action_required", re.compile(r"\b(?:please|must|required|need to|needs to|action required)\b", re.IGNORECASE)),
    ("expiry", re.compile(r"\b(?:expire|expires|expiring|ended?)\b", re.IGNORECASE)),
]

# ---------------------------------------------------------------------------
# Safety signal keywords / patterns
# ---------------------------------------------------------------------------

SAFETY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("otp_request", re.compile(r"\b(?:otp|one[\s-]?time[\s-]?password|verification code|login code)\b", re.IGNORECASE)),
    ("payment_request", re.compile(r"\b(?:pay|payment|upi|send\s+(?:money|amount)|transfer)\b", re.IGNORECASE)),
    ("pin_request", re.compile(r"\b(?:pin|password|passcode)\b", re.IGNORECASE)),
    ("bank_details", re.compile(r"\b(?:bank|account\s+number|card\s+number|cvv|debit|credit\s+card)\b", re.IGNORECASE)),
    ("prize_claim", re.compile(r"\b(?:congrats|winner|reward|voucher|lucky|prize|claim)\b", re.IGNORECASE)),
    ("account_block", re.compile(r"\b(?:block(?:ed)?|suspend(?:ed)?|restrict(?:ed)?|deactivat(?:e|ed)?|lock(?:ed)?)\b", re.IGNORECASE)),
    ("pay_now", re.compile(r"\b(?:pay\s+(?:now|today)|clear(?:ance)?\s+(?:amount|dues?|fee)|fee\s+due\s+today)\b", re.IGNORECASE)),
]

# Suspicious shortlink / wrong-domain cues for phishing detection.
PHISHING_DOMAIN_CUES = re.compile(
    r"\b(?:secure|verify|alert|help|support|reward|refund|login|account|kyc|delivery)[\s\-_.]*"
    r"(?:pay|secure|bank|login|verify)[\w\-_.]*\.(?:in|com|net|link|gl|cc|top|xyz|info)\b",
    re.IGNORECASE,
)

SUSPICIOUS_SHORTLINKS = re.compile(
    r"\b(?:bit\.ly|tinyurl|t\.co|shorturl|goo\.gl|rb\.gy|is\.gd|cutt\.ly|vl\.gl|link\.wame\.pro|(?:[a-z0-9-]+\.)?(?:xyz|top|cc|gl|link))\b",
    re.IGNORECASE,
)


def detect_entities(text: str) -> dict[str, list[str]]:
    """Extract dates, times, money, links, and phone numbers from text."""
    entities: dict[str, list[str]] = {
        "dates": _dedupe(DATE_PATTERN.findall(text)),
        "times": _dedupe(TIME_PATTERN.findall(text)),
        "money": _dedupe(MONEY_PATTERN.findall(text)),
        "links": _dedupe(URL_PATTERN.findall(text)),
        "phones": _dedupe(PHONE_PATTERN.findall(text)),
    }
    return entities


def detect_urgency(text: str) -> list[str]:
    """Return urgency indicators found in text."""
    found: list[str] = []
    for label, pattern in URGENCY_PATTERNS:
        if pattern.search(text):
            found.append(label)
    return _dedupe(found)


def detect_safety(text: str) -> list[str]:
    """Return safety/risk indicators found in text."""
    found: list[str] = []
    for label, pattern in SAFETY_PATTERNS:
        if pattern.search(text):
            found.append(label)

    if PHISHING_DOMAIN_CUES.search(text) or SUSPICIOUS_SHORTLINKS.search(text):
        found.append("phishing_links")

    if re.search(r"\b(?:qr|scan)\b", text, re.IGNORECASE):
        found.append("qr_payment_prompt")

    return _dedupe(found)


def _dedupe(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def safety_risk_level(text: str) -> str:
    """Return a coarse risk level: 'high', 'medium', 'low', or 'none'."""
    safety = detect_safety(text)
    if not safety:
        return "none"
    high_risk = {
        "otp_request",
        "pin_request",
        "bank_details",
        "pay_now",
        "phishing_links",
        "qr_payment_prompt",
    }
    if any(s in high_risk for s in safety):
        return "high"
    return "medium"


def analyze_text(text: str) -> dict[str, Any]:
    """Run all deterministic text analyses and return a combined dict."""
    return {
        "entities": detect_entities(text),
        "urgency_indicators": detect_urgency(text),
        "safety_indicators": detect_safety(text),
        "risk_level": safety_risk_level(text),
    }