"""Central configuration for the Message Notification Router.

All paths and tunable settings live here. Environment variables can override
defaults where appropriate (e.g. DATASET_DIR, LOG_LEVEL, OUTPUT_DIR).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the repository (code/ is one level below the repo root).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Directory containing all participant-facing CSVs.
DATASET_DIR: Path = Path(
    os.getenv("DATASET_DIR", str(REPO_ROOT / "dataset"))
).resolve()

# Directory where generated artifacts (e.g. output.csv) are written.
OUTPUT_DIR: Path = Path(
    os.getenv("OUTPUT_DIR", str(REPO_ROOT / "code" / "output"))
).resolve()

# Media root referenced by images.csv / voice_notes.csv (relative paths).
MEDIA_DIR: Path = DATASET_DIR / "media"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# Dataset files (name -> expected CSV filename inside DATASET_DIR)
# ---------------------------------------------------------------------------

DATASET_FILES: dict[str, str] = {
    "messages": "messages.csv",
    "users": "users.csv",
    "groups": "groups.csv",
    "group_members": "group_members.csv",
    "business_accounts": "business_accounts.csv",
    "user_business_history": "user_business_history.csv",
    "message_history": "message_history.csv",
    "message_events": "message_events.csv",
    "images": "images.csv",
    "voice_notes": "voice_notes.csv",
    "daily_notification_summary": "daily_notification_summary.csv",
}

# ---------------------------------------------------------------------------
# DataLoader behaviour
# ---------------------------------------------------------------------------

# Columns that should be parsed as dates when loading CSVs.
DATE_COLUMNS: dict[str, list[str]] = {
    "messages": ["created_at"],
    "groups": ["created_at"],
    "group_members": ["joined_at"],
    "user_business_history": ["last_activity_at", "promotions_opted_out_at", "last_reply_at"],
    "message_history": ["created_at"],
    "daily_notification_summary": ["date"],
}

# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

OUTPUT_COLUMNS: list[str] = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))