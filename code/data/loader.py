"""DataLoader: loads all dataset CSVs and assembles unified message contexts.

The loader is intentionally free of any routing/LLM/OCR logic. It only:

1. Reads every required CSV from ``dataset/`` into a pandas DataFrame.
2. Exposes those DataFrames via attributes and a ``get()`` method.
3. Builds a :class:`MessageContext` for every row in ``messages.csv`` by
   joining relevant metadata from the other tables based on the actual
   dataset schema (no invented columns).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import DATASET_DIR, DATASET_FILES, DATE_COLUMNS
from models.schemas import MessageContext
from utils.helpers import is_empty, safe_str
from utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Loads all challenge CSVs and assembles unified message contexts."""

    def __init__(self, dataset_dir: Path | str | None = None) -> None:
        self.dataset_dir: Path = Path(dataset_dir or DATASET_DIR).resolve()
        self._dataframes: dict[str, pd.DataFrame] = {}
        self._missing_files: list[str] = []
        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def dataframes(self) -> dict[str, pd.DataFrame]:
        """All loaded DataFrames keyed by dataset name."""
        return self._dataframes

    @property
    def missing_files(self) -> list[str]:
        """Names of expected dataset files that were not found."""
        return list(self._missing_files)

    def get(self, name: str) -> pd.DataFrame:
        """Return the DataFrame for a dataset name, or an empty DataFrame."""
        return self._dataframes.get(name, pd.DataFrame())

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame: dataset name, rows, columns."""
        rows = []
        for name, df in self._dataframes.items():
            rows.append(
                {
                    "dataset": name,
                    "rows": len(df),
                    "columns": ", ".join(df.columns),
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load every required CSV, logging progress and missing files."""
        logger.info("Loading datasets from %s", self.dataset_dir)

        for name, filename in DATASET_FILES.items():
            path = self.dataset_dir / filename
            if not path.exists():
                logger.warning("Missing dataset file: %s", filename)
                self._missing_files.append(filename)
                self._dataframes[name] = pd.DataFrame()
                continue

            try:
                df = pd.read_csv(path)
                self._apply_date_parsing(name, df)
                self._dataframes[name] = df
                logger.info(
                    "Loaded %s (%d rows, %d cols)",
                    filename,
                    len(df),
                    len(df.columns),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load %s: %s", filename, exc)
                self._missing_files.append(filename)
                self._dataframes[name] = pd.DataFrame()

        if self._missing_files:
            logger.warning(
                "Missing %d file(s): %s",
                len(self._missing_files),
                ", ".join(self._missing_files),
            )
        else:
            logger.info("All dataset files loaded successfully.")

    def _apply_date_parsing(self, name: str, df: pd.DataFrame) -> None:
        """Parse configured date columns for a dataset."""
        for col in DATE_COLUMNS.get(name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Lookup index construction
    # ------------------------------------------------------------------

    def _build_lookup(
        self, name: str, key_cols: list[str]
    ) -> dict[tuple[Any, ...], dict[str, Any]]:
        """Build a dict keyed by tuple of key column values -> row dict."""
        df = self.get(name)
        if df.empty:
            return {}

        lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = tuple(row[col] for col in key_cols)
            if any(is_empty(v) for v in key):
                continue
            lookup[key] = row.to_dict()
        return lookup

    def _build_multi_lookup(
        self, name: str, key_cols: list[str]
    ) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
        """Build a dict keyed by tuple of key column values -> list of row dicts."""
        df = self.get(name)
        if df.empty:
            return {}

        lookup: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for _, row in df.iterrows():
            key = tuple(row[col] for col in key_cols)
            if any(is_empty(v) for v in key):
                continue
            lookup.setdefault(key, []).append(row.to_dict())
        return lookup

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_contexts(self) -> list[MessageContext]:
        """Assemble a MessageContext for every row in messages.csv.

        Returns
        -------
        list[MessageContext]
            One context per incoming message, in the same order as messages.csv.
        """
        messages = self.get("messages")
        if messages.empty:
            logger.warning("messages.csv is empty; no contexts to build.")
            return []

        # Pre-build lookup indexes for efficient joins.
        users = self._build_lookup("users", ["user_id"])
        groups = self._build_lookup("groups", ["group_id"])
        group_members = self._build_lookup(
            "group_members", ["group_id", "user_id"]
        )
        businesses = self._build_lookup("business_accounts", ["business_id"])
        user_business = self._build_lookup(
            "user_business_history", ["user_id", "business_id"]
        )
        images = self._build_lookup("images", ["image_id"])
        voice_notes = self._build_lookup("voice_notes", ["voice_note_id"])

        # Daily notification summary: keyed by (user_id, date).
        daily = self._build_lookup(
            "daily_notification_summary", ["user_id", "date"]
        )

        # Historical messages: keyed by user_id -> list of message dicts.
        hist_by_user = self._build_multi_lookup(
            "message_history", ["user_id"]
        )

        # Historical events: keyed by message_id -> list of event dicts.
        events_by_message = self._build_multi_lookup(
            "message_events", ["message_id"]
        )

        contexts: list[MessageContext] = []
        for _, msg in messages.iterrows():
            ctx = self._build_single_context(
                msg=msg,
                users=users,
                groups=groups,
                group_members=group_members,
                businesses=businesses,
                user_business=user_business,
                images=images,
                voice_notes=voice_notes,
                daily=daily,
                hist_by_user=hist_by_user,
                events_by_message=events_by_message,
            )
            contexts.append(ctx)

        logger.info("Assembled %d message contexts.", len(contexts))
        return contexts

    def _build_single_context(
        self,
        msg: pd.Series,
        users: dict[tuple[Any, ...], dict[str, Any]],
        groups: dict[tuple[Any, ...], dict[str, Any]],
        group_members: dict[tuple[Any, ...], dict[str, Any]],
        businesses: dict[tuple[Any, ...], dict[str, Any]],
        user_business: dict[tuple[Any, ...], dict[str, Any]],
        images: dict[tuple[Any, ...], dict[str, Any]],
        voice_notes: dict[tuple[Any, ...], dict[str, Any]],
        daily: dict[tuple[Any, ...], dict[str, Any]],
        hist_by_user: dict[tuple[Any, ...], list[dict[str, Any]]],
        events_by_message: dict[tuple[Any, ...], list[dict[str, Any]]],
    ) -> MessageContext:
        """Build a single MessageContext for one incoming message row."""
        message_id = safe_str(msg.get("message_id"))
        user_id = safe_str(msg.get("user_id"))
        conversation_type = safe_str(msg.get("conversation_type"))
        group_id = safe_str(msg.get("group_id"))
        business_id = safe_str(msg.get("business_id"))
        sender_user_id = safe_str(msg.get("sender_user_id"))
        created_at = msg.get("created_at")
        media_type = safe_str(msg.get("media_type"))
        media_id = safe_str(msg.get("media_id"))
        forwarded_count = msg.get("forwarded_count", 0)
        if pd.isna(forwarded_count):
            forwarded_count = 0

        # --- Direct metadata joins -------------------------------------
        user = users.get((user_id,)) if user_id else None

        group = None
        group_member = None
        if conversation_type == "group" and group_id:
            group = groups.get((group_id,))
            if user_id:
                group_member = group_members.get((group_id, user_id))

        business = None
        ub_history = None
        if conversation_type == "business" and business_id:
            business = businesses.get((business_id,))
            if user_id:
                ub_history = user_business.get((user_id, business_id))

        # --- Media joins ------------------------------------------------
        image = None
        voice_note = None
        if media_type == "image" and media_id:
            image = images.get((media_id,))
        elif media_type == "voice" and media_id:
            voice_note = voice_notes.get((media_id,))

        # --- Daily notification summary for the message date -----------
        daily_notification = None
        if user_id and created_at is not None and not is_empty(created_at):
            date_key = pd.Timestamp(created_at).date()
            daily_notification = daily.get((user_id, date_key))

        # --- Historical messages for this user -------------------------
        historical_messages: list[dict[str, Any]] = []
        if user_id:
            for hist in hist_by_user.get((user_id,), []):
                # Only include messages strictly before the incoming one.
                hist_ts = hist.get("created_at")
                if hist_ts is not None and not is_empty(hist_ts):
                    if pd.Timestamp(hist_ts) < pd.Timestamp(created_at):
                        historical_messages.append(hist)
                else:
                    historical_messages.append(hist)

        # --- Historical events for the user's historical messages ------
        historical_events: list[dict[str, Any]] = []
        for hist in historical_messages:
            hist_id = safe_str(hist.get("message_id"))
            if hist_id:
                historical_events.extend(events_by_message.get((hist_id,), []))

        return MessageContext(
            message_id=message_id,
            user_id=user_id,
            conversation_type=conversation_type,
            created_at=str(created_at) if created_at is not None else "",
            message_text=safe_str(msg.get("message_text")),
            media_type=media_type,
            media_id=media_id,
            forwarded_count=int(forwarded_count),
            user=user,
            group=group,
            group_member=group_member,
            business=business,
            user_business=ub_history,
            image=image,
            voice_note=voice_note,
            daily_notification=daily_notification,
            historical_messages=historical_messages,
            historical_events=historical_events,
        )