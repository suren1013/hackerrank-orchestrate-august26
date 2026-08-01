"""Analysis modules for sender, business, group, and engagement relationships.

Each analyzer is a pure function of a MessageContext and returns a typed
summary object. No LLM, OCR, or ASR is used — everything is deterministic
and based on the loaded dataset fields.
"""

from __future__ import annotations

from typing import Any

from models.schemas import MessageContext
from retrieval.types import (
    BusinessRelationship,
    EngagementSummary,
    GroupRelationship,
    SenderRelationship,
)
from utils.helpers import is_empty, safe_str


# ---------------------------------------------------------------------------
# Sender analysis
# ---------------------------------------------------------------------------


def analyze_sender(context: MessageContext) -> SenderRelationship | None:
    """Analyze the receiver's historical interaction with the sender.

    For personal/group messages the sender is ``sender_user_id``. For
    business messages there is no user sender, so we return None.
    """
    sender_id = context.sender_user_id
    if not sender_id:
        return None

    # Historical messages from this sender to this user.
    sender_messages = [
        m
        for m in context.historical_messages
        if safe_str(m.get("sender_user_id")) == sender_id
    ]

    # Build a map of message_id -> event for quick lookup.
    event_by_msg: dict[str, dict[str, Any]] = {}
    for ev in context.historical_events:
        mid = safe_str(ev.get("message_id"))
        if mid:
            event_by_msg[mid] = ev

    total = len(sender_messages)
    reply_count = 0
    read_count = 0
    ignore_count = 0
    archive_count = 0

    for m in sender_messages:
        mid = safe_str(m.get("message_id"))
        ev = event_by_msg.get(mid, {})
        if ev.get("message_replied") == 1:
            reply_count += 1
        if ev.get("message_opened") == 1:
            read_count += 1
        if ev.get("notification_dismissed") == 1:
            ignore_count += 1
        if ev.get("muted_after_message") == 1:
            archive_count += 1

    rel = SenderRelationship(
        sender_id=sender_id,
        total_messages=total,
        reply_count=reply_count,
        read_count=read_count,
        ignore_count=ignore_count,
        archive_count=archive_count,
    )

    if total > 0:
        rel.reply_frequency = reply_count / total
        rel.read_frequency = read_count / total
        rel.ignore_frequency = ignore_count / total
        rel.archive_frequency = archive_count / total

    # Label the relationship.
    if total == 0:
        rel.relationship_label = "new_sender"
    elif rel.reply_frequency >= 0.5:
        rel.relationship_label = "highly_engaged"
    elif rel.reply_frequency >= 0.2:
        rel.relationship_label = "engaged"
    elif rel.ignore_frequency >= 0.5 or rel.archive_frequency >= 0.5:
        rel.relationship_label = "ignored"
    else:
        rel.relationship_label = "casual"

    return rel


# ---------------------------------------------------------------------------
# Business analysis
# ---------------------------------------------------------------------------


def analyze_business(context: MessageContext) -> BusinessRelationship | None:
    """Analyze the receiver's relationship with a business sender."""
    business_id = context.business_id
    if not business_id:
        return None

    biz = context.business or {}
    ub = context.user_business or {}

    display_name = safe_str(biz.get("display_name"))
    category = safe_str(biz.get("category"))
    verified = bool(biz.get("verified") == 1)

    is_known = bool(ub)
    is_active_customer = False
    has_frequent = False
    allows_promos = bool(ub.get("allows_promotions") == 1)
    opted_out = not is_empty(ub.get("promotions_opted_out_at"))

    why = safe_str(ub.get("why_user_knows_account"))
    if why in {
        "active_bank_account",
        "active_credit_card",
        "active_payment_wallet",
        "recent_grocery_delivery",
        "recent_card_payment",
        "monthly_utility_bill",
        "monthly_maintenance_payment",
        "delivery_expected_today",
        "ride_booked_today",
        "recent_product_purchase",
        "recent_return_pickup",
        "upcoming_clinic_appointment",
        "confirmed_travel_booking",
        "prescription_refill",
        "active_retail_membership",
        "frequent_food_orders",
        "frequent_delivery_updates",
        "active_sale_subscription",
        "recent_movie_booking",
        "recent_flight_booking",
        "active_us_bank_account",
        "active_global_bank_account",
        "trading_account_update",
        "society_payment_receipt",
        "repair_booking",
        "travel_card_account",
        "caregiver_insurance_claim",
        "medicine_order",
        "security_webinar_registration",
        "campus_event_registration",
        "student_event_booking",
        "education_survey_or_event",
        "workshop_registration",
        "local_event_booking",
        "book_adaptation_watchlist",
        "dance_class_booking",
        "dance_event_booking",
        "recent_delivery_order",
        "health_product_order",
        "land_listing_watchlist",
        "vendor_payment_account",
        "tech_webinar_registration",
        "movie_or_event_booking",
        "recent_food_order",
        "security_incident_updates",
        "vendor_security_notice",
        "work_vendor_payment",
        "office_delivery_order",
        "daytime_food_order",
        "streaming_subscription",
        "school_event_registration",
        "school_fee_payment",
        "marketplace_style_interest",
        "local_food_orders",
        "community_workshop_registration",
        "event_registration",
        "music_subscription",
        "traffic_challan_notice",
        "vehicle_insurance_renewal",
        "vehicle_service_booking",
        "hotel_booking",
        "restaurant_reservation",
        "cashback_wallet",
        "coupon_membership",
        "alumni_event_booking",
        "career_event_registration",
        "travel_package_interest",
        "business_payment_stack_interest",
        "new_food_delivery_signup",
        "registered_for_campus_event",
        "recent_movie_feedback",
    }:
        is_active_customer = True

    activity_count = ub.get("activity_count_180d", 0)
    if not is_empty(activity_count) and int(activity_count) >= 5:
        has_frequent = True

    # Trust score: start at 0, add signals.
    trust = 0.0
    if verified:
        trust += 0.4
    if is_known:
        trust += 0.2
    if is_active_customer:
        trust += 0.2
    if has_frequent:
        trust += 0.1
    if opted_out:
        trust -= 0.1
    trust = max(0.0, min(1.0, trust))

    label = "unknown"
    if not is_known:
        label = "unfamiliar_business"
    elif opted_out:
        label = "opted_out"
    elif is_active_customer and has_frequent:
        label = "trusted_active_customer"
    elif is_active_customer:
        label = "active_customer"
    elif allows_promos:
        label = "promotional_subscriber"
    else:
        label = "known_business"

    return BusinessRelationship(
        business_id=business_id,
        display_name=display_name,
        verified=verified,
        category=category,
        is_known_business=is_known,
        is_active_customer=is_active_customer,
        has_frequent_interactions=has_frequent,
        allows_promotions=allows_promos,
        opted_out=opted_out,
        trust_score=trust,
        relationship_label=label,
    )


# ---------------------------------------------------------------------------
# Group analysis
# ---------------------------------------------------------------------------


def analyze_group(context: MessageContext) -> GroupRelationship | None:
    """Analyze the receiver's relationship with a group."""
    group_id = context.group_id
    if not group_id:
        return None

    grp = context.group or {}
    member = context.group_member or {}

    group_name = safe_str(grp.get("group_name"))
    group_type = safe_str(grp.get("group_type"))
    role = safe_str(member.get("role"))
    is_member = bool(member)
    is_admin = role == "admin"
    is_muted = bool(member.get("group_muted_by_user") == 1)

    # Active if the user has read or sent messages recently.
    messages_read = member.get("messages_read_30d", 0)
    messages_sent = member.get("messages_sent_30d", 0)
    is_active = (
        not is_empty(messages_read) and int(messages_read) > 0
    ) or (not is_empty(messages_sent) and int(messages_sent) > 0)

    # Announcement groups: society, school, college_faculty, safety, etc.
    announcement_types = {
        "society",
        "school_group",
        "college_faculty",
        "safety",
        "finance_help",
        "real_estate",
        "investment_tips",
    }
    is_announcement = group_type in announcement_types

    # Importance label.
    if is_muted:
        label = "muted"
    elif is_admin:
        label = "admin"
    elif is_announcement:
        label = "announcement"
    elif group_type in {"family", "extended_family"}:
        label = "family"
    elif group_type in {"coworker", "college_students"}:
        label = "office"
    elif group_type in {"society", "school_group", "college_faculty", "safety"}:
        label = "community"
    elif is_active:
        label = "active"
    else:
        label = "inactive"

    return GroupRelationship(
        group_id=group_id,
        group_name=group_name,
        group_type=group_type,
        is_member=is_member,
        is_admin=is_admin,
        is_muted=is_muted,
        is_active=is_active,
        is_announcement_group=is_announcement,
        importance_label=label,
    )


# ---------------------------------------------------------------------------
# Engagement summary
# ---------------------------------------------------------------------------


def analyze_engagement(context: MessageContext) -> EngagementSummary:
    """Aggregate engagement metrics across all historical messages."""
    total = len(context.historical_messages)
    opened = 0
    replied = 0
    dismissed = 0
    muted = 0
    reported = 0

    for ev in context.historical_events:
        if ev.get("message_opened") == 1:
            opened += 1
        if ev.get("message_replied") == 1:
            replied += 1
        if ev.get("notification_dismissed") == 1:
            dismissed += 1
        if ev.get("muted_after_message") == 1:
            muted += 1
        if ev.get("message_reported") == 1:
            reported += 1

    summary = EngagementSummary(
        total_historical=total,
        opened=opened,
        replied=replied,
        dismissed=dismissed,
        muted=muted,
        reported=reported,
    )

    if total > 0:
        summary.open_rate = opened / total
        summary.reply_rate = replied / total
        summary.dismiss_rate = dismissed / total
        summary.mute_rate = muted / total
        summary.report_rate = reported / total

    return summary