# Prompt Strategy

## Prompt Evolution

### Version 1 (Phase 4) — Basic
Simple JSON output instructions with routing rules. The LLM often confused sender authenticity with notification priority.

### Version 2 (Phase 6) — Strict Output
Added strict JSON-only rules: no markdown, no code fences, no explanations. Improved output parsing reliability.

### Version 3 (Phase 6.5) — Field Renaming & Natural Language
- Renamed `Trust Score` → `Overall Notification Priority Score`
- Renamed `Business Trust` → `Business Verification Score`
- Replaced dense percentages with natural-language retrieval summaries
- Added structured prompt sections: USER PROFILE, BUSINESS CONTEXT, PRIORITY SIGNALS, CURRENT MESSAGE, SIMILAR HISTORY

### Version 4 (Phase 6.6) — Decision Intelligence
Added comprehensive sections:
- **Decision Guide**: Clear criteria for notify/digest/mute
- **Example Decision Matrix**: 7 quick-reference rules
- **Score Explanations**: Independent treatment of verification vs priority vs interest
- **DO NOT**: 8 explicit prohibitions (don't confuse authenticity with urgency, etc.)
- **Confidence Guide**: 5-tier calibration bands (0.95-1.00 down to below 0.40)
- **Few-shot Examples**: 6 concise examples covering all three actions

## Current System Prompt Structure

```
1. Role definition ("AI Notification Routing Engine")
2. Primary objective
3. DECISION GUIDE (notify/digest/mute criteria)
4. EXAMPLE DECISION MATRIX (7 rules)
5. SCORE EXPLANATIONS (3 independent scores)
6. DO NOT (8 prohibitions)
7. CONFIDENCE GUIDE (5 tiers)
8. OUTPUT RULES (strict JSON)
9. FEW-SHOT EXAMPLES (6 examples)
```

## User Prompt Structure

```
USER PROFILE (DND, opens, replies, dismissals, reports)
BUSINESS CONTEXT (name, category, verified, verification score, relationship)
SENDER CONTEXT (ID, relationship, reply/read/ignore rates)
GROUP CONTEXT (name, type, importance, muted)
USER PRIORITY SIGNALS (priority score, interest score)
CURRENT MESSAGE (ID, type, timestamp, media, forwarded, text)
MEDIA CONTENT (if present: summary, entities, urgency, safety)
RETRIEVAL SUMMARY (natural-language narrative)
SIMILAR HISTORY (top evidence messages with dates)
```

## JSON Schema

```json
{
  "action": "notify|digest|mute",
  "message_type": "personal|urgent|event|payment|business_update|promotion|greeting|forward|spam|scam|unknown",
  "reason": "short human-readable explanation",
  "confidence": 0.0
}
```

## Policy Interaction

The prompt is only sent to the LLM when no policy rule fires. Policy rules handle:
- Scam detection (OTP/PIN/phishing + low trust) → mute
- Trusted urgent (family/admin/verified bank + urgency) → notify
- Media failure (failed OCR/ASR) → digest
- Repetitive spam (high forward + high ignore) → mute

When a policy rule fires, the LLM is skipped entirely, saving API calls and improving reliability.

## Retrieval Context

The prompt includes a natural-language retrieval summary like:
> "The user is an active customer of HDFC Bank. Previously received 3 messages from HDFC Bank. Opened 3 of them. Replied to 1. The most recent HDFC Bank message was on 2026-04-18."

This is more effective than dense percentages for LLM reasoning.

## Prompt Design Rationale

1. **Structured sections** make the context easy for the LLM to parse
2. **Natural language** summaries reduce confusion vs raw numbers
3. **Score separation** prevents the model from conflating authenticity with priority
4. **Few-shot examples** anchor the model's output format and decision patterns
5. **DO NOT rules** directly address observed failure modes
6. **Confidence guide** helps the model produce better-calibrated confidence scores

Cross-references: [Architecture](architecture.md) | [System Design](system_design.md) | [Evaluation](evaluation.md)