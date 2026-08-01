"""Deterministic policy engine for the Message Notification Router.

The policy engine runs between the MediaProcessor and the LLM Router.
It evaluates RetrievalResult, MediaResult, and MessageContext with
deterministic rules. If a rule fires, the LLM is skipped entirely and
the final decision is produced directly.
"""