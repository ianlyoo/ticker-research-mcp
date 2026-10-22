"""Operational switches for the standalone research service."""
from __future__ import annotations

import os
import re

TICKER_TECHNICAL_ENABLED = os.environ.get("TICKER_TECHNICAL_ENABLED", "true") in {"1", "true"}

_EMAIL = re.compile(r"(?<![A-Za-z0-9._%+@-])[A-Za-z0-9][A-Za-z0-9._%+-]*@"
                    r"(?P<domain>localhost|(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
                    r"[A-Za-z]{2,63})(?![A-Za-z0-9._@-])", re.I)
_PLACEHOLDER_DOMAINS = {"example.com", "example.org", "example.net", "example", "invalid", "localhost"}


def sec_user_agent(value):
    """Validate without echoing contact details or accepting header controls."""
    if (not isinstance(value, str) or not 0 < len(value) <= 200
            or any(not 32 <= ord(char) <= 126 for char in value)):
        return None
    contacts = list(_EMAIL.finditer(value))
    if not contacts or any(
        match.group("domain").lower() == domain or match.group("domain").lower().endswith("." + domain)
        for match in contacts for domain in _PLACEHOLDER_DOMAINS
    ):
        return None
    return value


# Capture only the validated value; runtime composition never rereads the env.
SEC_USER_AGENT = sec_user_agent(os.environ.get("SEC_USER_AGENT"))

