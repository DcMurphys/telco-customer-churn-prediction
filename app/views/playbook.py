"""Retention Playbook wording for plans (UI/UX requirements, Section 7).

A starting proposal to be validated by the customer relations team; it states
no proven effect.
"""

from __future__ import annotations

PLAN_TALKING_POINTS: dict[tuple[str, str], list[str]] = {
    ("Contract", "Month-to-Month"): [
        "Present a one- or two-year plan with a loyalty discount.",
        "Explain the price lock that comes with a longer contract.",
    ],
    ("Internet type", "Fiber Optic"): [
        "Offer a trial of premium tech support or online security.",
        "Ask about speed or outage issues and log them for the network team.",
    ],
    ("Add-ons", "0 add-ons"): [
        "Review the plan; suggest a bundle with security or support add-ons.",
    ],
    ("Add-ons", "1–2 add-ons"): [
        "Check whether the add-ons match what the customer uses.",
        "Suggest a better-fitting bundle instead of single add-ons.",
    ],
    ("Offer", "Offer E"): [
        "Check whether the promotional period is ending soon.",
        "Present the next suitable offer before the current one expires.",
    ],
    ("Offer", "None"): [
        "Check eligibility for a current promotional offer.",
    ],
}
DEFAULT_TALKING_POINTS = [
    "Review the plan with the customer; suggest a better-fitting bundle.",
    "Mention the referral reward.",
]


def talking_points_for(attribute: str, group: str) -> list[str]:
    """Suggested conversation points for one plan group."""
    return PLAN_TALKING_POINTS.get((attribute, group), DEFAULT_TALKING_POINTS)
