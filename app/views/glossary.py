"""Single plain-language vocabulary used by every page and the Glossary.

Keeping every label and definition in one dictionary stops pages from
drifting apart (UI/UX requirements, Section 6).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    """One concept: the label shown in the app and its definition."""

    label: str
    definition: str


TERMS: dict[str, Term] = {
    "churn": Term("Customer left", "The customer ended the service during the period"),
    "probability": Term(
        "Chance of leaving", "How likely the customer is to leave, from 0 to 100%"
    ),
    "predicted_rate": Term(
        "Expected share leaving",
        "Average chance of leaving across the customers in a batch",
    ),
    "threshold": Term(
        "Risk cut-off",
        "The chance of leaving above which a customer is flagged for contact",
    ),
    "predicted_positive": Term(
        "Flagged for contact", "Customers the model suggests calling"
    ),
    "recall": Term(
        "Leavers we catch", "Of all customers who leave, the share the model flags"
    ),
    "precision": Term(
        "Correct flags", "Of all flagged customers, the share who would actually leave"
    ),
    "false_positive": Term("Unneeded call", "A flagged customer who would have stayed"),
    "false_negative": Term(
        "Missed leaver", "A customer who left without being flagged"
    ),
    "importance": Term(
        "Top reasons",
        "The customer characteristics that most influence the prediction",
    ),
    "plan": Term("Plan", "The subscription the customer is on"),
    "lift": Term(
        "Better than random by",
        "How many times more leavers the model finds than random calls",
    ),
}


def label(key: str) -> str:
    """Label of a term as shown in the app."""
    return TERMS[key].label


def tooltip(key: str) -> str:
    """Plain-language definition shown as a tooltip."""
    return f"{TERMS[key].definition}."
