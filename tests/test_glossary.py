"""Every term has one plain-language label and a definition (Section 6)."""

from app.views.glossary import TERMS, label, tooltip


def test_every_term_has_a_label_and_definition():
    for term in TERMS.values():
        assert term.label and term.definition


def test_labels_and_tooltips_are_plain_language():
    assert label("recall") == "Leavers we catch"
    assert tooltip("recall").startswith("Of all customers who leave")
