from pathlib import Path

import pytest

from llms.labels import load_labels, normalize_label

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def labels():
    return load_labels(ROOT / "configs/labels.yaml")


def test_label_file_has_77_unique_names(labels):
    assert len(labels) == 77 and len(set(labels.names)) == 77
    assert labels.names[51] == "refund_not_showing_up" and labels.names[53] == "reverted_card_payment"


@pytest.mark.parametrize("raw, expected", [
    ("card_arrival", "card_arrival"),
    ("  `Card_Arrival`. ", "card_arrival"),
    ("card arrival", "card_arrival"),
    ("Refund_not_showing_up", "refund_not_showing_up"),
    ("reverted_card_payment?", "reverted_card_payment"),
    ("Rationale: The card has not come.\nLabel: card_arrival", "card_arrival"),
])
def test_parse_valid(labels, raw, expected):
    assert labels.parse(raw) == labels.index(expected)


@pytest.mark.parametrize("raw", ["", "card", "card_arrival or card_delivery_estimate",
                                 "card_arrival\ncard_linking", "I think it's card_arrival", None])
def test_parse_invalid(labels, raw):
    assert labels.parse(raw) is None


def test_normalize_label():
    assert normalize_label('"Top-up failed".') == "top_up_failed"
