"""Tests for the DipVerdict schema and judge prompt rendering."""

import pytest
from pydantic import ValidationError

from src.dip.detection import DipCandidate
from src.dip.judge import DipVerdict, build_dip_prompt


CANDIDATE = DipCandidate(ticker="NKE", last_price=93.0, move_pct=-7.0, spy_move_pct=-0.2, excess_move_pct=-6.8, rel_volume=3.1, drawdown_pct=-15.5)
MATH_PACKET = {"fundamentals": {"signal": "bullish", "confidence": 67}, "valuation": {"signal": "bullish", "metrics": {"valuation_gap_pct": 22.0}}, "growth": {"error": "unavailable: no data"}}
HEADLINES = [{"date": "2026-06-11", "source": "Reuters", "title": "Nike falls after analyst downgrade on China inventory"}]


def test_verdict_validates_good_payload():
    v = DipVerdict(classification="transitory", confidence=78, event_summary="Analyst downgrade", reasoning="Inventory issues look cyclical", suggested_action="buy_dip", key_risk="China demand is structural", is_earnings_related=False)
    assert v.classification == "transitory"


def test_verdict_rejects_unknown_classification_and_out_of_range_confidence():
    with pytest.raises(ValidationError):
        DipVerdict(classification="maybe", confidence=50, event_summary="x", reasoning="x", suggested_action="avoid", key_risk="x", is_earnings_related=False)
    with pytest.raises(ValidationError):
        DipVerdict(classification="unclear", confidence=150, event_summary="x", reasoning="x", suggested_action="avoid", key_risk="x", is_earnings_related=False)


def test_prompt_contains_stats_packet_headlines_and_rubric():
    prompt = build_dip_prompt(CANDIDATE, MATH_PACKET, HEADLINES)
    assert "NKE" in prompt
    assert "-7.0%" in prompt  # today's move
    assert "-6.8%" in prompt  # excess move
    assert "3.1x" in prompt  # relative volume
    assert "valuation_gap_pct" in prompt  # math packet serialized
    assert "pre-drop context" in prompt.lower()
    assert "Nike falls after analyst downgrade" in prompt
    assert "titles only" in prompt.lower()
    assert "web" in prompt.lower()  # research instruction
    assert "transitory" in prompt and "thesis" in prompt  # rubric present


def test_prompt_handles_no_headlines():
    prompt = build_dip_prompt(CANDIDATE, MATH_PACKET, [])
    assert "no recent headlines found" in prompt.lower()
