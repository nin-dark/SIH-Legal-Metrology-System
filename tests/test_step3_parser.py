"""
Offline unit tests for step3_parser.py.

These run against synthetic OCR line data (no Docker, no PaddleOCR, no model
downloads) so they're fast enough to run on every change. They specifically
cover the two real bugs found and fixed in this file:

  1. The doubled-backslash regex escapes (\\b, \\d, \\s instead of \b, \d, \s)
     that silently made several patterns unmatchable.
  2. The `dx <= max_distance or dy <= max_distance` bug in _nearby_value,
     which let a text box far away on one axis but coincidentally aligned on
     the other axis (e.g. same x-column, very different y-row) count as
     "nearby" — this produced a real false-positive MRP match against an
     address line on the actual Maggi benchmark image.

Run with:
    python -m pytest tests/test_step3_parser.py -v
"""

from step3_parser import parse_legal_metrology_declarations


def box(text, x, y, w=100, h=30, confidence=0.9):
    """Build one synthetic OCR line dict with a bbox centered at (x, y)."""
    return {
        "text": text,
        "confidence": confidence,
        "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
    }


# ---------------------------------------------------------------------------
# Net quantity — split across two OCR boxes (the original bug report)
# ---------------------------------------------------------------------------

def test_net_quantity_split_across_two_close_boxes():
    raw_lines = [
        box("NET QUANTITY:", x=100, y=500),
        box("70g", x=230, y=502),  # same row, just to the right
    ]
    result = parse_legal_metrology_declarations(raw_lines)
    d = result["declarations"]

    assert d["net_quantity_value"] == 70.0
    assert d["net_quantity_unit"] == "g"
    assert "70g" in d["net_quantity"]


def test_net_quantity_absent_when_no_label_present():
    raw_lines = [box("SOME UNRELATED TEXT", x=50, y=50)]
    result = parse_legal_metrology_declarations(raw_lines)
    d = result["declarations"]

    assert d["net_quantity"] is None
    assert d["net_quantity_value"] is None
    assert d["net_quantity_unit"] is None
    assert "Rule 6(1)(c): Net quantity not detected" in result["issues"]


# ---------------------------------------------------------------------------
# MRP — regression test for the real false positive found on the Maggi label
# ---------------------------------------------------------------------------

def test_mrp_not_falsely_matched_to_distant_same_column_text():
    """
    Regression test for the real bug: an address line ("P.O. BAG NO.2,
    NEW DELHI-110001") sat in roughly the same x-column as the MRP label but
    ~350px away vertically. The old `or` condition in _nearby_value treated
    that as "nearby" and pulled the "2" out of the address as if it were the
    MRP. The fix requires both axes to be close (`and`), so this must now
    correctly return no MRP.
    """
    raw_lines = [
        box("P.O. BAG NO.2, NEW DELHI-110001", x=100, y=50),
        box("MRP (incl. of all taxes)", x=100, y=400),  # same x, far y
    ]
    result = parse_legal_metrology_declarations(raw_lines)
    d = result["declarations"]

    assert d["mrp"] is None
    assert d["mrp_value"] is None


def test_mrp_detected_when_genuinely_close_with_currency_symbol():
    raw_lines = [
        box("MRP (incl. of all taxes)", x=100, y=400),
        box("₹100", x=100, y=410),  # same column AND close vertically
    ]
    result = parse_legal_metrology_declarations(raw_lines)
    d = result["declarations"]

    assert d["mrp_value"] == 100.0


def test_mrp_ignores_nearby_license_number():
    raw_lines = [
        box("MRP (incl. of all taxes)", x=100, y=400),
        box("Lic. No. 10012063000064", x=100, y=410),
    ]
    result = parse_legal_metrology_declarations(raw_lines)
    d = result["declarations"]

    assert d["mrp"] is None
    assert d["mrp_value"] is None


def test_mrp_ignores_nearby_toll_free_number():
    raw_lines = [
        box("MRP (incl. of all taxes)", x=100, y=400),
        box("1800 103 1947", x=100, y=410),
    ]
    result = parse_legal_metrology_declarations(raw_lines)
    d = result["declarations"]

    assert d["mrp"] is None
    assert d["mrp_value"] is None


def test_mrp_absent_when_label_has_no_nearby_value_at_all():
    raw_lines = [box("MRP (incl. of all taxes)", x=100, y=400)]
    result = parse_legal_metrology_declarations(raw_lines)
    d = result["declarations"]

    assert d["mrp"] is None
    assert d["mrp_value"] is None
    assert "Rule 6(1)(e): Retail sale price (MRP) not detected" in result["issues"]


# ---------------------------------------------------------------------------
# Consumer care
# ---------------------------------------------------------------------------

def test_consumer_care_detected_in_same_box_as_label():
    raw_lines = [box("Consumer Care: 18001031947", x=50, y=600)]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["declarations"]["consumer_care"] == "18001031947"


def test_consumer_care_detected_via_nearby_box_fallback():
    raw_lines = [
        box("NESTLÉ CONSUMER CARE", x=50, y=600),
        box("18001031947", x=50, y=620),
    ]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["declarations"]["consumer_care"] == "18001031947"


def test_consumer_care_absent_does_not_hallucinate_a_license_number():
    raw_lines = [box("Lic. No. 10012063000064", x=50, y=600)]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["declarations"]["consumer_care"] is None
    assert "Rule 6(2): Consumer care contact details not detected" in result["issues"]


# ---------------------------------------------------------------------------
# Address and date — must not be hallucinated without an explicit label
# ---------------------------------------------------------------------------

def test_address_not_inferred_without_explicit_label():
    raw_lines = [box("P.O. BAG NO.2, NEW DELHI-110001", x=100, y=50)]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["declarations"]["manufacturer_address"] is None


def test_address_detected_with_explicit_label():
    raw_lines = [box("Manufactured by: Example Foods Pvt Ltd, Delhi 110001", x=100, y=50)]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["declarations"]["manufacturer_address"] is not None


def test_date_not_detected_without_any_date_like_text():
    raw_lines = [box("LMFD.-USEBY-MFG.By:", x=100, y=300)]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["declarations"]["date_of_mfg"] is None


def test_date_detected_with_keyword_and_value():
    raw_lines = [box("MFG DATE 12/2024", x=100, y=300)]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["declarations"]["date_of_mfg"] is not None
    assert "12/2024" in result["declarations"]["date_of_mfg"]


# ---------------------------------------------------------------------------
# Realistic end-to-end shape: approximates the real Maggi label layout
# without needing Docker/PaddleOCR — address top, MRP label with no value
# mid-page, net quantity bottom, consumer care top. Mirrors the actual
# benchmark result we validated manually.
# ---------------------------------------------------------------------------

def test_realistic_label_layout_matches_manually_verified_benchmark():
    raw_lines = [
        box("P.O. BAG NO.2, NEW DELHI-110001", x=50, y=50),
        box("NESTLÉ CONSUMER CARE", x=50, y=90),
        box("18001031947", x=50, y=110),
        box("MRP (incl. of all taxes)", x=300, y=400),
        box("Lic. No. 10012063000064", x=50, y=450),
        box("NET QUANTITY:", x=50, y=900),
        box("70g", x=180, y=902),
    ]
    result = parse_legal_metrology_declarations(raw_lines)
    d = result["declarations"]

    # Matches the manually-verified Docker run against the real image:
    assert d["mrp"] is None
    assert d["mrp_value"] is None
    assert d["net_quantity_value"] == 70.0
    assert d["net_quantity_unit"] == "g"
    assert d["consumer_care"] == "18001031947"
    assert d["manufacturer_address"] is None
    assert d["date_of_mfg"] is None

    assert "Rule 6(1)(e): Retail sale price (MRP) not detected" in result["issues"]
    assert "Rule 6(1)(a): Manufacturer/packer address not clearly detected" in result["issues"]
    assert "Rule 6(1)(d): Month/year of manufacture not detected" in result["issues"]
    assert "Rule 6(1)(c): Net quantity not detected" not in result["issues"]
    assert "Rule 6(2): Consumer care contact details not detected" not in result["issues"]


# ---------------------------------------------------------------------------
# Readability heuristic (font_ok / required_mm)
# ---------------------------------------------------------------------------

def test_font_ok_true_for_tall_text_boxes():
    raw_lines = [box("NET QUANTITY:", x=50, y=500, h=40), box("70g", x=200, y=500, h=40)]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["font_ok"] is True


def test_font_ok_false_for_short_text_boxes():
    raw_lines = [box("NET QUANTITY:", x=50, y=500, h=5), box("70g", x=200, y=500, h=5)]
    result = parse_legal_metrology_declarations(raw_lines)
    assert result["font_ok"] is False
    assert any("Readability heuristic" in issue for issue in result["issues"])


def test_required_mm_scales_with_declared_quantity():
    small = [box("NET QUANTITY:", x=50, y=500), box("70g", x=200, y=500)]
    large = [box("NET QUANTITY:", x=50, y=500), box("2kg", x=200, y=500)]

    small_result = parse_legal_metrology_declarations(small)
    large_result = parse_legal_metrology_declarations(large)

    assert small_result["required_mm"] == 1
    assert large_result["required_mm"] == 4