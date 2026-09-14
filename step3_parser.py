import re

def _reconstruct_reading_order(raw_lines, row_tolerance=15):
    items = []
    for item in raw_lines:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            y_center = sum(pt[1] for pt in bbox) / 4
            x_center = sum(pt[0] for pt in bbox) / 4
            items.append((y_center, x_center, item["text"]))
        else:
            items.append((0, 0, item["text"]))
    items.sort(key=lambda t: t[0])
    rows, current_row, last_y = [], [], None
    for y, x, text in items:
        if last_y is not None and abs(y - last_y) > row_tolerance:
            rows.append(current_row); current_row = []
        current_row.append((x, text)); last_y = y
    if current_row: rows.append(current_row)
    return " ".join(" ".join(t for _, t in sorted(row, key=lambda r: r[0])) for row in rows)


# Delimiters restricted to "/" and "-" only (NOT ".") — a period is heavily
# overloaded with decimal prices/weights (e.g. "0.20") on real labels, and
# including it caused false-positive date matches during testing.
DATE_VALUE = r"\d{1,2}[-/]\s?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/]\s?\d{2,4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{2,4}"


def parse_declarations_from_text(full_text: str) -> dict:
    """The reusable half of the parser: everything that only needs plain
    text, no bounding boxes. This is what Phase 6 (listing/text input mode)
    calls directly on a pasted product listing, skipping bbox
    reconstruction, font-height measurement, and placement entirely — none
    of that applies to text with no photo behind it.

    Returns declarations, required_mm, and the Rule 6-family
    declaration-completeness issues (missing MRP/qty/date/address/care).
    Does NOT include the readability (font) issue — that depends on
    bounding-box heights from an actual photo and is added on top by
    parse_legal_metrology_declarations() below, which is the only caller
    that has raw_lines to measure.
    """
    mrp_pattern = r"(?:MRP|M\.R\.P\.|Max(?:imum)?\s*Retail\s*Price)[^\d]{0,35}?(\d+(?:\.\d{1,2})?)"
    net_wt_pattern = r"(?:Net\s*Wt|Net\s*Qty|Net\s*Quantity|Net\s*Weight|Net\s*Volume|Net\s*Content|NET\s*WT\.?|Contents)[^\d]{0,15}?(\d+(?:\.\d+)?)\s*(g|grm|gram|grams|kg|ml|l|liter|litres|N|U|units|sticks?|pcs|pieces?|tablets?|capsules?)"

    # Restricted to labels that explicitly say "per <unit>" / "/<unit>" —
    # a bare second price on the label is too ambiguous to assume is
    # the unit price.
    unit_price_pattern = r"(?:Unit\s*Price|Price\s*per|Rs\.?|₹|INR)[^\d]{0,20}?(\d+(?:\.\d{1,2})?)\s*(?:per|/)\s*(kg|g|100\s*g|ml|l|litre|liter|unit|pc|piece)"

    discount_pct_pattern = r"(\d+(?:\.\d+)?)\s*%\s*(?:off|discount)"
    free_qty_pattern = r"(\d+(?:\.\d+)?)\s*(g|grm|gram|grams|kg|ml|l|liter|litres|N|U|units?|pcs?|pieces?)\s*free"

    address_pattern = (
        r"(?:Manufactured\s*by|Mfd\s*by|Mfg\s*by|Packed\s*by|"
        r"Pkd\s*by|Imported\s*by|Imp\s*by|Marketed\s*by|"
        r"Mkt\s*by|Mktd\s*by|Address)"
        r"\s*[:\.-]?\s*"
        r"[A-Za-z0-9,.\-/\s]{10,}?"
        r"(?=\s+(?:Consumer\s*Care|Customer\s*Care|Helpline|"
        r"Toll[-\s]?Free|Contact\s*Us|MRP|M\.R\.P\.|"
        r"Net\s*(?:Qty|Quantity|Wt|Weight)|"
        r"(?:Date\s*of\s*Manufacture|Mfg\.?\s*Date|MFD|PKD))\b|$)"
        r"|\b\d{6}\b"
    )

    consumer_care_pattern = (
        r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        r"|\d{10}|1800\s*\d{3}\s*\d{3,4})"
    )

    mrp_match = re.search(mrp_pattern, full_text, re.IGNORECASE)
    net_wt_match = re.search(net_wt_pattern, full_text, re.IGNORECASE)
    address_match = re.search(address_pattern, full_text, re.IGNORECASE)
    care_match = re.search(consumer_care_pattern, full_text, re.IGNORECASE)
    unit_price_match = re.search(unit_price_pattern, full_text, re.IGNORECASE)
    discount_match = re.search(discount_pct_pattern, full_text, re.IGNORECASE)
    free_qty_match = re.search(free_qty_pattern, full_text, re.IGNORECASE)

    mfg_keyword_pattern = rf"(?:Date\s*of\s*Manufactur(?:e|ing)|Date\s*of\s*Import|Imp\.?\s*Date|Mfg\.?\s*Date|Mfg|Packed|Pkg|MFD|PKD)[^\d]{{0,20}}?({DATE_VALUE})"
    date_match = re.search(mfg_keyword_pattern, full_text, re.IGNORECASE)

    date_value = None
    if date_match:
        date_value = date_match.group(1)
    else:
        all_dates = [
            m.group(0)
            for m in re.finditer(DATE_VALUE, full_text, re.IGNORECASE)
        ]
        if all_dates:
            date_value = f"{all_dates[0]} (auto-detected — verify placement)"

    required_mm = None
    if net_wt_match:
        try:
            qty_val = float(net_wt_match.group(1))
            unit = net_wt_match.group(2).lower()
            grams_equiv = (
                qty_val * 1000
                if unit in ("kg", "l", "liter", "litres")
                else qty_val
            )
            required_mm = (
                1 if grams_equiv <= 200
                else 2 if grams_equiv <= 500
                else 4
            )
        except (ValueError, IndexError):
            required_mm = None

    declarations = {
        "mrp": mrp_match.group(0) if mrp_match else None,
        "net_quantity": net_wt_match.group(0) if net_wt_match else None,
        "date_of_mfg": date_value,
        "manufacturer_address": (
            address_match.group(0) if address_match else None
        ),
        "consumer_care": care_match.group(0) if care_match else None,
        "unit_price": (
            unit_price_match.group(1) if unit_price_match else None
        ),
        "discount_claim": (
            f"{discount_match.group(1)}%"
            if discount_match
            else None
        ),
        "free_qty_claim": (
            f"{free_qty_match.group(1)}{free_qty_match.group(2)}"
            if free_qty_match
            else None
        ),
        "mrp_value": float(mrp_match.group(1)) if mrp_match else None,
        "net_quantity_value": (
            float(net_wt_match.group(1))
            if net_wt_match
            else None
        ),
        "net_quantity_unit": (
            net_wt_match.group(2).lower()
            if net_wt_match
            else None
        ),
        "unit_price_value": (
            float(unit_price_match.group(1))
            if unit_price_match
            else None
        ),
        "unit_price_basis": (
            unit_price_match.group(2).lower().replace(" ", "")
            if unit_price_match
            else None
        ),
        "discount_pct_value": (
            float(discount_match.group(1))
            if discount_match
            else None
        ),
        "free_qty_value": (
            float(free_qty_match.group(1))
            if free_qty_match
            else None
        ),
        "free_qty_unit": (
            free_qty_match.group(2).lower()
            if free_qty_match
            else None
        ),
    }

    issues = []

    if not declarations["mrp"]:
        issues.append("Rule 6(1)(e): Retail sale price (MRP) not detected")

    if not declarations["net_quantity"]:
        issues.append("Rule 6(1)(c): Net quantity not detected")

    if not date_value:
        issues.append("Rule 6(1)(d): Month/year of manufacture not detected")

    if not declarations["manufacturer_address"]:
        issues.append(
            "Rule 6(1)(a): Manufacturer/packer address not clearly detected"
        )

    if not declarations["consumer_care"]:
        issues.append(
            "Rule 6(2): Consumer care contact details not detected"
        )

    return {
        "declarations": declarations,
        "required_mm": required_mm,
        "issues": issues,
        "raw_text": full_text,
    }

def parse_legal_metrology_declarations(raw_lines):
    full_text = _reconstruct_reading_order(raw_lines)

    texts = [str(item.get("text", "")).strip() for item in raw_lines]
    def _nearby_value(index, max_distance=80):
        """
        Return OCR boxes that are physically close to the target box.

        PaddleOCR does not guarantee that raw result order matches visual
        proximity, so declaration labels and values are matched spatially.
        """
        target_bbox = raw_lines[index].get("bbox", [])
        if len(target_bbox) != 4:
            return []

        tx = sum(pt[0] for pt in target_bbox) / 4
        ty = sum(pt[1] for pt in target_bbox) / 4

        candidates = []
        for j, item in enumerate(raw_lines):
            if j == index:
                continue

            text = str(item.get("text", "")).strip()
            bbox = item.get("bbox", [])
            if not text or len(bbox) != 4:
                continue

            x = sum(pt[0] for pt in bbox) / 4
            y = sum(pt[1] for pt in bbox) / 4
            dx = abs(x - tx)
            dy = abs(y - ty)

            if dx <= max_distance and dy <= max_distance:
                candidates.append((dx + dy, text))

        candidates.sort(key=lambda item: item[0])
        return [text for _, text in candidates]

    # OCR frequently splits declaration labels and their values into
    # separate boxes, e.g. "NET QUANTITY:" followed by "70g".
    quantity_value_pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*"
        r"(g|grm|gram|grams|kg|ml|l|liter|litres|"
        r"N|U|units?|sticks?|pcs?|pieces?|tablets?|capsules?)\b",
        re.IGNORECASE,
    )

    split_net_match = None
    for i, text in enumerate(texts):
        if not re.search(
            r"\bnet\s*(?:qty|quantity|wt|weight|content|volume)\b",
            text,
            re.IGNORECASE,
        ):
            continue

        for candidate in [text] + _nearby_value(i):
            m = quantity_value_pattern.search(candidate)
            if m:
                split_net_match = m
                break

        if split_net_match:
            break

    split_mrp_match = None
    for i, text in enumerate(texts):
        if not re.search(
            r"\b(?:MRP|M\.R\.P\.|retail\s*price|max(?:imum)?\s*retail\s*price)\b",
            text,
            re.IGNORECASE,
        ):
            continue

        for candidate in [text] + _nearby_value(i):
            if re.search(r"\blic\.?\s*no\b", candidate, re.IGNORECASE):
                continue
            if re.search(r"1800\s*\d{3}\s*\d{3,4}", candidate):
                continue

            m = re.search(
                r"(?:₹|Rs\.?|INR)\s*(\d+(?:\.\d{1,2})?)\b",
                candidate,
                re.IGNORECASE,
            )
            if not m:
                continue

            value = m.group(1)
            try:
                numeric_value = float(value)
            except ValueError:
                continue

            if 0 < numeric_value < 100000:
                split_mrp_match = m
                break

        if split_mrp_match:
            break

    mrp_pattern = r"(?:MRP|M\.R\.P\.|Max(?:imum)?\s*Retail\s*Price)[^\d]{0,35}?(\d+(?:\.\d{1,2})?)"
    net_wt_pattern = r"(?:Net\s*Wt|Net\s*Qty|Net\s*Quantity|Net\s*Weight|Net\s*Volume|Net\s*Content|NET\s*WT\.?|Contents)[^\d]{0,15}?(\d+(?:\.\d+)?)\s*(g|grm|gram|grams|kg|ml|l|liter|litres|N|U|units|sticks?|pcs|pieces?|tablets?|capsules?)"
    # Restricted to labels that explicitly say "per <unit>" / "/<unit>" — a bare
    # second price on the label is too ambiguous to assume is the unit price.
    unit_price_pattern = r"(?:Unit\s*Price|Price\s*per|Rs\.?|₹|INR)[^\d]{0,20}?(\d+(?:\.\d{1,2})?)\s*(?:per|/)\s*(kg|g|100\s*g|ml|l|litre|liter|unit|pc|piece)"
    discount_pct_pattern = r"(\d+(?:\.\d+)?)\s*%\s*(?:off|discount)"
    free_qty_pattern = r"(\d+(?:\.\d+)?)\s*(g|grm|gram|grams|kg|ml|l|liter|litres|N|U|units?|pcs?|pieces?)\s*free"
    address_pattern = (
        r"(?:Manufactured\s*by|Mfd\s*by|Mfg\s*by|Packed\s*by|"
        r"Pkd\s*by|Imported\s*by|Imp\s*by|Marketed\s*by|"
        r"Mkt\s*by|Mktd\s*by|Address)"
        r"\s*[:\.-]?\s*"
        r"[A-Za-z0-9,.\-/\s]{10,}"
    )
    consumer_care_pattern = (
        r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        r"|1800\s*\d{3}\s*\d{3,4})"
    )

    mrp_match = None

    # Prefer OCR lines containing the MRP label and look for a nearby price.
    # This prevents unrelated numbers such as FSSAI licence numbers from
    # being captured as the MRP.
    for item in raw_lines:
        text = item.get("text", "")
        if re.search(r"(?:MRP|M\.R\.P\.|Max(?:imum)?\s*Retail\s*Price)", text, re.IGNORECASE):
            candidate = re.search(
                r"(?:MRP|M\.R\.P\.|Max(?:imum)?\s*Retail\s*Price)"
                r"[^\d]{0,12}(\d+(?:\.\d{1,2})?)",
                text,
                re.IGNORECASE,
            )
            if candidate:
                mrp_match = candidate
                break

    # Fallback to the reconstructed OCR text, but with a much tighter window.
    if mrp_match is None:
        mrp_match = re.search(
            r"(?:MRP|M\.R\.P\.|Max(?:imum)?\s*Retail\s*Price)"
            r"[^\d]{0,12}(\d+(?:\.\d{1,2})?)",
            full_text,
            re.IGNORECASE,
        )
    net_wt_match = re.search(net_wt_pattern, full_text, re.IGNORECASE)

    # Prefer the structured OCR-box match when the normal reconstructed-text
    # regex cannot connect the label to its value.
    if net_wt_match is None and split_net_match is not None:
        net_wt_match = split_net_match

    # Likewise, use a nearby OCR-box MRP match only when it is plausible.
    if mrp_match is None and split_mrp_match is not None:
        split_value = split_mrp_match.group(1)
        if split_value:
            try:
                if 0 < float(split_value) < 100000:
                    mrp_match = split_mrp_match
            except ValueError:
                pass

    # Prefer an explicit manufacturer/packer/importer/address label.
    address_match = re.search(
        address_pattern,
        full_text,
        re.IGNORECASE,
    )

    # Prefer contact information occurring near a consumer-care label.
    care_match = None
    for item in raw_lines:
        item_text = str(item.get("text", ""))
        if re.search(
            r"(?:consumer\s*care|customer\s*care|helpline|"
            r"toll[-\s]?free|contact\s*us)",
            item_text,
            re.IGNORECASE,
        ):
            care_match = re.search(
                consumer_care_pattern,
                item_text,
                re.IGNORECASE,
            )
            if care_match:
                break

    # Some labels put the phone number in the OCR box immediately before
    # or after the consumer-care text. Check nearby OCR boxes as a fallback.
    if care_match is None:
        for index, item in enumerate(raw_lines):
            item_text = str(item.get("text", ""))
            if not re.search(
                r"(?:consumer\s*care|customer\s*care|helpline|"
                r"toll[-\s]?free|contact\s*us)",
                item_text,
                re.IGNORECASE,
            ):
                continue

            for nearby_index in (index - 1, index + 1):
                if nearby_index < 0 or nearby_index >= len(raw_lines):
                    continue

                nearby_text = str(raw_lines[nearby_index].get("text", ""))
                care_match = re.search(
                    consumer_care_pattern,
                    nearby_text,
                    re.IGNORECASE,
                )

                if care_match:
                    break

            if care_match:
                break

    # Final email/toll-free fallback only. Do not accept arbitrary 10-digit
    # numbers because FSSAI/licence/batch numbers are common on packages.
    if care_match is None:
        care_match = re.search(
            consumer_care_pattern,
            full_text,
            re.IGNORECASE,
        )
    unit_price_match = re.search(unit_price_pattern, full_text, re.IGNORECASE)
    discount_match = re.search(discount_pct_pattern, full_text, re.IGNORECASE)
    free_qty_match = re.search(free_qty_pattern, full_text, re.IGNORECASE)

    mfg_keyword_pattern = rf"(?:Date\s*of\s*Manufactur(?:e|ing)|Date\s*of\s*Import|Imp\.?\s*Date|Mfg\.?\s*Date|Mfg|Packed|Pkg|MFD|PKD)[^\d]{{0,20}}?({DATE_VALUE})"
    date_match = re.search(mfg_keyword_pattern, full_text, re.IGNORECASE)

    date_value = None
    if date_match:
        date_value = date_match.group(1)
    else:
        all_dates = [m.group(0) for m in re.finditer(DATE_VALUE, full_text, re.IGNORECASE)]
        if all_dates:
            date_value = f"{all_dates[0]} (auto-detected — verify placement)"

    required_mm = None
    if net_wt_match:
        try:
            qty_val = float(net_wt_match.group(1))
            unit = net_wt_match.group(2).lower()
            grams_equiv = qty_val * 1000 if unit in ("kg", "l", "liter", "litres") else qty_val
            required_mm = 1 if grams_equiv <= 200 else 2 if grams_equiv <= 500 else 4
        except (ValueError, IndexError):
            required_mm = None

    heights = []
    for item in raw_lines:
        bbox = item.get("bbox", [])
        if len(bbox) == 4:
            ys = [pt[1] for pt in bbox]
            heights.append(max(ys) - min(ys))
    median_height_px = sorted(heights)[len(heights) // 2] if heights else 0
    font_ok = median_height_px >= 14

    mrp_numeric_value = None
    if mrp_match:
        for group_index in range(1, (mrp_match.lastindex or 0) + 1):
            value = mrp_match.group(group_index)
            if value:
                try:
                    mrp_numeric_value = float(value)
                    break
                except ValueError:
                    pass

    declarations = {
        "mrp": mrp_match.group(0) if mrp_match else None,
        "net_quantity": net_wt_match.group(0) if net_wt_match else None,
        "date_of_mfg": date_value,
        "manufacturer_address": address_match.group(0) if address_match else None,
        "consumer_care": care_match.group(0) if care_match else None,
        # These three are inputs to the Phase 4.1b misleading-declaration
        # checks, not required declarations in their own right — a label
        # with no discount/free claim isn't non-compliant, so none of these
        # feed into `issues` below.
        "unit_price": unit_price_match.group(1) if unit_price_match else None,
        "discount_claim": f"{discount_match.group(1)}%" if discount_match else None,
        "free_qty_claim": f"{free_qty_match.group(1)}{free_qty_match.group(2)}" if free_qty_match else None,
        # Clean numeric forms for the Phase 4.1b comparison checks in
        # worker.py — the fields above are display strings, these are for
        # arithmetic. All None when the corresponding text wasn't found.
        "mrp_value": mrp_numeric_value,
        "net_quantity_value": float(net_wt_match.group(1)) if net_wt_match else None,
        "net_quantity_unit": net_wt_match.group(2).lower() if net_wt_match else None,
        "unit_price_value": float(unit_price_match.group(1)) if unit_price_match else None,
        "unit_price_basis": unit_price_match.group(2).lower().replace(" ", "") if unit_price_match else None,
        "discount_pct_value": float(discount_match.group(1)) if discount_match else None,
        "free_qty_value": float(free_qty_match.group(1)) if free_qty_match else None,
        "free_qty_unit": free_qty_match.group(2).lower() if free_qty_match else None,
    }

    issues = []
    if not declarations["mrp"]:
        issues.append("Rule 6(1)(e): Retail sale price (MRP) not detected")
    if not declarations["net_quantity"]:
        issues.append("Rule 6(1)(c): Net quantity not detected")
    if not date_value:
        issues.append("Rule 6(1)(d): Month/year of manufacture not detected")
    if not declarations["manufacturer_address"]:
        issues.append("Rule 6(1)(a): Manufacturer/packer address not clearly detected")
    if not declarations["consumer_care"]:
        issues.append("Rule 6(2): Consumer care contact details not detected")
    if not font_ok:
        mm_note = f" (Rule 7 would require {required_mm}mm for this quantity, but this check does not measure physical mm)" if required_mm else ""
        issues.append(f"Readability heuristic: detected text is smaller than a general legibility threshold — a relative pixel-based estimate, not a measured Rule 7 numeral-height verdict{mm_note}")

    return {
        "declarations": declarations, "font_ok": font_ok, "required_mm": required_mm,
        "median_text_height_px": median_height_px, "issues": issues, "raw_text": full_text,
    }
