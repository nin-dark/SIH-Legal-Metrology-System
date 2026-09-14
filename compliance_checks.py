"""Compliance-decision logic shared between worker.py's async image
pipeline (Phase 4) and main.py's synchronous listing pipeline (Phase 6).

Deliberately kept free of step1_preprocess/step2_ocr imports — those pull
in OpenCV, EasyOCR, and the full PyTorch CPU stack, which the FastAPI
process has no reason to load just to check a pasted listing's declared
fields. Everything in this module operates on already-parsed dicts, never
on raw images.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import Scan, ScanImage


def build_missing_issues(
    declarations: dict,
    font_ok: bool | None,
    required_mm: float | None,
    placement_ok: bool | None = None,
) -> list[str]:
    """Compute parent-level compliance issues after declaration merge."""
    issues = []

    if declarations["mrp"] is None:
        issues.append(
            "Rule 6(1)(e): Retail sale price (MRP) not detected"
        )

    if declarations["net_quantity"] is None:
        issues.append(
            "Rule 6(1)(c): Net quantity not detected"
        )

    if declarations["date_of_mfg"] is None:
        issues.append(
            "Rule 6(1)(d): Month/year of manufacture not detected"
        )

    if declarations["manufacturer_address"] is None:
        issues.append(
            "Rule 6(1)(a): Manufacturer/packer address not clearly detected"
        )

    if declarations["consumer_care"] is None:
        issues.append(
            "Rule 6(2): Consumer care contact details not detected"
        )

    if font_ok is False:
        mm_note = (
            f" (Rule 7 would require {required_mm}mm for this quantity, "
            "but this check does not measure physical mm)"
            if required_mm
            else ""
        )
        issues.append(
            "Readability heuristic: detected text is smaller than a "
            "general legibility threshold — a relative pixel-based "
            "estimate, not a measured Rule 7 numeral-height verdict"
            + mm_note
        )

    if placement_ok is False:
        issues.append(
            "Required declarations are not grouped together in a single "
            "area of the label"
        )

    return issues


def _normalize_quantity(value: float, unit: str) -> tuple[str | None, float | None]:
    """Convert a (value, unit) pair to a (family, base_amount) tuple.

    family is 'weight' (base=grams), 'volume' (base=ml), or 'count'
    (base=item count). Returns (None, None) for an unrecognized unit.
    """
    unit = unit.lower()

    weight = {"g": 1, "grm": 1, "gram": 1, "grams": 1, "kg": 1000}
    volume = {"ml": 1, "l": 1000, "liter": 1000, "litres": 1000}
    count = {
        "n": 1, "u": 1, "unit": 1, "units": 1,
        "pc": 1, "pcs": 1, "piece": 1, "pieces": 1,
        "stick": 1, "sticks": 1, "tablet": 1, "tablets": 1,
        "capsule": 1, "capsules": 1,
    }

    if unit in weight:
        return "weight", value * weight[unit]
    if unit in volume:
        return "volume", value * volume[unit]
    if unit in count:
        return "count", value * count[unit]

    return None, None


def _normalize_basis(basis: str) -> tuple[str | None, float | None]:
    """Convert a unit-price basis (e.g. 'kg', '100g') to (family, base_amount)."""
    basis = basis.lower()

    weight_basis = {"g": 1, "kg": 1000, "100g": 100}
    volume_basis = {"ml": 1, "l": 1000, "litre": 1000, "liter": 1000}
    count_basis = {"unit": 1, "pc": 1, "piece": 1}

    if basis in weight_basis:
        return "weight", weight_basis[basis]
    if basis in volume_basis:
        return "volume", volume_basis[basis]
    if basis in count_basis:
        return "count", count_basis[basis]

    return None, None


# Relative tolerance for the MRP-vs-unit-price cross-check. OCR digits won't
# line up to the last paisa even on a fully compliant label, so this is a
# deliberate assumption, not a legal threshold from the PS — adjust freely.
UNIT_PRICE_TOLERANCE = 0.05


def check_misleading_declarations(declarations: dict) -> list[str]:
    """Phase 4.1b: cross-check declared values against each other.

    Both checks are skipped (not flagged) when there isn't enough
    structured data to compare safely — e.g. mismatched unit families,
    or a value simply wasn't detected. Absence of a discount/unit-price
    claim is never itself a violation (see step3_parser.py comment).

    Input-source-agnostic: works identically whether `declarations` came
    from merged image OCR (Phase 4) or a single parsed listing text
    (Phase 6) — it only ever looks at the dict's values.
    """
    issues = []

    mrp = declarations.get("mrp_value")
    qty_val = declarations.get("net_quantity_value")
    qty_unit = declarations.get("net_quantity_unit")
    price_val = declarations.get("unit_price_value")
    price_basis = declarations.get("unit_price_basis")

    if None not in (mrp, qty_val, qty_unit, price_val, price_basis) and mrp > 0:
        qty_family, qty_base = _normalize_quantity(qty_val, qty_unit)
        basis_family, basis_amount = _normalize_basis(price_basis)

        if qty_family is not None and qty_family == basis_family and basis_amount:
            implied_total = price_val * (qty_base / basis_amount)
            relative_diff = abs(implied_total - mrp) / mrp

            if relative_diff > UNIT_PRICE_TOLERANCE:
                issues.append(
                    "Misleading declaration: declared unit price implies a "
                    f"total price of ~Rs.{implied_total:.2f}, which differs "
                    f"from the declared MRP of Rs.{mrp:.2f} by more than "
                    f"{int(UNIT_PRICE_TOLERANCE * 100)}%"
                )

    free_val = declarations.get("free_qty_value")
    free_unit = declarations.get("free_qty_unit")

    if None not in (free_val, free_unit, qty_val, qty_unit):
        free_family, free_base = _normalize_quantity(free_val, free_unit)
        qty_family, qty_base = _normalize_quantity(qty_val, qty_unit)

        if free_family is not None and free_family == qty_family and free_base >= qty_base:
            issues.append(
                "Misleading declaration: claimed 'free' quantity is not "
                "smaller than the total declared net quantity"
            )

    # NOTE: discount_pct_value is intentionally not cross-checked here.
    # Validating a "% off" claim needs an original/pre-discount price field
    # that doesn't exist anywhere in the current schema or parser output —
    # there's nothing to compare it against. Flagging this as a known gap
    # rather than inventing a second price field silently.

    return issues


# Relative tolerance for cross-image conflicts. Tighter than
# UNIT_PRICE_TOLERANCE because this compares the SAME printed number across
# two photos of the SAME pack (no unit-conversion arithmetic involved) — a
# real conflict should differ by more than plain OCR digit noise.
CROSS_IMAGE_TOLERANCE = 0.02


def check_cross_image_conflicts(images: list["ScanImage"]) -> list[str]:
    """Phase 4.2: flag the same declared field reading differently across
    different images of the same scan (e.g. two photos of one pack showing
    different MRPs). Operates on each image's own `declarations` JSONB,
    not the already-merged dict, since the merge only keeps the first
    non-null value and would hide a disagreement.

    Only meaningful for image scans with 2+ images — a Phase 6 listing
    scan has no images at all, so its caller simply never calls this.

    Comparisons are skipped (not flagged) when there's nothing to compare
    against, or when quantities are in genuinely incompatible unit
    families — same policy as check_misleading_declarations.
    """
    issues = []

    mrp_values = [
        value
        for image in images
        if (value := (image.declarations or {}).get("mrp_value")) is not None
    ]

    if len(mrp_values) >= 2:
        lo, hi = min(mrp_values), max(mrp_values)

        if lo > 0 and (hi - lo) / lo > CROSS_IMAGE_TOLERANCE:
            issues.append(
                "Misleading declaration: MRP reads differently across "
                f"images of the same product (Rs.{lo:.2f} vs Rs.{hi:.2f})"
            )

    qty_bases_by_family: dict[str, list[float]] = {}

    for image in images:
        declarations = image.declarations or {}
        value = declarations.get("net_quantity_value")
        unit = declarations.get("net_quantity_unit")

        if value is None or unit is None:
            continue

        family, base = _normalize_quantity(value, unit)

        if family is None:
            continue

        qty_bases_by_family.setdefault(family, []).append(base)

    for family, bases in qty_bases_by_family.items():
        if len(bases) < 2:
            continue

        lo, hi = min(bases), max(bases)

        if lo > 0 and (hi - lo) / lo > CROSS_IMAGE_TOLERANCE:
            issues.append(
                "Misleading declaration: net quantity reads differently "
                f"across images of the same product ({lo:g} vs {hi:g}, "
                f"{family} terms)"
            )

    return issues


def apply_compliance_verdict(
    scan: "Scan",
    declarations: dict,
    severity_issues: list[str],
) -> None:
    """Shared status/issues decision, used by both pipelines.

    Caller must set scan.font_ok, scan.required_mm, scan.placement_ok,
    and the declared-field columns (mrp, net_quantity, etc.) BEFORE
    calling this — how those get produced differs between the two
    pipelines (aggregated across sibling images vs a single text source),
    but once they're set on `scan`, the issue-building and status
    decision are identical either way.

    `severity_issues` is whatever the caller has already decided counts
    as VIOLATION-tier (misleading-declaration checks, cross-image
    conflicts for the image pipeline) — this function doesn't compute
    those itself, since which checks apply differs by pipeline (a
    listing scan has no cross-image check to run, for instance).
    """
    issues = build_missing_issues(
        declarations,
        scan.font_ok,
        scan.required_mm,
        scan.placement_ok,
    )
    issues = issues + severity_issues

    scan.issues = issues

    if (
        declarations["mrp"] is None
        or declarations["net_quantity"] is None
        or severity_issues
    ):
        scan.status = "VIOLATION"
    elif issues:
        scan.status = "WARNING"
    else:
        scan.status = "COMPLIANT"