"""Phase 2 scan worker."""

import os
import tempfile
import time
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from compliance_checks import (
    apply_compliance_verdict,
    check_cross_image_conflicts,
    check_misleading_declarations,
)
from db import SessionLocal
from models import Scan, ScanImage
from step1_preprocess import preprocess_image
from step2_ocr import extract_text_from_image
from step3_parser import parse_legal_metrology_declarations
from step4_placement import check_placement
from storage import download_image


PROCESSING_TIMEOUT = timedelta(minutes=15)
POLL_INTERVAL_SECONDS = 2


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def recover_stale_processing() -> None:
    """Return processing images older than the timeout to pending."""
    cutoff = now_utc() - PROCESSING_TIMEOUT

    session = SessionLocal()

    try:
        stale_images = session.execute(
            select(ScanImage).where(
                ScanImage.status == "processing",
                ScanImage.started_at.is_not(None),
                ScanImage.started_at < cutoff,
            )
        ).scalars().all()

        for image in stale_images:
            image.status = "pending"
            image.started_at = None

        session.commit()
    finally:
        session.close()


def claim_image():
    """Atomically claim one pending image."""
    session = SessionLocal()

    try:
        image = session.execute(
            select(ScanImage)
            .where(ScanImage.status == "pending")
            .order_by(ScanImage.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        ).scalar_one_or_none()

        if image is None:
            session.rollback()
            return None

        image.status = "processing"
        image.started_at = now_utc()

        session.commit()

        return image.id
    finally:
        session.close()


def merge_declarations(images: list[ScanImage]) -> dict:
    """Take the first detected value for each declaration field."""
    fields = (
        "mrp",
        "net_quantity",
        "unit_price",
        "discount_claim",
        "free_qty_claim",
        "date_of_mfg",
        "manufacturer_address",
        "consumer_care",
        "mrp_value",
        "net_quantity_value",
        "net_quantity_unit",
        "unit_price_value",
        "unit_price_basis",
        "discount_pct_value",
        "free_qty_value",
        "free_qty_unit",
    )

    merged = {field: None for field in fields}

    for field in fields:
        for image in images:
            declarations = image.declarations or {}
            value = declarations.get(field)

            if value is not None:
                merged[field] = value
                break

    return merged


def finalize_scan(
    session,
    scan: Scan,
    images: list[ScanImage],
) -> None:
    """Finalize the parent scan once all sibling images are terminal."""
    if any(image.status == "failed" for image in images):
        scan.status = "failed"
        return

    if any(image.status != "done" for image in images):
        return

    declarations = merge_declarations(images)

    scan.mrp = declarations["mrp"]
    scan.net_quantity = declarations["net_quantity"]
    scan.unit_price = declarations["unit_price"]
    scan.discount_claim = declarations["discount_claim"]
    scan.free_qty_claim = declarations["free_qty_claim"]
    scan.date_of_mfg = declarations["date_of_mfg"]
    scan.manufacturer_address = declarations["manufacturer_address"]
    scan.consumer_care = declarations["consumer_care"]

    scan.raw_text = "\n\n".join(
        image.raw_text
        for image in images
        if image.raw_text
    )

    font_values = [
        image.font_ok
        for image in images
        if image.font_ok is not None
    ]

    if any(value is False for value in font_values):
        scan.font_ok = False
    elif any(value is True for value in font_values):
        scan.font_ok = True
    else:
        scan.font_ok = None

    required_mm_values = [
        image.required_mm
        for image in images
        if image.required_mm is not None
    ]

    scan.required_mm = (
        required_mm_values[0]
        if required_mm_values
        else None
    )

    placement_values = [
        image.placement_ok
        for image in images
        if image.placement_ok is not None
    ]

    if any(value is False for value in placement_values):
        scan.placement_ok = False
    elif any(value is True for value in placement_values):
        scan.placement_ok = True
    else:
        scan.placement_ok = None

    misleading_issues = check_misleading_declarations(declarations)
    cross_image_issues = check_cross_image_conflicts(images)
    severity_issues = misleading_issues + cross_image_issues

    apply_compliance_verdict(scan, declarations, severity_issues)


def process_image(image_id) -> None:
    """Process one claimed image and persist its result or failure."""
    session = SessionLocal()
    temp_path = None

    try:
        image = session.get(ScanImage, image_id)

        if image is None:
            return

        image_bytes = download_image(image.object_key)

        extension = os.path.splitext(image.object_key)[1] or ".jpg"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:
            temp_file.write(image_bytes)
            temp_path = temp_file.name

        original, resized, enhanced, _ = preprocess_image(temp_path)

        if original is None or resized is None or enhanced is None:
            raise RuntimeError("Could not read the uploaded image")

        raw_lines = extract_text_from_image(
            enhanced,
            original_image=original,
            resized_image=resized,
        )

        parsed = parse_legal_metrology_declarations(raw_lines)

        placement_result = check_placement(raw_lines)

        image.raw_text = parsed["raw_text"]
        image.declarations = parsed["declarations"]
        image.font_ok = parsed["font_ok"]
        image.required_mm = parsed["required_mm"]
        image.placement_ok = placement_result["placement_ok"]
        image.processed_at = now_utc()
        image.error_message = None
        image.status = "done"

        scan = session.get(Scan, image.scan_id)

        sibling_images = session.execute(
            select(ScanImage)
            .where(ScanImage.scan_id == image.scan_id)
            .order_by(ScanImage.created_at)
        ).scalars().all()

        if scan is not None:
            finalize_scan(session, scan, sibling_images)

        session.commit()

    except Exception as error:
        session.rollback()

        failure_session = SessionLocal()

        try:
            image = failure_session.get(ScanImage, image_id)

            if image is None:
                return

            image.attempts += 1
            image.error_message = str(error)

            if image.attempts >= 3:
                image.status = "failed"
                image.processed_at = now_utc()

                scan = failure_session.get(Scan, image.scan_id)

                if scan is not None:
                    scan.status = "failed"
            else:
                image.status = "pending"
                image.started_at = None

            failure_session.commit()
        finally:
            failure_session.close()

    finally:
        session.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> None:
    recover_stale_processing()

    while True:
        image_id = claim_image()

        if image_id is None:
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        process_image(image_id)


if __name__ == "__main__":
    main()