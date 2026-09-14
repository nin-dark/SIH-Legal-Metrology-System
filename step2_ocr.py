import os

import numpy as np

OCR_ENGINE = os.getenv("OCR_ENGINE", "paddle").strip().lower()

_paddle_reader = None


def _get_paddle_reader():
    global _paddle_reader

    if _paddle_reader is None:
        from paddleocr import PaddleOCR

        _paddle_reader = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )

    return _paddle_reader


def _as_3channel(image):
    if image is None:
        return None

    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)

    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)

    return image


def _bbox_from_poly(poly):
    if poly is None:
        return []

    return [
        [int(point[0]), int(point[1])]
        for point in poly
    ]


def _restore_rotated_bbox(bbox, original_shape, k):
    """Map bbox coordinates from np.rot90(image, k) back to original image."""
    if len(bbox) != 4:
        return bbox

    height, width = original_shape[:2]
    normalized_k = k % 4

    restored = []

    for x_rot, y_rot in bbox:
        if normalized_k == 1:
            x = width - 1 - y_rot
            y = x_rot
        elif normalized_k == 2:
            x = width - 1 - x_rot
            y = height - 1 - y_rot
        elif normalized_k == 3:
            x = y_rot
            y = height - 1 - x_rot
        else:
            x = x_rot
            y = y_rot

        restored.append([int(x), int(y)])

    return restored


def _run_paddleocr(image):
    reader = _get_paddle_reader()
    image = _as_3channel(image)
    result = reader.predict(image)

    lines = []

    for page in result:
        page_json = getattr(page, "json", None)

        if callable(page_json):
            page_json = page_json()

        if not isinstance(page_json, dict):
            continue

        data = page_json.get("res", page_json)

        texts = data.get("rec_texts") or []
        scores = data.get("rec_scores") or []
        polys = data.get("rec_polys") or data.get("dt_polys") or []

        for index, text in enumerate(texts):
            confidence = (
                float(scores[index])
                if index < len(scores)
                else 0.0
            )

            bbox = (
                _bbox_from_poly(polys[index])
                if index < len(polys)
                else []
            )

            lines.append(
                {
                    "text": str(text),
                    "confidence": round(confidence, 2),
                    "bbox": bbox,
                }
            )

    return lines


def _run_ocr(image):
    if OCR_ENGINE != "paddle":
        raise ValueError(
            f"Unsupported OCR_ENGINE={OCR_ENGINE!r}. "
            "Production OCR_ENGINE must be 'paddle'."
        )

    return _run_paddleocr(image)


def _score(lines):
    score = 0.0

    declaration_keywords = (
        "mrp",
        "retail price",
        "net quantity",
        "quantity",
        "mfg",
        "manufactured",
        "manufacture",
        "packed",
        "packer",
        "consumer care",
        "customer care",
        "helpline",
        "1800",
        "address",
        "lic. no",
    )

    for line in lines:
        text = line["text"]
        confidence = line["confidence"]

        score += len(text) * confidence

        lowered = text.lower()

        for keyword in declaration_keywords:
            if keyword in lowered:
                score += 25

    return score



def extract_text_from_image(
    image,
    original_image=None,
    resized_image=None,
):
    """
    Run PaddleOCR over the original/preprocessed candidates.

    The original and resized candidates improve OCR robustness.
    If normal OCR results are weak, rotated versions of the enhanced
    image are also tested at 90, 180, and 270 degrees.
    """

    candidates = []

    if original_image is not None:
        try:
            candidates.append(
                ("original", _run_ocr(original_image))
            )
        except Exception:
            pass

    candidates.append(
        ("input", _run_ocr(image))
    )

    if resized_image is not None:
        try:
            candidates.append(
                ("resized", _run_ocr(resized_image))
            )
        except Exception:
            pass

    _, best_lines = max(
        candidates,
        key=lambda item: _score(item[1]),
    )

    best_score = _score(best_lines)

    # Prefer the original image when it contains strong declaration
    # evidence. This avoids replacing useful original-image detections
    # with noisy enlarged-image detections.
    for name, lines in candidates:
        lowered = "\n".join(
            line["text"].lower()
            for line in lines
        )

        declaration_hits = sum(
            keyword in lowered
            for keyword in (
                "mrp",
                "net quantity",
                "consumer care",
                "1800",
                "mfg",
                "manufactured",
                "packed",
                "lic. no",
            )
        )

        if name == "original" and (
            declaration_hits >= 2
            or _score(lines) >= 60
        ):
            return lines

    # Don't perform expensive rotations when normal OCR is strong.
    if best_score >= 60:
        return best_lines

    # Rotation fallback for sideways/rotated package labels.
    for k in (1, 2, 3):
        rotated = np.rot90(image, k=k)
        lines = _run_ocr(rotated)

        for line in lines:
            line["bbox"] = _restore_rotated_bbox(
                line.get("bbox", []),
                image.shape,
                k,
            )

        score = _score(lines)

        if score > best_score:
            best_score = score
            best_lines = lines

        if score >= 60:
            return lines

    return best_lines