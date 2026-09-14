import numpy as np
import pytest

import step2_ocr


def line(text, confidence=0.9):
    return {
        "text": text,
        "confidence": confidence,
        "bbox": [[0, 0], [20, 0], [20, 10], [0, 10]],
    }


def test_as_3channel_converts_grayscale():
    image = np.zeros((10, 20), dtype=np.uint8)

    result = step2_ocr._as_3channel(image)

    assert result.shape == (10, 20, 3)
    assert np.array_equal(result[:, :, 0], image)
    assert np.array_equal(result[:, :, 1], image)
    assert np.array_equal(result[:, :, 2], image)


def test_as_3channel_converts_single_channel():
    image = np.zeros((10, 20, 1), dtype=np.uint8)

    result = step2_ocr._as_3channel(image)

    assert result.shape == (10, 20, 3)


def test_bbox_from_poly_normalizes_coordinates():
    poly = np.array(
        [
            [1.8, 2.9],
            [10.2, 2.1],
            [10.9, 12.7],
            [1.1, 12.2],
        ]
    )

    assert step2_ocr._bbox_from_poly(poly) == [
        [1, 2],
        [10, 2],
        [10, 12],
        [1, 12],
    ]


def test_unsupported_ocr_engine_is_rejected(monkeypatch):
    monkeypatch.setattr(step2_ocr, "OCR_ENGINE", "unsupported")

    with pytest.raises(ValueError, match="Production OCR_ENGINE must be 'paddle'"):
        step2_ocr._run_ocr(np.zeros((10, 10, 3), dtype=np.uint8))


def test_strong_result_skips_rotation(monkeypatch):
    calls = []

    def fake_run(image):
        calls.append(image)
        return [
            line("MRP ₹100", confidence=0.99),
            line("NET QUANTITY 100 g", confidence=0.99),
        ]

    monkeypatch.setattr(step2_ocr, "_run_ocr", fake_run)

    image = np.zeros((20, 20, 3), dtype=np.uint8)

    result = step2_ocr.extract_text_from_image(image)

    assert result[0]["text"] == "MRP ₹100"
    assert len(calls) == 1


def test_weak_result_tries_three_rotations(monkeypatch):
    calls = []

    def fake_run(image):
        calls.append(image)
        return [line("x", confidence=0.1)]

    monkeypatch.setattr(step2_ocr, "_run_ocr", fake_run)

    image = np.zeros((20, 30, 3), dtype=np.uint8)

    step2_ocr.extract_text_from_image(image)

    # One normal pass + 90 + 180 + 270.
    assert len(calls) == 4


def test_rotation_result_can_replace_weak_upright_result(monkeypatch):
    calls = []

    upright = [line("x", confidence=0.1)]
    rotated = [
        line("MRP ₹100", confidence=0.99),
        line("NET QUANTITY 100 g", confidence=0.99),
    ]

    def fake_run(image):
        calls.append(image)
        return upright if len(calls) == 1 else rotated

    monkeypatch.setattr(step2_ocr, "_run_ocr", fake_run)

    image = np.zeros((20, 30, 3), dtype=np.uint8)

    result = step2_ocr.extract_text_from_image(image)

    assert result[0]["text"] == "MRP ₹100"
    assert len(calls) == 2


def test_original_candidate_is_preferred_when_it_has_declaration_hits(
    monkeypatch,
):
    def fake_run(image):
        # Distinguish candidates by the array object identity.
        if image is original:
            return [
                line("MRP ₹100", confidence=0.9),
                line("NET QUANTITY 100 g", confidence=0.9),
            ]

        return [line("random text " * 4, confidence=0.99)]

    monkeypatch.setattr(step2_ocr, "_run_ocr", fake_run)

    image = np.zeros((20, 30, 3), dtype=np.uint8)
    original = np.zeros((20, 30, 3), dtype=np.uint8)
    resized = np.zeros((40, 60, 3), dtype=np.uint8)

    result = step2_ocr.extract_text_from_image(
        image,
        original_image=original,
        resized_image=resized,
    )

    assert [item["text"] for item in result] == [
        "MRP ₹100",
        "NET QUANTITY 100 g",
    ]


def test_restore_rotated_bbox_90_degrees():
    bbox = [[0, 0], [3, 0], [3, 2], [0, 2]]

    restored = step2_ocr._restore_rotated_bbox(
        bbox,
        (5, 10, 3),
        1,
    )

    assert restored == [
        [9, 0],
        [9, 3],
        [7, 3],
        [7, 0],
    ]


def test_restore_rotated_bbox_180_degrees():
    bbox = [[0, 0], [3, 0], [3, 2], [0, 2]]

    restored = step2_ocr._restore_rotated_bbox(
        bbox,
        (5, 10, 3),
        2,
    )

    assert restored == [
        [9, 4],
        [6, 4],
        [6, 2],
        [9, 2],
    ]


def test_rotation_result_bboxes_are_restored_to_original_coordinates(
    monkeypatch,
):
    calls = []

    def fake_run(image):
        calls.append(image)

        if len(calls) == 1:
            return [line("x", confidence=0.1)]

        return [
            {
                "text": "MRP ₹100",
                "confidence": 0.99,
                "bbox": [
                    [0, 0],
                    [3, 0],
                    [3, 2],
                    [0, 2],
                ],
            },
            {
                "text": "NET QUANTITY 100 g",
                "confidence": 0.99,
                "bbox": [
                    [10, 0],
                    [13, 0],
                    [13, 2],
                    [10, 2],
                ],
            },
        ]

    monkeypatch.setattr(step2_ocr, "_run_ocr", fake_run)

    image = np.zeros((5, 10, 3), dtype=np.uint8)

    result = step2_ocr.extract_text_from_image(image)

    assert len(result) == 2
    assert result[0]["bbox"] == [
        [9, 0],
        [9, 3],
        [7, 3],
        [7, 0],
    ]
