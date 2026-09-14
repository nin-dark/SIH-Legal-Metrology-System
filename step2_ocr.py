import easyocr
import numpy as np

reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def _run_ocr(image):
    ocr = get_ocr_reader()
    results = ocr.readtext(image)
    lines = []
    for (bbox, text, confidence) in results:
        if confidence > 0.20:
            lines.append({
                "text": text,
                "confidence": round(float(confidence), 2),
                "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox]
            })
    return lines

def _score(lines):
    return sum(len(l["text"]) * l["confidence"] for l in lines)

# If the upright pass already found a decent amount of confident text,
# we skip trying other rotations entirely — this is what keeps normal,
# correctly-oriented photos fast. Only weak/sparse results trigger the
# slower full rotation search (for genuinely sideways photos).
EARLY_EXIT_SCORE = 60

def extract_text_from_image(enhanced_image):
    upright_lines = _run_ocr(enhanced_image)
    if _score(upright_lines) >= EARLY_EXIT_SCORE:
        return upright_lines

    best_lines, best_score = upright_lines, _score(upright_lines)
    for k in (1, 2, 3):  # 90, 180, 270 degrees
        rotated = np.rot90(enhanced_image, k=k)
        lines = _run_ocr(rotated)
        s = _score(lines)
        if s > best_score:
            best_score, best_lines = s, lines
    return best_lines