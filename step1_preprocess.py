import cv2
import os


def preprocess_image(image_path, target_height=2000):
    image = cv2.imread(image_path)
    if image is None:
        return None, None, None, None

    original = image.copy()

    height, width = image.shape[:2]
    aspect_ratio = width / height
    target_width = int(target_height * aspect_ratio)

    interpolation = (
        cv2.INTER_AREA
        if target_height <= height
        else cv2.INTER_LANCZOS4
    )

    resized = cv2.resize(
        image,
        (target_width, target_height),
        interpolation=interpolation,
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=3,
    )

    sharpened = cv2.addWeighted(
        gray,
        1.5,
        blurred,
        -0.5,
        0,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(sharpened)

    return original, resized, enhanced, blur_score


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    print("Preprocessing module ready!")
