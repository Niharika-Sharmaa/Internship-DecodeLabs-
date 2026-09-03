# DecodeLabs Internship - Project 4
# Image / Text Recognition (Basic)
# File: image_text_recognition.py
#
# Covers the Project 4 requirements:
# - Pre-trained AI/library integration
# - OCR using pytesseract (Tesseract)
# - OpenCV image preprocessing
# - Grayscale conversion
# - Gaussian blur
# - Adaptive thresholding
# - Deskewing
# - OCR confidence calculation
# - Minimum 80% confidence validation
# - Clear visual output
# - Optional MobileNet-SSD object detection
# - Bounding-box coordinates and labels
#
# Install:
# pip install opencv-python pytesseract numpy
#
# Tesseract OCR must also be installed on Windows.
# After installation, update TESSERACT_PATH below if required.


import os
import sys
import cv2
import numpy as np
import pytesseract
from pytesseract import Output


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 80.0

# Change this path if Tesseract is installed elsewhere.
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# MobileNet-SSD files for optional object detection.
SSD_PROTOTXT = "MobileNetSSD_deploy.prototxt"
SSD_MODEL = "MobileNetSSD_deploy.caffemodel"

# MobileNet-SSD class labels.
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


# ============================================================
# TESSERACT SETUP
# ============================================================

def configure_tesseract():
    """
    Configure Tesseract OCR.

    If the default Windows installation path exists,
    it is automatically used.
    """

    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


def check_tesseract():
    """Check whether Tesseract OCR is available."""

    try:
        version = pytesseract.get_tesseract_version()
        print(f"[OK] Tesseract OCR detected: {version}")
        return True

    except Exception:
        print("\n[ERROR] Tesseract OCR was not found.")
        print(
            "Install Tesseract OCR and update TESSERACT_PATH "
            "at the top of this file."
        )
        return False


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """Load an image using OpenCV."""

    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        return None

    image = cv2.imread(image_path)

    if image is None:
        print("[ERROR] Unable to read the image.")
        return None

    return image


# ============================================================
# IMAGE PRE-PROCESSING
# ============================================================

def preprocess_image(image):
    """
    OCR preprocessing pipeline:

    1. Grayscale conversion
    2. Gaussian blur
    3. Adaptive thresholding
    """

    # Step 1: Convert RGB/BGR image to grayscale.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Remove small noise using Gaussian blur.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Adaptive thresholding.
    thresholded = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return gray, blurred, thresholded


# ============================================================
# DESKEWING
# ============================================================

def deskew_image(image):
    """
    Detect the dominant text angle and rotate the image
    to bring text closer to a horizontal baseline.
    """

    inverted = cv2.bitwise_not(image)

    coords = np.column_stack(np.where(inverted > 0))

    if len(coords) < 10:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    height, width = image.shape[:2]

    center = (width // 2, height // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    deskewed = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return deskewed


# ============================================================
# OCR RECOGNITION
# ============================================================

def perform_ocr(image, psm=6):
    """
    Perform OCR and extract:
    - recognized text
    - confidence values
    - word bounding boxes
    """

    config = f"--psm {psm}"

    data = pytesseract.image_to_data(
        image,
        config=config,
        output_type=Output.DICT
    )

    words = []
    confidence_values = []

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        try:
            confidence = float(data["conf"][i])
        except (ValueError, TypeError):
            confidence = -1

        if text and confidence >= 0:

            words.append({
                "text": text,
                "confidence": confidence,
                "x": data["left"][i],
                "y": data["top"][i],
                "width": data["width"][i],
                "height": data["height"][i]
            })

            confidence_values.append(confidence)

    extracted_text = " ".join(
        word["text"] for word in words
    )

    if confidence_values:
        average_confidence = sum(confidence_values) / len(
            confidence_values
        )
    else:
        average_confidence = 0.0

    return extracted_text, average_confidence, words


# ============================================================
# OCR VISUAL CONFIRMATION
# ============================================================

def create_ocr_visual_output(image, words, confidence):
    """
    Draw bounding boxes around recognized text.

    Only detections meeting the 80% confidence requirement
    are displayed as validated results.
    """

    output = image.copy()

    for word in words:

        score = word["confidence"]

        if score < CONFIDENCE_THRESHOLD:
            continue

        x = word["x"]
        y = word["y"]
        w = word["width"]
        h = word["height"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = (
            f"{word['text']} "
            f"{score:.1f}%"
        )

        cv2.putText(
            output,
            label,
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    status = (
        "VALIDATED - Confidence >= 80%"
        if confidence >= CONFIDENCE_THRESHOLD
        else "REJECTED - Confidence < 80%"
    )

    cv2.putText(
        output,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0)
        if confidence >= CONFIDENCE_THRESHOLD
        else (0, 0, 255),
        2
    )

    return output


# ============================================================
# MOBILE NET-SSD OBJECT DETECTION
# ============================================================

def load_mobile_net():

    if not os.path.exists(SSD_PROTOTXT):
        print(
            f"[ERROR] Missing model file: {SSD_PROTOTXT}"
        )
        return None

    if not os.path.exists(SSD_MODEL):
        print(
            f"[ERROR] Missing model file: {SSD_MODEL}"
        )
        return None

    try:
        network = cv2.dnn.readNetFromCaffe(
            SSD_PROTOTXT,
            SSD_MODEL
        )

        print("[OK] MobileNet-SSD model loaded.")
        return network

    except Exception as error:
        print(f"[ERROR] Could not load MobileNet-SSD: {error}")
        return None


def perform_object_detection(image, network):
    """
    MobileNet-SSD object detection.

    Uses:
    - cv2.dnn
    - blobFromImage
    - 300 x 300 input
    - normalized bounding-box coordinates
    """

    height, width = image.shape[:2]

    # Blob construction.
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5
    )

    network.setInput(blob)

    detections = network.forward()

    results = []

    for i in range(detections.shape[2]):

        confidence = float(detections[0, 0, i, 2])

        # Project 4 requires a minimum 80% confidence.
        if confidence < 0.80:
            continue

        class_index = int(
            detections[0, 0, i, 1]
        )

        if class_index >= len(CLASSES):
            continue

        label = CLASSES[class_index]

        # Normalized spatial coordinates.
        x1 = int(detections[0, 0, i, 3] * width)
        y1 = int(detections[0, 0, i, 4] * height)
        x2 = int(detections[0, 0, i, 5] * width)
        y2 = int(detections[0, 0, i, 6] * height)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        results.append({
            "label": label,
            "confidence": confidence * 100,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1
        })

    return results


# ============================================================
# OBJECT DETECTION VISUAL OUTPUT
# ============================================================

def create_detection_output(image, detections):

    output = image.copy()

    for detection in detections:

        x = detection["x"]
        y = detection["y"]
        w = detection["width"]
        h = detection["height"]

        confidence = detection["confidence"]

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        label = (
            f"{detection['label']} "
            f"{confidence:.1f}%"
        )

        cv2.putText(
            output,
            label,
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    if detections:
        status = "VALIDATED - All detections >= 80%"
        status_color = (0, 255, 0)
    else:
        status = "NO VALIDATED DETECTIONS"
        status_color = (0, 0, 255)

    cv2.putText(
        output,
        status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    return output


# ============================================================
# SAVE OUTPUT
# ============================================================

def save_image(image, filename):

    success = cv2.imwrite(filename, image)

    if success:
        print(f"[OK] Output saved: {filename}")
    else:
        print(f"[ERROR] Could not save: {filename}")


# ============================================================
# OCR PIPELINE
# ============================================================

def run_ocr_pipeline(image_path):

    print("\n" + "=" * 65)
    print("             PATH 1: OCR RECOGNITION")
    print("=" * 65)

    image = load_image(image_path)

    if image is None:
        return

    # Pre-processing.
    gray, blurred, thresholded = preprocess_image(image)

    save_image(gray, "01_grayscale.png")
    save_image(blurred, "02_gaussian_blur.png")

    # Deskew after thresholding.
    deskewed = deskew_image(thresholded)

    save_image(deskewed, "03_deskewed.png")

    # OCR.
    extracted_text, confidence, words = perform_ocr(
        deskewed,
        psm=6
    )

    print("\n--- OCR OUTPUT ---")

    if extracted_text:
        print(extracted_text)
    else:
        print("No readable text detected.")

    print(
        f"\nAverage OCR Confidence: "
        f"{confidence:.2f}%"
    )

    # 80% confidence validation.
    if confidence >= CONFIDENCE_THRESHOLD:
        print(
            "VALIDATION: PASS "
            "(confidence >= 80%)"
        )
    else:
        print(
            "VALIDATION: FAIL "
            "(confidence < 80%)"
        )

    # Visual confirmation.
    output = create_ocr_visual_output(
        image,
        words,
        confidence
    )

    save_image(
        output,
        "ocr_validated_output.png"
    )

    print("\nValidated OCR words:")

    validated_words = [
        word
        for word in words
        if word["confidence"] >= CONFIDENCE_THRESHOLD
    ]

    if validated_words:

        for word in validated_words:
            print(
                f"  {word['text']} "
                f"-> {word['confidence']:.2f}%"
            )

    else:
        print("  No words passed the 80% threshold.")

    return confidence


# ============================================================
# OBJECT DETECTION PIPELINE
# ============================================================

def run_object_detection_pipeline(image_path):

    print("\n" + "=" * 65)
    print("          PATH 2: MOBILE NET-SSD DETECTION")
    print("=" * 65)

    image = load_image(image_path)

    if image is None:
        return

    network = load_mobile_net()

    if network is None:
        return

    detections = perform_object_detection(
        image,
        network
    )

    print("\n--- OBJECT DETECTION OUTPUT ---")

    if detections:

        for detection in detections:

            print(
                f"Object: {detection['label']}"
            )

            print(
                f"Confidence: "
                f"{detection['confidence']:.2f}%"
            )

            print(
                "Bounding Box: "
                f"(X={detection['x']}, "
                f"Y={detection['y']}, "
                f"W={detection['width']}, "
                f"H={detection['height']})"
            )

            print()

    else:
        print(
            "No object passed the 80% confidence threshold."
        )

    output = create_detection_output(
        image,
        detections
    )

    save_image(
        output,
        "object_detection_output.png"
    )

    return detections


# ============================================================
# MAIN MENU
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("             DECODELABS - PROJECT 4")
    print("       IMAGE / TEXT RECOGNITION (BASIC)")
    print("=" * 65)

    configure_tesseract()

    if not check_tesseract():
        return

    image_path = input(
        "\nEnter the image path: "
    ).strip().strip('"')

    if not image_path:
        print("[ERROR] No image path entered.")
        return

    print("\nChoose recognition path:")
    print("1. OCR - Text Recognition")
    print("2. Object Detection - MobileNet-SSD")
    print("3. Run Both")

    choice = input(
        "\nEnter your choice (1/2/3): "
    ).strip()

    if choice == "1":

        run_ocr_pipeline(image_path)

    elif choice == "2":

        run_object_detection_pipeline(image_path)

    elif choice == "3":

        run_ocr_pipeline(image_path)
        run_object_detection_pipeline(image_path)

    else:

        print("[ERROR] Invalid choice.")

    print("\n" + "=" * 65)
    print("              PROJECT 4 COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
