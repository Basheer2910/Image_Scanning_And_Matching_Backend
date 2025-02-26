import cv2
import pytesseract
import numpy as np
import sys

def recognize_text(image_path):
    image = cv2.imread(image_path)

    # Convert to grayscale for better OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (Otsu's method) for better contrast
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # (Optional) Save the preprocessed image for inspection
    cv2.imwrite("processed_image.jpg", gray)

    # Extract text using Tesseract OCR
    extracted_text = pytesseract.image_to_string(gray)
    return extracted_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_model.py <image_paths_comma_separated>", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    text = recognize_text(image_path)

    print("text:",text)