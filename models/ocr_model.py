import cv2
import numpy as np
import os
import sys
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

try:
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        "microsoft/trocr-base-handwritten", ignore_mismatched_sizes=True
    )
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

def recognize_text(image_path):
    """Perform OCR on an image."""
    
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}", file=sys.stderr)
        return ""

    try:
        image = Image.open(image_path).convert("RGB")       
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text
    except Exception as e:
        print(f"Error during OCR: {e}", file=sys.stderr)
        return ""

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ocr_model.py <image_paths_comma_separated>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    text = recognize_text(image_path)

    print("text:",text)
