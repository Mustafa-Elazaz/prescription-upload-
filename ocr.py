from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from app.config import *

print("Loading OCR model...")

processor = TrOCRProcessor.from_pretrained(
    OCR_MODEL_PATH
)

model = VisionEncoderDecoderModel.from_pretrained(
    OCR_MODEL_PATH
)

model.to(DEVICE)
model.eval()

print("OCR model loaded.")


def extract_text(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    inputs = processor(
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_length=30,
        num_beams=2,
        early_stopping=True
    )

    text = processor.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    return text.strip()

from PIL import Image
import cv2

def extract_text_from_crop(crop):

    image = Image.fromarray(
        cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB
        )
    )

    inputs = processor(
        images=[image],
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_length=20,
        num_beams=2,
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    text = processor.batch_decode(
        outputs,
        skip_special_tokens=True
    )[0]

    return text.strip()
