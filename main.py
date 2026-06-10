from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

from fastapi.middleware.cors import CORSMiddleware

import shutil
import os

from app.ocr import extract_text
from app.validator import (
    validate_medicine,
    validate_medicine_list
)

app = FastAPI(
    title="Prescription OCR API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

UPLOAD_FOLDER = "app/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


@app.get("/health")
def health():

    return {
        "status": "running"
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    # -----------------------------
    # OCR
    # -----------------------------

    ocr_text = extract_text(
        file_path
    )

    print("=" * 50)
    print("OCR OUTPUT:")
    print(ocr_text)
    print("=" * 50)

    # -----------------------------
    # Split OCR output
    # -----------------------------

    candidates = []

    for line in ocr_text.split("\n"):

        line = line.strip()

        if len(line) > 2:

            candidates.append(
                line
            )

    # If OCR returned one line only

    if len(candidates) == 0:

        candidates.append(
            ocr_text
        )

    # -----------------------------
    # Validation
    # -----------------------------

    medicines = validate_medicine_list(
        candidates
    )

    return {

        "success": True,

        "ocr_text":
        ocr_text,

        "detected_text":
        candidates,

        "medicines":
        medicines

    }
