from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File

from fastapi.middleware.cors import CORSMiddleware

import shutil
import os

from app.preprocessing import preprocess_image

from app.segmentation import segment_medicines

from app.ocr import (
    extract_text,
    extract_text_from_crop
)

from app.validator import (
    validate_medicine,
    validate_medicine_list
)

from app.filter import is_medicine

app = FastAPI(
    title="Prescription OCR API"
)

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
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

    # ---------------------------------
    # PREPROCESS
    # ---------------------------------

    original, thresh = preprocess_image(
        file_path
    )

    # ---------------------------------
    # SEGMENT
    # ---------------------------------

    crops = segment_medicines(
        original,
        thresh
    )

    # ---------------------------------
    # OCR EACH CROP
    # ---------------------------------

    ocr_results = []

    for crop in crops:

        text = extract_text_from_crop(
            crop
        )

        if is_medicine(text):

            ocr_results.append(
                text
            )

    # ---------------------------------
    # FALLBACK
    # ---------------------------------

    if len(ocr_results) == 0:

        text = extract_text(
            file_path
        )

        ocr_results.append(
            text
        )

    # ---------------------------------
    # VALIDATION
    # ---------------------------------
    if len(ocr_results == 1):
        
        medicines = validate_medicine(
        ocr_results
        )
    else:
        medicines = validate_medicine_list(
            ocr_results
        )

    return {

        "success": True,

        "detected_text":

        ocr_results,

        "medicines":

        medicines

    }