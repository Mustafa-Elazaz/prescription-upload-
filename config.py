import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

OCR_MODEL_PATH = "app/models/trocr-finetuned"

MEDICINE_DATASET = "app/data/medicines_data_updated.csv"