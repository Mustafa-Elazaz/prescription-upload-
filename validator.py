import re
import pandas as pd

from rapidfuzz import process
from rapidfuzz import fuzz
from unidecode import unidecode

from app.config import *

print("Loading medicine database...")

df = pd.read_csv(
    MEDICINE_DATASET
)

df = df[
    [
        "Drugname",
        "Form",
        "Company",
        "Category"
    ]
].fillna("Unknown")

df = df.drop_duplicates()


def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = unidecode(text)

    text = text.replace("0", "o")
    text = text.replace("1", "l")
    text = text.replace("5", "s")

    text = re.sub(
        r'[^a-z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    text = re.sub(
    r'\b\d+(\.\d+)?\s*mg\b',
    '',
    text
)

    REMOVE_WORDS = [

        "tab",

        "tablet",

        "cap",

        "capsule",

        "inj",

        "injection",

        "syrup"

    ]

    for word in REMOVE_WORDS:

        text = text.replace(
            word,
            ""
        )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


df["Drugname_norm"] = (
    df["Drugname"]
    .apply(normalize_text)
)

medicine_choices = (
    df["Drugname_norm"]
    .unique()
)


def validate_medicine(
        ocr_output,
        threshold=70
):

    cleaned_text = normalize_text(
        ocr_output
    )

    exact = df[
        df["Drugname_norm"]
        == cleaned_text
    ]

    if not exact.empty:

        result = exact.iloc[0]

        return {
            "Drugname":
            result["Drugname"],

            "Form":
            result["Form"],

            "Category":
            result["Category"]
        }

    best_match = process.extractOne(
        cleaned_text,
        medicine_choices,
        scorer=fuzz.WRatio
    )

    if best_match:

        matched = best_match[0]
        score = best_match[1]

        if score >= threshold:

            result = df[
                df["Drugname_norm"]
                == matched
            ].iloc[0]

            return {
                "Drugname":
                result["Drugname"],

                "Form":
                result["Form"],

                "Category":
                result["Category"]
            }

    return None

def validate_medicine_list(ocr_results):

    medicines = []

    seen = set()

    for text in ocr_results:

        result = validate_medicine(text)

        if result:

            if result["Drugname"] not in seen:

                seen.add(
                    result["Drugname"]
                )

                medicines.append(
                    result
                )

    return medicines
