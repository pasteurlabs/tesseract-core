from pathlib import Path

from tesseract_core import Tesseract

input_path = "./testdata"

# these are relative to input_path
data = [
    "sample_7.json",
    "sample_6.json",
    "sample_1.json",
    "sample_0.json",
    "sample_3.json",
    "sample_2.json",
    "sample_9.json",
    "sample_5.json",
    "sample_4.json",
    "sample_8.json",
]

with Tesseract.from_tesseract_api("tesseract_api.py", input_path=input_path) as tess:
    tess.apply({"data": data})
