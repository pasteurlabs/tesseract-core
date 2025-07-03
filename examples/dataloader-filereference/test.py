from pathlib import Path

from rich import print

from tesseract_core import Tesseract

here = Path(__file__).parent.resolve()
input_path = "./testdata"
output_path = "./output"

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

with Tesseract.from_tesseract_api(
    "tesseract_api.py", input_path=input_path, output_path=output_path
) as tess:
    result = tess.apply({"data": data})
    print(result)
    assert all(p.exists() for p in result["data"])


with Tesseract.from_image(
    "dataloader-filereference",
    # FIXME: to be replaced with input_path and output_path args
    volumes=[
        f"{here.as_posix()}/testdata:/tesseract/input/:ro",
        f"{here.as_posix()}/output:/tesseract/output/:rw",
    ],
) as tess:
    result = tess.apply({"data": data})
    print(result)
    assert all(Path(p).exists() for p in result["data"])
