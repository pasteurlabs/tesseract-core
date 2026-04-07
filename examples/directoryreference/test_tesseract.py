from pathlib import Path

from rich import print

from tesseract_core import Tesseract

input_path = Path("./test_cases/testdata")
output_path = Path("./output")

# these are relative to input_path
dirs = [
    "sample_dir_0",
    "sample_dir_1",
]

with Tesseract.from_tesseract_api(
    "tesseract_api.py", input_path=input_path, output_path=output_path
) as tess:
    result = tess.apply({"dirs": dirs})
    print(result)
    paths = [(output_path / p) for p in result["dirs"]]
    assert len(paths) == len(dirs)
    assert all(p.is_dir() for p in paths)


with Tesseract.from_image(
    "directoryreference",
    input_path=input_path,
    output_path=output_path,
) as tess:
    result = tess.apply({"dirs": dirs})
    print(result)
    paths = [(output_path / p) for p in result["dirs"]]
    assert len(paths) == len(dirs)
    assert all(p.is_dir() for p in paths)
