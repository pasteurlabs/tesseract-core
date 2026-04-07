from pathlib import Path

from rich import print

from tesseract_core import Tesseract

input_path = Path("./test_cases/testdata")
output_path = Path("./output")

# mix of a file and a directory, both relative to input_path
paths = [
    "sample_0.json",
    "sample_dir",
]

expected = ["sample_file.copy", "sample_dir"]

with Tesseract.from_tesseract_api(
    "tesseract_api.py", input_path=input_path, output_path=output_path
) as tess:
    result = tess.apply({"paths": paths})
    print(result)
    out_paths = [(output_path / p) for p in result["paths"]]
    assert len(out_paths) == len(paths)
    assert all(p.exists() for p in out_paths)


with Tesseract.from_image(
    "pathreference",
    input_path=input_path,
    output_path=output_path,
) as tess:
    result = tess.apply({"paths": paths})
    print(result)
    out_paths = [(output_path / p) for p in result["paths"]]
    assert len(out_paths) == len(paths)
    assert all(p.exists() for p in out_paths)
