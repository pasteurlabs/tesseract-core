import shutil
from pathlib import Path

from rich import print

from tesseract_core import Tesseract


def clean():
    # delete before copy
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()


input_path = Path("./test_cases/testdata")
output_path = Path("./output")

# mix of a file and a directory, both relative to input_path
paths = [
    "sample_0.json",
    "sample_8.json",  # contains .bin reference
    "sample_dir",
]

clean()
with Tesseract.from_tesseract_api(
    "tesseract_api.py", input_path=input_path, output_path=output_path, stream_logs=True
) as tess:
    result = tess.apply({"paths": paths})
    print(result)
    out_paths = [(output_path / p) for p in result["paths"]]
    assert len(out_paths) == len(paths)
    assert all(p.exists() for p in out_paths)
    assert len(list(output_path.glob("*.bin"))) == 1


clean()
with Tesseract.from_image(
    "pathreference", input_path=input_path, output_path=output_path, stream_logs=True
) as tess:
    result = tess.apply({"paths": paths})
    print(result)
    out_paths = [(output_path / p) for p in result["paths"]]
    assert len(out_paths) == len(paths)
    assert all(p.exists() for p in out_paths)
    assert len(list(output_path.glob("*.bin"))) == 1
