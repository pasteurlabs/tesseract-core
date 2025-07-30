from pathlib import Path

import tesseract_core.sdk.docker_client as docker_client
from tesseract_core import Tesseract
from tesseract_core.sdk.engine import build_tesseract

helloworld_img = build_tesseract(Path("../../helloworld"), "latest")
multi_helloworld_img = build_tesseract(Path("../multi_helloworld"), "latest")

with Tesseract.from_image(helloworld_img.tags[-1], no_compose=True) as helloworld_tess:
    helloworld_container = docker_client.Containers.get(
        helloworld_tess._serve_context["container_id"]
    )
    helloworld_url = f"{helloworld_container.attrs['NetworkSettings']['Networks']['bridge']['IPAddress']}:8000"
    payload = {
        "name": "YOU",
        "helloworld_tesseract_url": helloworld_url,
    }
    with Tesseract.from_image(
        multi_helloworld_img.tags[-1], no_compose=True
    ) as multi_helloworld_tess:
        result = multi_helloworld_tess.apply(inputs=payload)
        print(result["greeting"])
