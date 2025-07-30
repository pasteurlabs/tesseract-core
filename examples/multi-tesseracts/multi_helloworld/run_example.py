import tesseract_core.sdk.docker_client as docker_client
from tesseract_core import Tesseract

with Tesseract.from_image("multi-helloworld:latest") as multi_helloworld_tess:
    with Tesseract.from_image("helloworld:latest") as helloworld_tess:
        helloworld_container = docker_client.Containers.get(
            helloworld_tess._serve_context["container_name"]
        )
        helloworld_url = f"{helloworld_container.attrs['NetworkSettings']['Networks']['bridge']['IPAddress']}:8000"
        payload = {
            "name": "YOU",
            "helloworld_tesseract_url": helloworld_url,
        }
        result = multi_helloworld_tess.apply(inputs=payload)
        print(result["greeting"])
