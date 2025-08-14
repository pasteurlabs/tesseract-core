from tesseract_core import Tesseract

with Tesseract.from_image(
    "multi-helloworld:latest", network="my_network", network_alias="multi-helloworld"
) as multi_helloworld_tess:
    with Tesseract.from_image(
        "helloworld:latest", network="my_network", network_alias="helloworld"
    ) as helloworld_tess:
        payload = {
            "name": "YOU",
            "helloworld_tesseract_url": f"http://helloworld:{helloworld_tess._serve_context['port']}",
        }
        result = multi_helloworld_tess.apply(inputs=payload)
        print(result["greeting"])
