# `FileReference` example

```bash
tesseract-runtime apply \
    --input-path ./testdata \
    --output-path ./outputs \
    '{"inputs": {"data": ["sample_0.json", "sample_1.json"]}}'
```

```bash
tesseract run filereference apply \
    --input-path ./testdata \
    --output-path ./outputs \
    @inputs.json
```

For the Python SDK usage examples see `test_tesseract.py`.
