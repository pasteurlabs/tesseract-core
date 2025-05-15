# Dataloader

## Context
This is an example of a Tesseract that loads in data from a folder by mounting the folder in the cli.

## Example Tesseract (examples/dataloader)

In the actual Tesseract, we may have logic that's expecting data samples for the input schema

```{literalinclude} ../../../examples/dataloader/tesseract_api.py
:pyobject: InputSchema
:language: python
```

The inputted data may be processed as such in an apply function

```{literalinclude} ../../../examples/dataloader/tesseract_api.py
:pyobject: apply
:language: python
```

You can then pass in data into this Tesseract by mounting the directory where the data samples are stored using the tesseract flag `--volume`

```{literalinclude} ../../../examples/dataloader/test-tesseract.sh
:start-after: # tesseract-run-label-begin
#end-before: # tesseract-run-label-end
:language: bash
```
