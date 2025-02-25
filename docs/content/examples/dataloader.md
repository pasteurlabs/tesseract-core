# Example Tesseract: Out-of-core dataloading (advanced)

This example demonstrates how to create a Tesseract that can ingest large datasets
without loading the whole dataset to memory.

```{seealso}
For many more examples, see the `examples/` directory in the Tesseract repository, or download them {download}`here </downloads/examples.zip>`.
```

An example dataset is stored at `examples/unit_tesseracts/dataloader/testdata`:

```bash
$ ls examples/unit_tesseracts/dataloader/testdata
sample_0.json sample_2.json sample_4.json sample_6.json sample_8.json
sample_1.json sample_3.json sample_5.json sample_7.json sample_9.json
```

Each sample contains just a small, `base64` encoded array:
```{literalinclude} ../../../examples/unit_tesseracts/dataloader/testdata/sample_0.json
```

But assuming these arrays would be too large to fit into memory all at once,
we can still process them one by one by defining an `InputSchema` with a
`LazySequence`.

```{literalinclude} ../../../examples/unit_tesseracts/dataloader/tesseract_api.py
:start-after: input-schema-label-begin
:end-before: input-schema-label-end
```

Note that the `InputSchema` only contains the actual types that
the data will be parsed into, no file references. The paths to the data will be
provided by the `inputs.json` payload in the end. This makes the tesseract more
flexible as it will accept either file references or data directly.


You can build and run this Tesseract locally:

```bash
$ tesseract build examples/unit_tesseracts/dataloader
```

When running the tesseract, instead of providing the full dataset
in the input payload we can use a glob pattern starting with `@` in place of
the `LazySequence`.

```{literalinclude} ../../../examples/unit_tesseracts/dataloader/test-tesseract.sh
:language: bash
:start-after: tesseract-run-label-begin
:end-before: tesseract-run-label-end
```

The command above is part of the file `examples/unit_tesseracts/dataloader/test-tesseract.sh`.
