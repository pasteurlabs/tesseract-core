# Array Encodings

```{note}
This section is only relevant to CLI or REST API usage; if you are using the Python API you can ignore
this section, as in the Python client everything is base64 encoded under the hood.
```

By default, Tesseracts return the numeric data contained in arrays encoded as a human-readable string; this
is often convenient, but it is not optimal in terms of memory footprint and in order to avoid loss of precision.
If you are using the CLI or REST API and don't need human-readable numeric values,
you can make Tesseracts return base64-encoded arrays by setting the format to `json+base64`:

::::{tab-set}
:::{tab-item} CLI
:sync: cli
```bash
$ tesseract run vectoradd apply -f "json+base64" @examples/vectoradd/example_inputs_b64.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"AAAAAAAALEAAAAAAAAA2QAAAAAAAAD5A","encoding":"base64"}}}
```
:::
:::{tab-item} REST API
:sync: http
```bash
$ curl \
  -H "Accept: application/json+base64" \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json \
  http://<tesseract-address>:<port>/apply
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"AAAAAAAALEAAAAAAAAA2QAAAAAAAAD5A","encoding":"base64"}}}
```
:::
::::

For large payloads you can use the `json+binref` format, which dumps a
`.json` with references to a `.bin` file that contains the array data as raw binary. This
avoids dealing with otherwise huge JSON files, and provides a powerful way to lazily load binary data with [LazySequence](#tesseract_core.runtime.experimental.LazySequence). Check out the [`Array`
docstring](#tesseract_core.runtime.Array) for details on how to use different array
encodings in Tesseracts.

::::{tab-set}
:::{tab-item} CLI
:sync: cli
```bash
$ tesseract run vectoradd apply -f "json+binref" -o /tmp/output @examples/vectoradd/example_inputs.json

$ ls /tmp/output
7796fb36-849a-42ce-8288-a07426111f0c.bin results.json

$ cat /tmp/output/results.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"7796fb36-849a-42ce-8288-a07426111f0c.bin:0","encoding":"binref"}}}
```
:::
:::{tab-item} REST API
:sync: http

To access the `.bin` files that are written when using the `json+binref` format, make sure
to specify `--output-path` when serving your Tesseract. Otherwise the `.bin` files will only be accessible *inside* the Tesseract (under `/tesseract/output_path`).
```bash
$ tesseract serve <tesseract-name> --output-path /tmp/output
$ curl \
  -H "Accept: application/json+binref" \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json \
  http://<tesseract-address>:<port>/apply
```
The references to `.bin` files are relative to the `--output-path` you specified when serving the Tesseract.
:::
::::
