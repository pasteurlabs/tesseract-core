# Array encodings

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
$ tesseract run vectoradd apply -f "json+binref" -o /tmp/output @example_inputs.json

$ ls /tmp/output
7796fb36-849a-42ce-8288-a07426111f0c.bin results.json

$ cat /tmp/output/results.json
{"result":{"object_type":"array","shape":[3],"dtype":"float64","data":{"buffer":"7796fb36-849a-42ce-8288-a07426111f0c.bin:0","encoding":"binref"}}}
```
:::
:::{tab-item} REST API
:sync: http
```bash
Tesseracts can read json+binref encoded payloads, but outputting json+binref via
REST API is not supported.
```
:::
::::
It is also possible to use [MessagePack](https://msgpack.org/index.html) for an efficient (but less human readable) encoding:
::::{tab-set}
:::{tab-item} CLI
:sync: cli
```bash
$ tesseract run vectoradd apply --output-format msgpack @examples/vectoradd/example_inputs.json
��result��nd��type�<f8�kind��shape��data�@@"@
```
:::
:::{tab-item} REST API
:sync: http
```bash
$ curl \
  -H "Accept: application/msgpack" \
  -H "Content-Type: application/json" \
  -d @examples/vectoradd/example_inputs.json \
  --output - \
  http://<tesseract-address>:<port>/apply
��result��nd��type�<f8�kind��shape��data�@@"@
```
:::
::::

Here the returned data is binary, so it is suggested to save it in a file rather than
to print it directly to the shell.
