# SDK and Runtime Version Compatibility

The Python SDK (`tesseract_core.sdk` and the `tesseract` CLI) and the runtime that serves a Tesseract (`tesseract-runtime`) are released together, but they often run at different versions. An image keeps the runtime it was built with while the host's `tesseract-core` is upgraded, and an older client may call a freshly built Tesseract.

## Supported versions

Each release supports the SDK and runtime of its own minor release and of the three minor releases before it, in both directions:

- The SDK works with Tesseracts served by any runtime in that range.
- The runtime works with clients that use any SDK in that range.

For example, release 1.16 supports runtimes and SDKs from 1.13 to 1.16. Patch releases do not change the range.

Supported means that every endpoint returns the same results in every output format, and that asking an older counterpart for something it cannot do either fails with a clear error before the endpoint runs, or is listed under {ref}`reduced-features-with-older-runtimes`. Older combinations may still work, but nothing guarantees it.

The guarantee covers the protocol between a client and a served Tesseract:

- the request and response schemas of the endpoints,
- the array encodings (`json`, `json+base64`, `json+binref`, and `lz4` compression, see {doc}`/content/reference/array-encodings`),
- the `Accept` header and its `compression` and `gpu_transport` parameters,
- the `x-supported-*` keys in the server's OpenAPI schema,
- error responses, such as input validation errors.

Anything else, such as the Python API of the runtime or the internals of a Tesseract image, comes with no such guarantee.

(reduced-features-with-older-runtimes)=

## Reduced features with older runtimes

Some features of the current SDK need a newer runtime than the oldest one supported:

| Runtime | Limitation                                                                                                                                                 |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.13    | Ignores the requested `compression`; the SDK warns and the response comes back uncompressed. Requesting a GPU transport other than `none` raises an error. |
| 1.14    | Requesting a GPU transport other than `none` raises an error.                                                                                              |

Runtimes older than 1.13 cannot parse `Accept` parameters, so the SDK raises an error when asked for `compression` or `gpu_transport` per request, before sending it.

## Changing the protocol

Changes keep supported combinations working:

- Add to the wire format instead of changing what existing fields mean.
- Do not change a default silently. A change that alters the behavior an older client or runtime relies on is a breaking change, and is announced as such in the release notes.
- If a supported combination cannot keep a feature, the newer side detects that and raises a clear error before any work is done, and the limitation is listed above.
