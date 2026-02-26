from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from tesseract_core import Tesseract
from tesseract_core.sdk.tesseract import (
    HTTPClient,
    _decode_array,
    _encode_array,
    _tree_map,
)


@pytest.fixture
def mock_serving(mocker):
    fake_container = SimpleNamespace()
    fake_container.host_port = 1234
    fake_container.id = "container-id-123"

    serve_mock = mocker.patch("tesseract_core.sdk.engine.serve")
    serve_mock.return_value = fake_container.id, fake_container

    teardown_mock = mocker.patch("tesseract_core.sdk.engine.teardown")
    logs_mock = mocker.patch("tesseract_core.sdk.engine.logs")
    return {
        "serve_mock": serve_mock,
        "teardown_mock": teardown_mock,
        "logs_mock": logs_mock,
    }


@pytest.fixture
def mock_clients(mocker):
    mocker.patch("tesseract_core.sdk.tesseract.HTTPClient.run_tesseract")


def test_Tesseract_init():
    # Instantiate with a url
    with pytest.warns(
        UserWarning, match="Direct instantiation of Tesseract is deprecated"
    ):
        t = Tesseract(url="localhost")

    # Using it as a context manager should be a no-op
    with t:
        pass


def test_Tesseract_from_url():
    # Instantiate with a url
    t = Tesseract.from_url("localhost")

    # Using it as a context manager should be a no-op
    with t:
        pass


def test_Tesseract_from_tesseract_api(dummy_tesseract_location, dummy_tesseract_module):
    all_endpoints = {
        "apply",
        "jacobian",
        "jacobian_vector_product",
        "vector_jacobian_product",
        "health",
        "abstract_eval",
        "test",
    }

    t = Tesseract.from_tesseract_api(dummy_tesseract_location / "tesseract_api.py")
    endpoints = set(t.available_endpoints)
    assert endpoints == all_endpoints

    # should also work when importing the module
    t = Tesseract.from_tesseract_api(dummy_tesseract_module)
    endpoints = set(t.available_endpoints)
    assert endpoints == all_endpoints


def test_Tesseract_from_image(mock_serving, mock_clients):
    # Object is built and has the correct attributes set
    t = Tesseract.from_image(
        "sometesseract:0.2.3", input_path="/my/files", gpus=["all"]
    )

    # Now we can use it as a context manager
    # NOTE: we invoke available_endpoints because it requires an active client and is not cached
    with t:
        _ = t.available_endpoints

    # Trying to use methods from outside the context manager should raise
    with pytest.raises(RuntimeError):
        _ = t.available_endpoints

    # Works if we serve first
    try:
        t.serve()
        _ = t.available_endpoints
    finally:
        t.teardown()


def test_Tesseract_schema_method(mocker, mock_serving):
    mocked_run = mocker.patch("tesseract_core.sdk.tesseract.HTTPClient.run_tesseract")
    mocked_run.return_value = {"#defs": {"some": "stuff"}}

    with Tesseract.from_image("sometesseract:0.2.3") as t:
        openapi_schema = t.openapi_schema

    assert openapi_schema == mocked_run.return_value


def test_serve_lifecycle(mock_serving, mock_clients):
    t = Tesseract.from_image("sometesseract:0.2.3")

    with t:
        pass

    mock_serving["serve_mock"].assert_called_with(
        image_name="sometesseract:0.2.3",
        port=None,
        volumes=[],
        environment={},
        gpus=None,
        debug=True,
        num_workers=1,
        network=None,
        network_alias=None,
        host_ip="127.0.0.1",
        user=None,
        memory=None,
        input_path=None,
        output_path=None,
        output_format="json+base64",
    )

    mock_serving["teardown_mock"].assert_called_with("container-id-123")

    # check that the same Tesseract obj cannot be used to instantiate two containers
    with pytest.raises(RuntimeError):
        with t:
            with t:
                pass


@pytest.mark.parametrize(
    "run_id",
    [None, "fizzbuzz"],
)
def test_HTTPClient_run_tesseract(mocker, run_id):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"result": [4, 4, 4]}
    mock_response.raise_for_status = mocker.Mock()

    mocked_request = mocker.patch(
        "requests.request",
        return_value=mock_response,
    )

    client = HTTPClient("somehost")

    out = client.run_tesseract("apply", {"inputs": {"a": 1}}, run_id=run_id)

    assert out == {"result": [4, 4, 4]}
    expected_params = {} if run_id is None else {"run_id": run_id}
    mocked_request.assert_called_with(
        method="POST",
        url="http://somehost/apply",
        json={"inputs": {"a": 1}},
        params=expected_params,
    )


def test_HTTPClient_run_tesseract_raises_validation_error(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "detail": [
            {
                "type": "missing",
                "loc": ["body", "inputs", "a"],
                "msg": "Field required",
                "input": {"whoops": "whatever"},
            },
            {
                "type": "missing",
                "loc": ["body", "inputs", "b"],
                "msg": "Field required",
                "input": {"whoops": "whatever"},
            },
            {
                "type": "extra_forbidden",
                "loc": ["body", "inputs", "whoops"],
                "msg": "Extra inputs are not permitted",
                "input": "whatever",
            },
            {
                "type": "value_error",
                "loc": ["body", "inputs", "bar"],
                "msg": "Value error, Dimensionality mismatch: 2D array cannot be cast to 1D",
                "input": [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                "error": {},
            },
        ]
    }
    mock_response.status_code = 422

    mocker.patch(
        "requests.request",
        return_value=mock_response,
    )

    client = HTTPClient("somehost")

    with pytest.raises(ValidationError) as excinfo:
        client.run_tesseract("apply", {"inputs": {"whoops": "whatever"}})

    # This checks as well that no duplicate "Value error" is in msg
    assert (
        excinfo.value.errors()[3]["msg"]
        == "Value error, Dimensionality mismatch: 2D array cannot be cast to 1D"
    )


@pytest.mark.parametrize(
    "b64, expected_data",
    [
        (True, {"buffer": "AACAPwAAAEAAAEBA", "encoding": "base64"}),
        (False, {"buffer": [1.0, 2.0, 3.0], "encoding": "raw"}),
    ],
)
def test_encode_array(b64, expected_data):
    a = np.array([1.0, 2.0, 3.0], dtype="float32")

    encoded = _encode_array(a, b64=b64)

    assert encoded["shape"] == (3,)
    assert encoded["dtype"] == "float32"
    assert encoded["data"] == expected_data


@pytest.mark.parametrize(
    "encoded, expected",
    [
        (
            {
                "shape": (3,),
                "dtype": "float32",
                "data": {"buffer": [1.0, 2.0, 3.0], "encoding": "raw"},
            },
            np.array([1.0, 2.0, 3.0], dtype="float32"),
        ),
        (
            {
                "shape": (1, 3),
                "dtype": "float32",
                "data": {"buffer": "AACAPwAAAEAAAEBA", "encoding": "base64"},
            },
            np.array([[1.0, 2.0, 3.0]], dtype="float32"),
        ),
    ],
)
def test_decode_array(encoded, expected):
    decoded = _decode_array(encoded)
    np.testing.assert_array_equal(decoded, expected, strict=True)


@pytest.mark.parametrize(
    "dtype",
    ["float32", "float64", "int32", "int64", "bool"],
)
def test_decode_array_various_dtypes(dtype):
    """Test that _decode_array handles various dtypes correctly with base64 encoding."""
    # Create test array with appropriate values for each dtype
    if dtype == "bool":
        original = np.array([True, False, True], dtype=dtype)
    else:
        original = np.array([1, 2, 3], dtype=dtype)

    # Encode using _encode_array with base64
    encoded = _encode_array(original, b64=True)

    # Decode back
    decoded = _decode_array(encoded)

    # Verify equivalence
    np.testing.assert_array_equal(decoded, original, strict=True)
    assert decoded.dtype == original.dtype


def test_tree_map():
    tree = {
        "a": [10, 20],
        "b": {"c": np.array([30])},
        "d": {"e": np.array([1.0, 2.0, 3.0])},
        "f": "hello",
    }

    encoded = _tree_map(_encode_array, tree, is_leaf=lambda x: hasattr(x, "shape"))

    assert encoded == {
        "a": [10, 20],
        "b": {
            "c": {
                "shape": (1,),
                "dtype": "int64",
                "data": {"buffer": "HgAAAAAAAAA=", "encoding": "base64"},
            }
        },
        "d": {
            "e": {
                "shape": (3,),
                "dtype": "float64",
                "data": {
                    "buffer": "AAAAAAAA8D8AAAAAAAAAQAAAAAAAAAhA",
                    "encoding": "base64",
                },
            }
        },
        "f": "hello",
    }


def test_test_endpoint_success_local(dummy_tesseract_package):
    """Test test() endpoint with LocalClient."""
    tess = Tesseract.from_tesseract_api(dummy_tesseract_package / "tesseract_api.py")

    # Should not raise
    tess.test(
        {
            "endpoint": "apply",
            "payload": {
                "inputs": {
                    "a": np.array([1.0, 2.0], dtype=np.float32),
                    "b": np.array([3.0, 4.0], dtype=np.float32),
                    "s": 1,
                }
            },
            "expected_outputs": {"result": np.array([4.0, 6.0], dtype=np.float32)},
        }
    )


def test_test_endpoint_failure_local(dummy_tesseract_package):
    """Test test() endpoint failure with LocalClient."""
    tess = Tesseract.from_tesseract_api(dummy_tesseract_package / "tesseract_api.py")

    with pytest.raises(AssertionError, match="Values are not sufficiently close"):
        tess.test(
            {
                "endpoint": "apply",
                "payload": {
                    "inputs": {
                        "a": np.array([1.0, 2.0], dtype=np.float32),
                        "b": np.array([3.0, 4.0], dtype=np.float32),
                        "s": 1,
                    }
                },
                "expected_outputs": {
                    "result": np.array([999.0, 999.0], dtype=np.float32)
                },
            }
        )


def test_test_endpoint_with_exception_type_local(dummy_tesseract_package):
    """Test test() endpoint with exception type (not string) using LocalClient."""
    tess = Tesseract.from_tesseract_api(dummy_tesseract_package / "tesseract_api.py")

    # Should not raise - exception type passed directly
    tess.test(
        {
            "endpoint": "apply",
            "payload": {
                "inputs": {
                    "a": np.array([1.0, 2.0], dtype=np.float32),
                    "b": np.array([4.0], dtype=np.float32),  # Wrong shape
                    "s": 1,
                }
            },
            "expected_exception": ValidationError,  # Type, not string
        }
    )
