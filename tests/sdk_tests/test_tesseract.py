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
    serve_mock = mocker.patch("tesseract_core.sdk.engine.serve")
    serve_mock.return_value = "proj-id-123"

    subprocess_run_mock = mocker.patch("subprocess.run")
    subprocess_run_mock.return_value.stdout = (
        '{"ID": "abc1234", "Publishers":[{"PublishedPort": 54321}]}'
    )

    teardown_mock = mocker.patch("tesseract_core.sdk.engine.teardown")
    return {
        "serve_mock": serve_mock,
        "subprocess_run_mock": subprocess_run_mock,
        "teardown_mock": teardown_mock,
    }


@pytest.fixture
def mock_clients(mocker):
    mocker.patch("tesseract_core.sdk.tesseract.HTTPClient._run_tesseract")


def test_Tesseract_init():
    # Instantiate with a url
    t = Tesseract(url="localhost")

    # The attributes for local Tesseracts should not be set
    assert not hasattr(t, "image")
    assert not hasattr(t, "gpus")
    assert not hasattr(t, "volumes")


def test_Tesseract_from_image():
    # Object is built and has the correct attributes set
    t = Tesseract.from_image("sometesseract:0.2.3", volumes=["/my/files"], gpus=["all"])

    assert t.image == "sometesseract:0.2.3"
    assert t.gpus == ["all"]
    assert t.volumes == ["/my/files"]

    # Let's also check that stuff we don't expect there is not there
    assert not hasattr(t, "url")


def test_Tesseract_schema_methods(mocker, mock_serving):
    mocked_run = mocker.patch("tesseract_core.sdk.tesseract.HTTPClient._run_tesseract")
    mocked_run.return_value = {"#defs": {"some": "stuff"}}

    with Tesseract.from_image("sometesseract:0.2.3") as t:
        input_schema = t.input_schema
        output_schema = t.output_schema
        openapi_schema = t.openapi_schema

    assert input_schema == output_schema == openapi_schema == mocked_run.return_value


def test_serve_lifecycle(mock_serving, mock_clients):
    t = Tesseract.from_image("sometesseract:0.2.3")

    with t:
        pass

    mock_serving["serve_mock"].assert_called_with(
        ["sometesseract:0.2.3"], port="", volumes=None, gpus=None
    )

    mock_serving["teardown_mock"].assert_called_with("proj-id-123")

    # check that the same Tesseract obj cannot be used to instantiate two containers
    with pytest.raises(RuntimeError):
        with t:
            with t:
                pass


def test_HTTPClient_run_tesseract(mocker):
    mock_response = mocker.Mock()
    mock_response.json.return_value = {"result": [4, 4, 4]}
    mock_response.raise_for_status = mocker.Mock()

    mocked_request = mocker.patch(
        "requests.request",
        return_value=mock_response,
    )

    client = HTTPClient("somehost")

    out = client._run_tesseract("apply", {"inputs": {"a": 1}})

    assert out == {"result": [4, 4, 4]}
    mocked_request.assert_called_with(
        method="POST",
        url="http://somehost/apply",
        json={"inputs": {"a": 1}},
    )


def test_HTTPClient__run_tesseract_raises_validation_error(mocker):
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
        ]
    }
    mock_response.status_code = 422

    mocker.patch(
        "requests.request",
        return_value=mock_response,
    )

    client = HTTPClient("somehost")

    with pytest.raises(ValidationError):
        client._run_tesseract("apply", {"inputs": {"whoops": "whatever"}})


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
            [1.0, 2.0, 3.0],
        ),
        (
            {
                "shape": (3,),
                "dtype": "float32",
                "data": {"buffer": "AACAPwAAAEAAAEBA", "encoding": "base64"},
            },
            [1.0, 2.0, 3.0],
        ),
    ],
)
def test_decode_array(encoded, expected):
    decoded = _decode_array(encoded)
    assert np.all(decoded == expected)


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
