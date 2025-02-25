import numpy as np

from tesseract_core import Tesseract

tess = Tesseract(url="http://localhost:8000")
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

print(tess.jacobian({"a": a, "b": b}, jac_inputs=["a"], jac_outputs=["result"]))
