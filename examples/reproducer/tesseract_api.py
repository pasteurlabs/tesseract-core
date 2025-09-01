from concurrent.futures import ProcessPoolExecutor

import numpy as np
from pydantic import BaseModel


class InputSchema(BaseModel):
    pass


class OutputSchema(BaseModel):
    pass


# FIXME: if pool.submit uses a function defined in tesseract_api.py
# we get a pickling error: module tesseract_api not found.
def preprocess_fn(data_id: int):
    print(data_id, "processing")
    return data_id


def apply(inputs):
    data_ids = list(range(10))

    pool = ProcessPoolExecutor()
    futures = []

    for idx in data_ids:
        # this causes the pickling error
        # x = pool.submit(preprocess_fn, idx)
        x = pool.submit(np.identity, idx)
        futures.append(x)
        print(idx, "submitted")

    for f in futures:
        res = f.result()
        print(res, "done")

    return OutputSchema()
