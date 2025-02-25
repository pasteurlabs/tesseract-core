(tr-runtime-api)=
# `tesseract_runtime` Python API

The `tesseract_runtime` Python module contains some useful utilities to create and extend `tesseract_api.py` files. It is available in all Tesseract container images.

```{eval-rst}
.. automodule:: tesseract_core.runtime
   :members:
```

```{eval-rst}
.. py:class:: Float16

   Float16 scalar array. Equivalent to ``Array[(), Float16]``.

.. py:class:: Float32

   Float32 scalar array. Equivalent to ``Array[(), Float32]``.

.. py:class:: Float64

   Float64 scalar array. Equivalent to ``Array[(), Float64]``.

.. py:class:: Int8

   Int8 scalar array. Equivalent to ``Array[(), Int8]``.

.. py:class:: Int16

   Int16 scalar array. Equivalent to ``Array[(), Int16]``.

.. py:class:: Int32

   Int32 scalar array. Equivalent to ``Array[(), Int32]``.

.. py:class:: Int64

   Int64 scalar array. Equivalent to ``Array[(), Int64]``.

.. py:class:: UInt8

   UInt8 scalar array. Equivalent to ``Array[(), UInt8]``.

.. py:class:: UInt16

   UInt16 scalar array. Equivalent to ``Array[(), UInt16]``.

.. py:class:: UInt32

   UInt32 scalar array. Equivalent to ``Array[(), UInt32]``.

.. py:class:: UInt64

   UInt64 scalar array. Equivalent to ``Array[(), UInt64]``.

.. py:class:: Bool

   Bool scalar array. Equivalent to ``Array[(), Bool]``.

.. py:class:: Complex64

   Complex64 scalar array. Equivalent to ``Array[(), Complex64]``.

.. py:class:: Complex128

   Complex128 scalar array. Equivalent to ``Array[(), Complex128]``.
```


## `tesseract_core.runtime.experimental`

The experimental namespace includes features that are under active development
and may not be fully stable. Use these at your own risk, as APIs, behaviors, and
implementations can change without notice, or be removed entirely in future
releases.

```{eval-rst}
.. automodule:: tesseract_core.runtime.experimental
   :members:
   :exclude-members: PydanticLazySequenceAnnotation
```
