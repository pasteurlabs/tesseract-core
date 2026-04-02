(pytorch-integration)=

# PyTorch Integration

Tesseract Core includes a PyTorch compatibility layer that lets you use any Tesseract as a **first-class differentiable operation** in PyTorch's autograd graph. This means a Tesseract — even one wrapping a Fortran solver, a JAX model, or a black-box simulator — can participate in `.backward()`, `torch.autograd.grad`, and forward-mode AD (`torch.autograd.forward_ad`) just like a native `torch.autograd.Function`.

This is the PyTorch counterpart to what [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) provides for JAX.

```{tip}
Install with `pip install tesseract-core[torch]` to pull in the PyTorch dependency.
```

## `apply_tesseract`

The entire integration is a single function: `apply_tesseract` from `tesseract_core.torch_compat`.

```python
from tesseract_core.torch_compat import apply_tesseract
```

It inspects the Tesseract's OpenAPI schema to determine which inputs and outputs are differentiable, then routes them through a `torch.autograd.Function` that:

- **Forward** — calls `tesseract.apply()`, converting tensors to NumPy and back.
- **Backward** — calls `tesseract.vector_jacobian_product()` for reverse-mode AD.
- **JVP** — calls `tesseract.jacobian_vector_product()` for forward-mode AD.

### Inputs

Pass a dict matching the Tesseract's input schema. For differentiable fields, provide `torch.Tensor` values to have them participate in autograd. Everything else (plain Python values, NumPy arrays) is treated as static:

```python
result = apply_tesseract(tesseract, {
    "x": x_tensor,                          # differentiable — tracked by autograd
    "A": np.eye(3, dtype=np.float32),        # static — not tracked
    "b": torch.zeros(3),                     # differentiable
})
```

### Outputs

Returns a nested dict matching the Tesseract's output schema. Differentiable array outputs come back as `torch.Tensor` (with `grad_fn` attached when inputs require grad); non-differentiable outputs are returned as-is.

## Reverse-mode AD (`.backward()`)

The most common case: compute a loss that involves a Tesseract, then call `.backward()` to get gradients.

```python
import torch
from tesseract_core import Tesseract
from tesseract_core.torch_compat import apply_tesseract

quadratic = Tesseract.from_image("quadratic")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
A = torch.eye(3, dtype=torch.float32, requires_grad=True)
b = torch.zeros(3, dtype=torch.float32, requires_grad=True)

result = apply_tesseract(quadratic, {"x": x, "A": A, "b": b})
y = result["y"]  # y has a grad_fn — autograd is tracking it

loss = y.sum()
loss.backward()  # dispatches to the Tesseract's VJP endpoint

print(x.grad)  # [2, 4, 6]  (for y = x², dy/dx = 2x)
print(b.grad)  # [1, 1, 1]
```

This also works with `torch.autograd.grad` for finer control:

```python
grad_y0, = torch.autograd.grad(y[0], x, retain_graph=True)
grad_y1, = torch.autograd.grad(y[1], x, retain_graph=True)
# Each gives one row of the Jacobian
```

## Forward-mode AD (`torch.autograd.forward_ad`)

Forward-mode is efficient when you have few inputs and many outputs — the natural choice for directional derivatives and building Jacobians column-by-column.

```python
import torch.autograd.forward_ad as fwAD

x = torch.tensor([1.0, 2.0, 3.0])
tangent = torch.tensor([1.0, 0.0, 0.0])  # direction e_0

with fwAD.dual_level():
    x_dual = fwAD.make_dual(x, tangent)
    result = apply_tesseract(quadratic, {"x": x_dual, "A": A, "b": b})
    primal, tangent_out = fwAD.unpack_dual(result["y"])

print(tangent_out)  # [2, 0, 0] — first column of the Jacobian
```

Build the full Jacobian column-by-column:

```python
J = torch.stack([
    fwAD.unpack_dual(
        apply_tesseract(quadratic, {
            "x": fwAD.make_dual(x, torch.eye(3)[i]),
            "A": A, "b": b,
        })["y"]
    )[1]
    for i in range(3)
], dim=1)
```

## Composing with neural networks

The real power is end-to-end differentiation through a Tesseract and a neural network together. Gradients flow from the loss, through the Tesseract's VJP endpoint, and into the network weights:

```python
import torch.nn as nn

net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 3))
target = torch.tensor([2.0, 3.0, 1.0])
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

for step in range(200):
    optimizer.zero_grad()
    x_pred = net(torch.randn(5))

    # Tesseract in the forward pass
    result = apply_tesseract(quadratic, {"x": x_pred, "A": A, "b": b})

    loss = ((result["y"] - target) ** 2).mean()
    loss.backward()  # gradients flow through the Tesseract into the network
    optimizer.step()
```

## How it works

```
                 forward                       backward (reverse-mode)
┌──────────┐   ──────►   ┌──────────────┐   ──────►   ┌──────────┐
│  PyTorch  │             │  Tesseract    │             │  PyTorch  │
│  tensors  │  tensors →  │  .apply()     │  grad →    │  grads    │
│  (with    │  numpy →    │               │  numpy →   │  (for     │
│  grad_fn) │  HTTP/local │  .vjp()       │  HTTP/local│  upstream)│
└──────────┘             └──────────────┘             └──────────┘

                 forward-mode AD
┌──────────┐             ┌──────────────┐             ┌──────────┐
│  dual     │  primal +   │  Tesseract    │  primal +   │  dual     │
│  tensors  │  tangent →  │  .apply()  +  │  tangent →  │  tensors  │
│           │  numpy →    │  .jvp()       │  numpy →    │           │
└──────────┘             └──────────────┘             └──────────┘
```

The Tesseract computes derivatives however it wants internally — analytical, JAX autodiff, finite differences — PyTorch doesn't need to know.

## Nested schemas

`apply_tesseract` handles nested input/output schemas using the same dotted-path convention described in {ref}`tr-autodiff`:

```python
result = apply_tesseract(mesh_tesseract, {
    "mesh": {"n_points": 3, "points": points_tensor}
})
result["statistics"]["barycenter"].sum().backward()
```

## Containerized and remote Tesseracts

The examples above work identically with containerized or remote Tesseracts — the only difference is that forward and backward calls go over HTTP:

```python
# Local development (no Docker)
t = Tesseract.from_tesseract_api("my_tesseract/tesseract_api.py")

# Containerized
t = Tesseract.from_image("my_tesseract")

# Remote
t = Tesseract.from_url("http://remote-host:8000")

# apply_tesseract works the same way with all three
result = apply_tesseract(t, {"x": x_tensor})
```

## See also

- {ref}`tr-autodiff` for background on differentiable programming in Tesseracts
- [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) for the equivalent JAX integration
- {doc}`/content/examples/building-blocks/gradient-fallbacks` for deriving missing gradient endpoints
