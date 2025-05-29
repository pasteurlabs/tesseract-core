# The JAX surrogate (Mesh Graph Net) for the displacement

```{figure} jax_surrogate.png
:alt: jax_surrogate
:width: 400px
```
Just like for the Julia surrogate, in this section we describe the differentiable functions that need to be componsed to obtain a single differentiable function that maps the ellipse parameters into a quantity of interest. In this case, we have a JAX surrogate that maps the loading vector in the whole domain into the displacement vector field, also in the whole domain. We augment this ML surrogate with functions that map the ellipse parameters into the ML surrogate input (the loading), and the ML surrogate output (the displacement field) into the quantity of interest, i.e. the average magnitude of the displacement on the free boundaries. These functions, to be composed, are described next.

## Generate the Input Field
The JAX based module is very similar to the one presented above for Julia. Here, the surrogate itself provides a map from loading vector field to the displacement field ($\mathbf{f} \rightarrow \mathbf{d} ; \; \forall \; \mathbf{x} \in \Omega \quad \text{where}\; \mathbf{f}= \{f_x, f_y\}, \mathbf{d}=\{d_x, d_y\}$). Just like above, this module is also a composition of three mappings. The first function, i.e., `generate_field`, is almost identical to the one for [Julia](julia_surrogate.md) and is therefore not shown here.

## Evaluate the Surrogate
The second function, i.e., `eval_surrogate`, maps the input loading field into the displacement field ($\mathbf{f} \rightarrow \mathbf{d}$) over the whole domain.

```Python
def eval_surrogate(load_graph):
    result = GraphsTuple(**exported_fn(loaded_params, load_graph._asdict()))
    return result
```

## Evaluate the Surrogate Quantity of Interest
The third function computes the quantity of interest starting from the stress field. Here, we compute the mean displacement on the free boundaries. The corresponding `calc_mean_displacement_from_field` function is such that:  $\mathbf{d} \; \forall \; \mathbf{x} \in \Omega\rightarrow \frac{1}{n}\sum_{y=1,\,x=1}\; \| {\bf d}(x_c,y_c,a,\theta)  \|_2$

```Python
# Objective: total displacement on free boundaries (x=1, y=1)
def calc_mean_displacement_from_field(graph):
    predicted_ux = graph.nodes[edge_coords, 0]
    predicted_uy = graph.nodes[edge_coords, 1]
    total_displacement = jnp.sqrt(predicted_ux**2 + predicted_uy**2)
    return jnp.sum(total_displacement) / len(total_displacement)
```

## Generate End-to-End Map
The final end-to-end mapping from user-defined ellipse parameters to the quantity of interest is then given by$p \rightarrow \frac{1}{n}\sum_{y=1,\,x=1}\; \| {\bf d}(x_c,y_c,a,\theta)  \|_2$ and implemented as follows.

```Python
def calc_mean_displacement_from_input(x):
    load_graph = generate_field(x)
    predicted_graph = eval_surrogate(load_graph)
    mean_displacement = calc_mean_displacement_from_field(predicted_graph)
    return mean_displacement
```
JAX's `jit` and `grad` functionalities are utilized for the function evaluations and to compute the gradient of the mean stress with respect to the input design parameters respectively.

```Python
# JIT compile
jit_generate_field = jit(generate_field)
jit_eval_surrogate = jit(eval_surrogate)
jit_calc_mean_displacement_from_field = jit(calc_mean_displacement_from_field)
jit_calc_mean_displacement_from_input = jit(calc_mean_displacement_from_input)
jit_grad_mean_displacement_from_input = jit(grad(jit_calc_mean_displacement_from_input))
```

> **_NOTE:_**  Just like with the [Julia surrogate](julia_surrogate.md), we define the end-to-end map here to be exposed to a <span class="product">Tesseract</span> for gradient calculation. This capability is used in optimization contexts as our [separate demo](../../optimization_with_surrogates/linear_ellipse_optimization.md) demonstrates.
