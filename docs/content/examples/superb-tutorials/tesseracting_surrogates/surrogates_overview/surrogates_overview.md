# Surrogates overview

Below, we assume that previously trained Mesh-Graph-Net (MGN) and MLP surrogates are available (*but note that we do not illustrate the training pipeline*). These were trained using ~10K samples corresponding to the above linear elasticity problem. With the purpose of highlighting the flexibility of <span class="product">Tesseracts</span> when it comes to languages, the surrogates are trained in different environments. The MGN surrogate is trained using `JAX` and predicts displacement fields ${\bf d}_x$ and ${\bf d}_y$ components given a load field. The MLP surrogate was trained using `Julia`'s Lux.jl package and predicts the von-Mises stress at each grid point given the same load field.


## [Julia Surrogate](julia_surrogate.md)
The Julia surrogate predicts the von-Mises stress given an input force loading.
```{figure} julia_surrogate.png
:alt: julia_surrogate
:width: 300px
```

## [JAX Surrogate](jax_surrogate.md)
The JAX surrogate predicts displacement given an input force loading.
```{figure} jax_surrogate.png
:alt: jax_surrogate
:width: 300px
```

```{toctree}
:maxdepth: 1
julia_surrogate.md
jax_surrogate.md
```
