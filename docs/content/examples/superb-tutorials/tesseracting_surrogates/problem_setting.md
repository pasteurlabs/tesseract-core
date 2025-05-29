# Problem Setting

## Linear-Elasticity Material with an Ellipse-Shaped Loading
The problem of interest is a simple proof of concept in the context of mechanics. Despite its simplicity, the linear-elasticity problem presented in this demo easily generalizes to more complex industrial problems such as surrogates for jet-engine compressors, heat exchangers, and HVAC ducts. Indeed, the pipeline for more complex problems is identical to the one reported here.

The pipeline presented in this demo answers the question of *how to package any trained differentiable ROM or ML surrogate in a standardized, universal container, i.e. in a <span class="product">Tesseract</span>*. Indeed, the steps below illustrate how building and using <span class="product">Tesseracts</span> is independent of the specific framework (or machine) the surrogates were built in.

Formally, the surrogates in this demo deal with a two-dimensional linear elasticity problem in the domain $\Omega=[0, 1]^2$ where an ellipse-shaped loading field is parametrized. The maps of interest, reproduced by the surrogates, are

- The map from the loading parameters to the von-Mises stress over the domain;
- The map from the loading parameters to the displacement vector field over the domain.

```{figure} surrogate_fields.png
:alt: linear-elasticity-ellipse
:width: 600px

<small>Surrogates predict 2D displacement and stress fields.</small>
```

The loading vector is constant and of positive components over an ellipse parametrized by:

- The length of one semi axis: $a$
- The angle of that axis with respect to the x axis: $\theta$
- The coordinates of the center: $(x_c,y_c)$

The ellipse parameters are detailed in the below figure:

```{figure} ellipse.png
:alt: linear-elasticity-ellipse
:width: 300px

<small>Ellipse-shaped loading vector.</small>
```

The length of the second semi axis is defined such that the area of the ellipse is constant, fixed (i.e. not a parameter) and equal to $A$.
The value of the loading vector $ \mathbf{f} = \{f_x,f_y\} \in \mathbb{R^2} \; \forall \; \mathbf{x} \in \Omega $ is such that the total load is constant for all ellipse areas (in fact, $f_x$ and $f_y$ are both constants).
The boundary conditions are given by a Homogeneous Dirichlet for the displacement on $y=0, x=0$ and a traction free condition on $y=1, x=1$.

Given an example ellipse-shaped load vector (below on the left), the two surrogates predict the von-Mises stress (middle) and displacement components (right) respectively.
```{figure} ellipse_in_out.png
:alt: surrogate_in_out
:width: 900px
```

This demo shows how each surrogate is packaged and deployed as a stand-alone <span class="product">Tesseracts</span> that accepts the ellipse parameters ($x_c$,$y_c$,$a$,$\theta$), generates the corresponding load field (using a user-defined differentiable map), evaluates the surrogate, and finally evaluates an output quantity of interest. In this case, the quantities of interest are the average stress and displacement on the domain's free boundaries. Each <span class="product">Tesseracts</span> also implements the `jacobian` for these quantities which can be used to evaluate their gradient with respect to the ellipse parameters (they are denoted as differentiable quantities in the <span class="product">Tesseracts</span>). Each <span class="product">Tesseracts</span> can also optionally return the associated load field, displacement fields, and stress field for visualization purposes.
