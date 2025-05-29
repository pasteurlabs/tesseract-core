# The Julia MLP surrogate for the von-Mises stress

```{figure} julia_surrogate.png
:alt: julia_surrogate
:width: 400px
```

In contexts like optimization, uncertainty quantification, and sensitivity analysis, one may want to compute maps from problem parameters to quantities of interest. In the case of the linear elasticity problem under consideration, we are interested in the function that maps the ellipse parameter set into a quantity of interest that depends on the von Mises stress, e.g. the averege stress on a subset of the boundary. However, the ML surrogates are maps from the loading vector in the whole domain to the stress field, also in the whole domain. Thus, in this section, we show how to augment the ML surrogates with functions that map the ellipse parameters into the ML surrogate input (the loading), and the ML surrogate output (the stress field) into the quantity of interest, i.e. the average von Mises stress on the free boundaries. These functions, to be composed, are described next.

## Generate the Input Field
First, we define a differentiable mapping from user-defined input parameters, $p = \{ x_c, y_c, a, \theta \} $, to the loading field, $\mathbf{f} = \{f_x, f_y \}$ . Here, `SURF_AREA` corresponding to $A$ above is a constant. Note that although this map is by definition non-differentiable as it contains the indicator function that selects the ellipse area, with the purpose of obtaining fully differentiable maps, we smooth out the indicator function using a sigmoid approximation (see the implementation below).

```julia
function generate_field(x)
    xc, yc, axis_x, theta = x

    # all points in the grid
    axis_y = SURF_AREA / pi / axis_x

    # define tolerance for ellipse boundary
    tol = (1.1) ^ 2 - 1

    # convert to radians
    theta = deg2rad(theta)

    # make rotation matrix
    R = [ cos(theta) sin(theta); -sin(theta) cos(theta)]

    # translate and rotate
    translate = [xc yc]
    y = R * (coordinates .- translate)'
    y = y'

    # check if modified coordinate is inside the ellipse
    z = y[:,1].^2 ./ axis_x.^2 .+ y[:,2].^2 ./ axis_y.^2

    # we use a sigmoid to smoothly approximate the ellipse mapping
    scaled_value = 1.0 .- (1 ./ (1 .+ exp.(-10.0*(z .- (1.0 + tol)))))

    # true force values at grid points
    force_values = scaled_value .* force_value
    input_field = [force_values; force_values]
    return input_field
end
```

## Evaluate the Surrogate
Having obtained the loading vectors in the whole domain (i.e., the surrogate's input), the MLP surrogate ($\mathbf{f} \rightarrow \sigma;  \; \forall \; \mathbf{x} \in \Omega$) for the von-Mises stress at each point can be invoked.

 ```julia
function eval_surrogate(load_field)
    prediction = Lux.apply(model, load_field, trained_parameters, trained_states)[1]
    return prediction
end
```

## Evaluate the Surrogate Quantity of Interest
At this point, when we have access to the whole stress field, any related quantity of interest can be computed. Here, we compute the average of stress along the free boundaries ($\sigma \; \forall \; \mathbf{x} \in \Omega \rightarrow  \frac{1}{n}\sum_{y=1,\,x=1}\; \sigma$).
```julia
function calc_mean_stress_from_field(prediction)
    # average von-mises stress over x = 1 and y = 1
    prediction = reshape(prediction, 51, 51)
    boundary_stress = [prediction[:,end]; prediction[end,:]]
    return sum(boundary_stress) / length(boundary_stress)
end
```

## Generate End-to-End Map
Finally, all the functions described above can be composed to give us the end-to-end mapping from user-defined parameters to the quantity of interest ($p \rightarrow \frac{1}{n}\sum_{y=1,\,x=1}\; \sigma$ where $p = \{x_c, y_c, a, \theta\}$).

```julia
function eval_forward(x)
    load_field = generate_field(x)
    predicted_field = eval_surrogate(load_field)
    average_stress = calc_mean_stress_from_field(predicted_field)
    return average_stress
end
```
Since all of the above functions including the MLP surrogate for stress field are differentiable, the gradient of the quantity of interest with respect to the design parameters ($\frac{d\mathcal{L}}{dp}$ where $\mathcal{L} = \frac{1}{n}\sum_{y=1,\,x=1}\; \sigma$) can easily be obtained utilizing the autodiff capability of Julia.

```julia
function eval_gradient(x)
    return ForwardDiff.gradient(eval_forward, x)
end
```

> **_NOTE:_**  `eval_forward(x)` and `eval_gradient(x)` are important to define because they denote the end-to-end differentiable map that can be exposed as <span class="product">Tesseract</span> endpoints. Together, these functions allow us to obtain gradients of the quantity of interest (the mean stress) with respect to the ellipse parameters. Their use in optimization contexts is illustrated in a demostrated in a [separate demo](../../optimization_with_surrogates/linear_ellipse_optimization.md).
