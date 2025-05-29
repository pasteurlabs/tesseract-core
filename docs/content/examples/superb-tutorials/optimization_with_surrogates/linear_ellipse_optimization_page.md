# Using differentiable containers in optimization frameworks
This demo shows how to use containerized SciML components, namely ML surrogates, in workflows that require (continuous) gradient information such as optimization for engineering design. While the latter is the focus of this demo, other relevant contexts include inverse problem solving and design of experiment. At Pasteur, a SciML component is wrapped with a Python API into a container, called <span class="product">Tesseract</span>, that has standardized interfaces and embedded automatic differentiation.


This demo illustrates how to use Tesseracted Surrogates within a gradient-based optimization framework with the purpose of (ultimately) solving a multi-objective optimization problem in the context of mechanics applications. In particular, this demo deals with a design problem where the objective is to find the best loading conditions such that a displacement metric is maximized and a stress metric is minimized. To this end, the optimization framework relies on two ML surrogates, one for the displacement and one for the stress, that have been Tesseracted after being implemented and trained in `JAX` and `julia` respectively. As such, this demo illustrates how SciML components built independently and in different environments, can seamlessly interact with each other and with external tools once packaged into <span class="product">Tesseracts</span>.

By Tesseracting surrogates, this demo showcases how the complexities of optimization workflows, such as interoperability, can be easily circumvented. In fact, <span class="product">Tesseracts</span> allow for

- Freedom of implementation: components can be implemented in any framework / language;
- Separation of concerns: the job of the R&D staff ends when the surrogate respects the <span class="product">Tesseract</span>'s specs;
- Heterogeneous computing: the surrogates do not need to be executed on the same hardware to work together in a shared pipeline;
- Universal deployment: deployment of components consists of applying a fixed containerization recipe and parsing the exposed API endpoint.

# Problem Setting

## Optimal Loading for a Linear-Elasticity Material using an Ellipse-Shaped Force
The problem of interest is a simple proof of concept in the context of mechanics. Despite its simplicity, the linear-elasticity optimization problem presented in this demo easily generalizes to more complex industrial problems; indeed, the pipeline for more complex problems is identical to the one reported here. As such, the same pipeline can be applied to the design of, e.g.,

- jet-engine compressors;
- heat exchangers;
- HVAC ducts;
- and many more scenarios.

In general, the solution presented in this demo answers the question of *how to use any differentiable ROM or ML surrogate within an optimization framework without worrying about the actual emulator-optimizer interactions* as these are automatically taken care of by the <span class="product">Tesseracts</span>.

Formally, the optimization problem in this demo involves finding the optimal parameters to determine the shape and location of a loading field in the two-dimensional domain $\Omega=[0,1]^2$.
The ellipse-shaped load is applied to a linear-elasticity material which induces stress and displacement fields like the following:

```{figure} surrogate_fields.png
:alt: linear-elasticity-ellipse
:width: 600px
```

The loading vector is constant and of positive components over an ellipse parametrized by:

- The length of one semi axis: $a$
- The angle of that axis with respect to the x axis: $\theta$
- The coordinates of the center: $(x_c,y_c)$

The ellipse parameters are detailed in the below figure:

```{figure} ./ellipse.png
:alt: linear-elasticity-ellipse
:width: 300px

<small>Ellipse-shaped force field parameters.</small>
```

The length of the second semi axis is defined such that the area of the ellipse is constant and equal to $A$, the latter is fixed and not part of the parameter set.
The value of the loading vector $\{f_x,f_y\}$ is such that the total load is constant for all ellipse areas (in fact, $f_x$ and $f_y$ are both constants).
The boundary conditions are given by a Homogeneous Dirichlet for the displacement on $y=0, x=0$ and a traction free condition on $y=1, x=1$.

The optimization problem consists of two objectives. First, we seek to maximize the magnitude of the average displacement on the traction-free boundary. This is given by the following objective function:
$\Big\{\; \max_{x_c,y_c,a,\theta}\; \sum_{y=1,\,x=1}\; \| {\bf d}(x_c,y_c,a,\theta)  \|_2 \Big\}$
where $\bf d(\cdot)$ returns the norm of the average displacement.

We additionally seek to minimize the average von-Mises stress on the same traction-free boundary. This is given by the following term:
$ \Big\{\; \min_{x_c,y_c,a,\theta}\; \sum_{y=1,\,x=1}\; \sigma(x_c,y_c,a,\theta)  \Big\} $
where $\sigma(\cdot)$ returns the von-Mises stress.

In this context, we need simulators or emulators that map the ellipse parameters into average stress and displacement. In this demo we make use of ML surrogates and assume that we previously trained Mesh-Graph-Net (MGN) and MLP surrogates using ~10K samples corresponding to the above linear-elasticity problem. Specifically, the MGN surrogate was trained using JAX and predicts displacement fields ${\bf d}_x$ and ${\bf d}_y$ components given a load field. The MLP surrogate was trained using Julia's Lux.jl package and predicts the von-Mises stress at each grid point given the same load field. Note that the surrogates themselves do not reproduce the needed map and need to be augmented with a differentiable function that maps the ellipse parameters into the loading and a differentiable function that maps the computed stress and displacement into their averages. This process is explained in the previous demo.

Then, each surrogate is packaged and deployed as a stand-alone <span class="product">Tesseract</span>. Each <span class="product">Tesseract</span> accepts the ellipse parameters ($x_c$,$y_c$,$a$,$\theta$), generates the corresponding load field (using the differentiable map mentioned above), evaluates the surrogate, and finally evaluates the output quantity we wish to optimize (i.e. the <span class="product">Tesseract</span> ultimately returns the mean displacement and mean stress over the free boundaries). Each <span class="product">Tesseract</span> also implements the `jacobian` for these quantities which we can use to evaluate their gradient with respect to the ellipse parameters (they are denoted as differentiable quantities in the <span class="product">Tesseract</span>). Each <span class="product">Tesseract</span> can also optionally return the associated load field, displacement fields, and stress field for visualization purposes.

```{figure} run_optimization.png
:alt: linear-elasticity-ellipse
:width: 900px

<small>Tesseract-based optimization workflow.</small>
```

# Optimization setup in Julia using Plasmosis
We use Julia to develop the optimization solution. We show how to define a simple <span class="product">Tesseract</span> interface and how to use the Pasteur <span class="product">Plasmosis</span> engine to perform optimization with each <span class="product">Tesseract</span> both in isolation and in a coupled multi-objective setting. We require the following Julia packages to assemble our optimization solution.

```julia
# for tesseract interface
using HTTP
using JSON
using Base64

# for optimization
using JuMP
using Plasmosis
using Ipopt
import MultiObjectiveAlgorithms as MOA

# for visualization
using CairoMakie
```

# Creating Tesseract Interfaces
To interface with <span class="product">Tesseracts</span>, we define wrapper functions that call their endpoints and unpack their results. <span class="product">Tesseracts</span> can optionally be instructed to return encoded buffers as opposed to numerical data (this saves on communication bandwidth). When using encoded buffers, it is necessary to decode and unpack the result.

```julia
function decode_result(result, dtype)
    if result["data"]["encoding"] == "raw"
        # if results are not encoded, just unpack them
        return result["data"]["buffer"]
    else
        # decode the result
        decoded_result = Base64.base64decode(result["data"]["buffer"])
        debuffered_result = reinterpret(dtype, decoded_result)
        return debuffered_result
    end
end
```

## Define the Displacement Tesseract
We assume the displacement <span class="product">Tesseract</span> is served locally on port 8001 (in practice it could be hosted at any web location). For our interface we define `eval_forward_displacement` to evaluate the mean displacement (i.e. the objective of the optimization problem) on the free boundary corresponding to certain ellipse parameters and we use `eval_fields_displacement` as a convenience function that returns the force and/or displacement fields. Both of these functions simply wrap the <span class="product">Tesseract</span> `apply` endpoint. We also define `eval_gradient_displacement` to obtain the gradient of the average displacement. Note that we call the `jacobian` endpoint to obtain the objective gradient (the <span class="product">Tesseract</span> jacobian is simply a vector in this case).

```julia
URL_DISPLACEMENT="http://localhost:32783"

# call tesseract for mean displacement on free boundaries
function eval_forward_displacement(x::Vector{Float64})
    url = "$(URL_DISPLACEMENT)/apply"
    headers = ["Content-Type" => "application/json"]
    data = JSON.json(
        Dict(
            "inputs" =>  Dict(
                "xc" => x[1],
                "yc" => x[2],
                "axis_x"=> x[3],
                "theta"=>x[4],
                "return_force_components"=>false,
                "return_displacement_components"=>false
            ),
        )
    )
    response = HTTP.post(url, headers=headers, body=data)
    response_json = JSON.parse(String(response.body))
    debuffered = decode_result(response_json["mean_displacement"], Float64)[1]
    return debuffered
end

# call tesseract for force and/or displacement fields on entire grid
function eval_fields_displacement(x::Vector{Float64})
    url = "$(URL_DISPLACEMENT)/apply"
    headers = ["Content-Type" => "application/json"]
    data = JSON.json(
        Dict(
            "inputs" =>  Dict(
                "xc" => x[1],
                "yc" => x[2],
                "axis_x"=> x[3],
                "theta"=>x[4],
                "return_force_components"=>true,
                "return_displacement_components"=>true
            ),
        )
    )
    response = HTTP.post(url, headers=headers, body=data)
    response_json = JSON.parse(String(response.body))
    result_fx = decode_result(response_json["fx"], Float32)
    result_fy = decode_result(response_json["fy"], Float32)
    result_ux = decode_result(response_json["ux"], Float32)
    result_uy = decode_result(response_json["uy"], Float32)

    return Dict(:fx=>result_fx, :fy=>result_fy, :ux=>result_ux, :uy=>result_uy)
end

# call tesseract for jacobian evaluation
function eval_gradient_displacement(x::Vector{Float64})
    url = "$(URL_DISPLACEMENT)/jacobian"
    headers = ["Content-Type" => "application/json"]
    data = JSON.json(
        Dict(
            "inputs" =>  Dict("xc" => x[1], "yc" => x[2], "axis_x"=>x[3], "theta"=>x[4]),
            "jac_inputs" => ["xc", "yc", "axis_x", "theta"],
            "jac_outputs" => ["mean_displacement"]
        )
    )
    response = HTTP.post(url, headers=headers, body=data)
    response_json = JSON.parse(String(response.body))

    # return ordered gradient
    order = ["xc", "yc", "axis_x", "theta"]
    gradient = zeros(length(order))
    for i = 1:length(order)
        input = order[i]
        debuffered = decode_result(response_json[input]["mean_displacement"], Float64)[1]
        gradient[i] = debuffered
    end
    return gradient
end
```

## Evaluate the Displacement Tesseract
To test that the displacement <span class="product">Tesseract</span> works as expected, we pass in a set example ellipse parameters and check the results of the wrapper functions.

```julia
# sample ellipse parameters
x_test =[0.5, 0.5, 0.1, 45.0]

# evaluate mean displacement, mean displacement gradient, and resulting fields
result = eval_forward_displacement(x_test)
result_jac = eval_gradient_displacement(x_test)
result_fields = eval_fields_displacement(x_test)

println("forward result: ", result)
println("gradient result: ", result_jac)
println("result fields: ", keys(result_fields))
```

As expected, we calculate the mean magnitude of displacement and its gradient with respect to the ellipse parameters. We also calculate the full field solution, but elect not to print the full numerical output.
```
forward result: 0.13451500236988068
gradient result: [0.1309431493282318, 0.2320975661277771, 0.026332035660743713, -9.480804146733135e-5]
result fields: [:fy, :uy, :fx, :ux]
```

## Define the Stress Tesseract
Now we define an interface for the stress <span class="product">Tesseract</span> just like we did for the displacement <span class="product">Tesseract</span> above. We define wrapper functions to evaluate the `apply` and `jacobian` endpoints and unpack the results we are interested in. This <span class="product">Tesseract</span> is used to calculate the mean von-Mises stress on the free boundaries (and gradient of mean stress) as well as the field values for the entire grid.

```julia
URL_STRESS="http://localhost:32782"

# call tesseract for mean stress on free boundary
function eval_forward_stress(x::Vector{Float64})
    url = "$(URL_STRESS)/apply"
    headers = ["Content-Type" => "application/json"]
    data = JSON.json(
        Dict(
            "inputs" =>  Dict(
                "xc" => x[1],
                "yc" => x[2],
                "axis_x"=> x[3],
                "theta"=>x[4],
                "return_force_components"=>false,
                "return_displacement_components"=>false
            ),
        )
    )
    response = HTTP.post(url, headers=headers, body=data)
    response_json = JSON.parse(String(response.body))
    debuffered = decode_result(response_json["mean_stress"], Float64)[1]
    return debuffered
end

# call tesseract for force and/or von-Mises stress fields on entire grid
function eval_fields_stress(x::Vector{Float64})
    url = "$(URL_STRESS)/apply"
    headers = ["Content-Type" => "application/json"]
    data = JSON.json(
        Dict(
            "inputs" =>  Dict(
                "xc" => x[1],
                "yc" => x[2],
                "axis_x"=> x[3],
                "theta"=>x[4],
                "return_force_components"=>true,
                "return_stress_components"=>true
            ),
        )
    )
    response = HTTP.post(url, headers=headers, body=data)
    response_json = JSON.parse(String(response.body))
    result_fx = decode_result(response_json["fx"], Float64)
    result_fy = decode_result(response_json["fy"], Float64)
    result_s = decode_result(response_json["s"], Float64)

    return Dict(:fx=>result_fx, :fy=>result_fy, :s=>result_s)
end

# call tesseract for gradient of mean stress on boundary
function eval_gradient_stress(x::Vector{Float64})
    url = "$(URL_STRESS)/jacobian"
    headers = ["Content-Type" => "application/json"]
    data = JSON.json(
        Dict(
            "inputs" =>  Dict("xc" => x[1], "yc" => x[2], "axis_x"=>x[3], "theta"=>x[4]),
            "jac_inputs" => ["xc", "yc", "axis_x", "theta"],
            "jac_outputs" => ["mean_stress"]
        )
    )
    response = HTTP.post(url, headers=headers, body=data)
    response_json = JSON.parse(String(response.body))

    # return ordered gradient
    order = ["xc", "yc", "axis_x", "theta"]
    gradient = zeros(length(order))
    for i = 1:length(order)
        input = order[i]
        debuffered = decode_result(response_json[input]["mean_stress"], Float64)[1]
        gradient[i] = debuffered
    end
    return gradient
end
```

## Evaluate the Stress Tesseract
As we did for the displacement <span class="product">Tesseract</span>, we check the stress <span class="product">Tesseract</span> output.

```julia
x_test =[0.5, 0.5, 0.1, 45.0]
result = eval_forward_stress(x_test)
result_jac = eval_gradient_stress(x_test)
result_fields = eval_fields_stress(x_test)

println("forward result: ", result)
println("gradient result: ", result_jac)
println("result fields: ", keys(result_fields))
```

As expected, we obtain the mean von-Mises stress and gradient. We also elect not to print the full fields, but note that they were returned.
```
forward result: 18167.72491967325
gradient result: [43717.17432091829, 31464.92046612833, 15602.167564557289, -0.8033427390813097]
result fields: [:fy, :s, :fx]
```

# Plot the Solution for a Circle
To help develop a sense for the problem, we select a parameter set that corresponds to a circular force at the center of the domain and visualize the force, stress, and displacement fields. Both stress and displacement fields
are returned from the corresponding <span class="product">Tesseracts</span>.

```julia
x_circle = [0.5, 0.5, 0.113, 90.0]

fontsize_theme = Theme(fontsize = 21)
set_theme!(fontsize_theme)

field_values_displacement = eval_fields_displacement(x_circle);
field_values_stress = eval_fields_stress(x_circle)

input_plot_fx = reshape(field_values_displacement[:fx], 51, 51)
input_plot_fy = reshape(field_values_displacement[:fy], 51, 51)
prediction_plot_ux = reshape(field_values_displacement[:ux], 51, 51)
prediction_plot_uy = reshape(field_values_displacement[:uy], 51, 51)
prediction_plot_stress = reshape(field_values_stress[:s], 51, 51)

# calculate the displacement magnitude given the components
# note that we use permutedims on displacement because numpy is row major
displacement_norm = sqrt.( (field_values_displacement[:ux].^2 + field_values_displacement[:uy].^2) )
prediction_plot_d = permutedims(reshape(displacement_norm, 51, 51))

fig = Figure(size = (1200,300))

ax1 = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y", title="Force")
hm1 = heatmap!(ax1, input_plot_fx)
Colorbar(fig[1,2], hm1)

ax3 = CairoMakie.Axis(fig[1, 3]; xlabel="x", ylabel="y", title="Von-Mises Stress")
hm3 = heatmap!(ax3, prediction_plot_stress)
Colorbar(fig[1,4], hm3)

ax4 = CairoMakie.Axis(fig[1,5]; xlabel="x", ylabel="y", title="Norm Displacement")
hm4 = heatmap!(ax4, prediction_plot_d)
Colorbar(fig[1,6], hm4)
```

We observe the expected circular force which induces circular displacement and stress fields.
```{figure} circle_fields.png
:alt: linear-elasticity-circle
:width: 1200px

<small>Force, stress, and displacement fields given a circular input ellipse parameterization.</small>
```

# Maximize Average Displacement
To begin our optimization implementation, we first seek to only maximize the average magnitude of displacement and, to this end, we use the displacement <span class="product">Tesseract</span>.

## Formulate Optimization Problem to Maximize the Average Magnitude of Displacement Components
Remember that the displacement <span class="product">Tesseract</span> returns the average displacement magnitude on the free boundary (i.e. the displacement on $x=1$ and $y=1$) as well as the gradient of this quantity (using the jacobian endpoint). As our optimizer ultimately needs evaluator functions that return an objective function value and its gradient, we utilize our wrapper functions defined above.
We use Plasmosis.jl (which builds on JuMP.jl) to formulate the actual optimization problem given the surrogate information (e.g. variable bounds) and the evaluator functions. As Plasmosis.jl is built on
an optimization framework (JuMP.jl), it will formulate the total objective function and gradient (using JuMP.jl internal AD) using the provided wrapper functions.

Here we define a convenience function to build the displacement optimization problem.

```julia
function build_displacement_model()
    model = Model()

    # generate definition
    lower_bounds = [0.335, 0.335, 0.0386, 0.0234]
    upper_bounds = [0.664, 0.664, 0.329, 179.98]
    n_inputs = 4
    n_outputs = 1
    surrogate_definition = GenericSurrogate(n_inputs, n_outputs, lower_bounds, upper_bounds)

    # create evaluator
    f_eval_forward_displacement(x::Vector{Float64}) = eval_forward_displacement(x)
    f_eval_gradient_displacement(x::Vector{Float64}) = eval_gradient_displacement(x)
    surrogate_evaluator = SurrogateEvaluator(f_eval_forward_displacement, f_eval_gradient_displacement)

    # build optimization formulation
    build_formulation(
        model,
        ComposedFormulation(),
        surrogate_definition,
        surrogate_evaluator;
        name=:displacement
    )

    # set initial value
    x0 = [0.5, 0.5, 0.113, 45.0]
    set_start_value.(all_variables(model), x0)

    # minimize the max stress
    mean_displacement = model.ext[:displacement][:outputs][1]
    @objective(model, Max, mean_displacement)

    return model
end
```

## Run the Optimizer
We use Ipopt to solve the optimization problem. Since we are not providing an exact Hessian, we select the LBFGS algorithm. We choose the centered circle as the initial guess for the optimal solution.

```julia
model_displacement = build_displacement_model()
set_optimizer(model_displacement, Ipopt.Optimizer)
set_optimizer_attribute(model_displacement, "hessian_approximation", "limited-memory")
set_optimizer_attribute(model_displacement, "print_level", 5)
set_optimizer_attribute(model_displacement, "tol", 1e-3)
set_optimizer_attribute(model_displacement, "max_iter", 100)
JuMP.optimize!(model_displacement)
```
```
This is Ipopt version 3.14.16, running with linear solver MUMPS 5.7.3.

Number of nonzeros in equality constraint Jacobian...:        0
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:        0

Total number of variables............................:        4
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        4
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.3479525e-01 0.00e+00 1.03e+00   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.3955873e-01 0.00e+00 7.88e-01  -0.7 7.78e-02    -  9.91e-01 5.00e-01f  2
   2  1.4089131e-01 0.00e+00 3.60e-01  -2.0 8.01e-03    -  9.97e-01 1.00e+00f  1
   3  1.6166382e-01 0.00e+00 3.08e-01  -2.5 4.49e-02    -  1.00e+00 1.00e+00f  1
   4  1.8585198e-01 0.00e+00 1.73e-01  -3.1 7.36e-02    -  1.00e+00 9.92e-01f  1
   5  1.8641870e-01 0.00e+00 5.37e-01  -4.4 2.93e-03    -  1.00e+00 4.59e-01f  1
   6  2.0452875e-01 0.00e+00 6.62e-01  -2.4 4.35e-01    -  5.77e-01 3.95e-01f  1
   7  2.0479538e-01 0.00e+00 9.65e-01  -2.8 2.64e-01    -  1.00e+00 4.53e-03f  3
   8  1.2144028e-01 0.00e+00 1.49e+00  -1.1 3.59e-01    -  9.62e-01 8.61e-01f  1
   9  1.4014521e-01 0.00e+00 6.54e-01  -1.2 2.15e-01    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  1.5179922e-01 0.00e+00 2.99e-01  -1.2 9.69e-02    -  1.00e+00 1.00e+00f  1
  11  1.6035944e-01 0.00e+00 1.39e-01  -1.9 2.54e-02    -  1.00e+00 1.00e+00f  1
  12  1.6406503e-01 0.00e+00 1.38e+00  -1.9 1.90e-02    -  1.00e+00 1.00e+00f  1
  13  1.8707672e-01 0.00e+00 3.51e-01  -1.9 9.14e-02    -  1.00e+00 1.00e+00f  1
  14  1.9372366e-01 0.00e+00 2.76e-01  -1.9 2.20e-01    -  9.44e-01 2.74e-01f  2
  15  1.8928078e-01 0.00e+00 1.07e+00  -1.9 3.62e-02    -  1.00e+00 1.00e+00f  1
  16  1.8951359e-01 0.00e+00 3.98e-01  -1.9 6.99e-02    -  6.51e-01 1.96e-02f  6
  17  1.8583426e-01 0.00e+00 2.57e-01  -1.9 1.13e-02    -  1.00e+00 1.00e+00f  1
  18  1.8675648e-01 0.00e+00 2.66e-01  -1.9 4.83e-03    -  1.00e+00 1.00e+00f  1
  19  1.8780954e-01 0.00e+00 9.60e-02  -1.9 3.77e-03    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  20  1.8873060e-01 0.00e+00 3.14e-01  -2.9 4.00e-03    -  1.00e+00 1.00e+00f  1
  21  1.9247115e-01 0.00e+00 2.87e+00  -2.9 1.07e-02    -  1.00e+00 1.00e+00f  1
  22  2.0425883e-01 0.00e+00 1.91e-01  -2.9 3.63e-02    -  1.00e+00 9.54e-01f  1
  23  2.0600249e-01 0.00e+00 5.49e-01  -2.9 6.26e-02    -  1.00e+00 1.79e-01f  2
  24  1.2981348e-01 0.00e+00 8.24e-01  -1.5 2.28e-01    -  9.55e-01 1.00e+00f  1
  25  1.5267931e-01 0.00e+00 3.66e-01  -1.5 9.69e-02    -  1.00e+00 1.00e+00f  1
  26  1.5032741e-01 0.00e+00 2.16e-01  -1.5 2.61e-02    -  1.00e+00 1.00e+00f  1
  27  1.5569420e-01 0.00e+00 2.81e-01  -2.3 2.53e-02    -  1.00e+00 1.00e+00f  1
  28  1.9502480e-01 0.00e+00 7.35e-01  -2.3 8.15e-02    -  1.00e+00 1.00e+00f  1
  29  1.8804508e-01 0.00e+00 3.49e-01  -2.3 2.55e-02    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  30  1.8948166e-01 0.00e+00 3.56e-01  -2.3 2.75e-02    -  1.00e+00 1.00e+00f  1
  31  1.9682877e-01 0.00e+00 1.21e-01  -2.3 1.36e-01    -  5.91e-01 2.63e-01f  2
  32  1.9850300e-01 0.00e+00 1.44e-01  -2.3 6.29e-03    -  1.00e+00 1.00e+00f  1
  33  2.0069736e-01 0.00e+00 2.82e-01  -2.3 4.91e-02    -  1.00e+00 5.00e-01f  2
  34  2.0031443e-01 0.00e+00 1.41e-01  -2.3 1.90e-02    -  1.00e+00 5.00e-01f  2
  35  2.0162746e-01 0.00e+00 1.24e-01  -2.3 6.79e-03    -  1.00e+00 5.00e-01f  2
  36  2.0210259e-01 0.00e+00 2.32e-01  -2.3 4.90e-03    -  1.00e+00 2.50e-01f  3
  37  2.0217937e-01 0.00e+00 2.78e-01  -2.3 7.84e-03    -  1.00e+00 6.25e-02f  5
  38  2.0152318e-01 0.00e+00 1.58e-01  -2.3 1.95e-03    -  1.00e+00 1.00e+00f  1
  39  2.0157348e-01 0.00e+00 7.28e-02  -2.3 3.23e-04    -  1.00e+00 1.00e+00f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  40  2.0167218e-01 0.00e+00 3.90e-02  -2.3 3.85e-04    -  1.00e+00 1.00e+00f  1
  41  2.0276132e-01 0.00e+00 1.15e-01  -3.5 3.46e-03    -  1.00e+00 1.00e+00f  1
  42  2.0318525e-01 0.00e+00 1.04e-01  -3.5 1.94e-03    -  1.00e+00 1.00e+00f  1
  43  2.0862460e-01 0.00e+00 1.31e-01  -3.5 3.48e-02    -  1.00e+00 4.59e-01f  1
  44  2.0834917e-01 0.00e+00 2.15e-01  -3.7 2.78e-03    -  1.00e+00 1.00e+00f  1
  45  2.0855373e-01 0.00e+00 6.05e-02  -3.8 1.23e-03    -  1.00e+00 1.00e+00f  1
  46  2.0857111e-01 0.00e+00 5.54e-02  -3.8 1.29e-04    -  1.00e+00 1.00e+00f  1
  47  2.0871988e-01 0.00e+00 5.68e-02  -3.8 1.17e-03    -  1.00e+00 1.00e+00f  1
  48  2.0929082e-01 0.00e+00 5.25e-02  -5.4 3.41e-03    -  1.00e+00 6.05e-01f  1
  49  2.0930074e-01 0.00e+00 3.15e-02  -7.3 8.92e-04    -  1.00e+00 3.76e-01f  2
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  50  2.0930114e-01 0.00e+00 1.79e-02  -7.5 1.79e-02    -  1.00e+00 7.36e-03f  8
  51  2.0930538e-01 0.00e+00 2.86e-03  -8.4 4.96e-05    -  1.00e+00 9.53e-01f  1
  52  2.0930558e-01 0.00e+00 1.23e-04  -9.6 4.06e-06    -  1.00e+00 9.78e-01f  1

Number of Iterations....: 52

                                   (scaled)                 (unscaled)
Objective...............:  -2.0930558443069458e-01    2.0930558443069458e-01
Dual infeasibility......:   1.2308047540654510e-04    1.2308047540654510e-04
Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
Variable bound violation:   6.6732854797635355e-09    6.6732854797635355e-09
Complementarity.........:   6.8379677343541802e-10    6.8379677343541802e-10
Overall NLP error.......:   1.2308047540654510e-04    1.2308047540654510e-04


Number of objective function evaluations             = 133
Number of objective gradient evaluations             = 53
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 0
Total seconds in IPOPT                               = 228.539

EXIT: Optimal Solution Found.
```
The optimizer spends the majority of its time evaluating the objective gradient. In this example our displacement <span class="product">Tesseract</span> is using JAX CPU to evaluate the gradient of a fairly deep MGN.
It is possible (and advisable) to implement a GPU-based gradient evaluation for this surrogate <span class="product">Tesseract</span>.

## Visualize Ipopt Solution for Maximizing Displacement
Ipopt finds a solution that moves the ellipse in the top right corner of the grid. This intuitively makes sense as we induce more displacement based on where we move the ellipse. The solution also stretches the ellipse towards the corner of the domain.

```julia
fontsize_theme = Theme(fontsize = 21)
set_theme!(fontsize_theme)

field_values_stress = eval_fields_stress(x_ipopt_displacement)
field_values_displacement = eval_fields_displacement(x_ipopt_displacement)

input_plot_fx = reshape(field_values_stress[:fx], 51, 51)
input_plot_fy = reshape(field_values_stress[:fy], 51, 51)

prediction_plot_ux = reshape(field_values_displacement[:ux], 51, 51)
prediction_plot_uy = reshape(field_values_displacement[:uy], 51, 51)
prediction_plot_stress = reshape(field_values_stress[:s], 51, 51)
displacement_norm = sqrt.( (field_values_displacement[:ux].^2 + field_values_displacement[:uy]).^2 )
prediction_plot_d = permutedims(reshape(displacement_norm, 51, 51))

fig = Figure(size = (1200,300))

ax1 = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y", title="Force")
hm1 = heatmap!(ax1, input_plot_fx)
Colorbar(fig[1,2], hm1)

ax3 = CairoMakie.Axis(fig[1, 3]; xlabel="x", ylabel="y", title="Von-Mises Stress")
hm3 = heatmap!(ax3, prediction_plot_stress)
Colorbar(fig[1,4], hm3)

ax4 = CairoMakie.Axis(fig[1,5]; xlabel="x", ylabel="y", title="Norm Displacement")
hm4 = heatmap!(ax4, prediction_plot_d)
Colorbar(fig[1,6], hm4)

fig
```

```{figure} optimize_displacement.png
:alt: linear-elasticity-maximize-displacement
:width: 1200px

<small>Force, stress, and displacement fields corresponding to a maximum displacement solution..</small>
```

# Minimize von-Mises Stress
We minimize the average von-Mises stress in the same way as maximized displacement. We formulate the optimization model, we define the objective function, and solve with Ipopt.

## Formulate Optimization Problem to Minimize the Average von-Mises Stress
Remember that the stress <span class="product">Tesseract</span> returns the average von-Mises stress on the free boundary (i.e. the stress on $x=1$ and $y=1$) as well as its gradient. We again use Plasmosis.jl to formulate an optimization problem.

```julia
function build_stress_model()
    # generate definition
    lower_bounds = [0.335, 0.335, 0.0386, 0.0234]
    upper_bounds = [0.664, 0.664, 0.329, 179.98]
    n_inputs = 4
    n_outputs = 1
    surrogate_definition = GenericSurrogate(n_inputs, n_outputs, lower_bounds, upper_bounds)

    # create evaluator
    f_eval_forward_stress(x::Vector{Float64}) = eval_forward_stress(x)
    f_eval_gradient_stress(x::Vector{Float64}) = eval_gradient_stress(x)
    surrogate_evaluator = SurrogateEvaluator(f_eval_forward_stress, f_eval_gradient_stress)

    # build optimization formulation
    model = Model()
    build_formulation(
        model,
        ComposedFormulation(),
        surrogate_definition,
        surrogate_evaluator;
        name=:stress
    )

    # set initial value
    x0 = [0.5, 0.5, 0.113, 45.0]
    set_start_value.(all_variables(model), x0)

    mean_stress = model.ext[:stress][:outputs][1]
    @objective(model, Min, mean_stress/1e5) # smaller objective improves scaling for Ipopt
    return model
end
```

## Run the Optimizer
We use simlar optimization settings and run the optimizer in the same fashion as the displacement problem above.

```julia
model_stress = build_stress_model()
set_optimizer(model_stress, Ipopt.Optimizer)
set_optimizer_attribute(model_stress, "hessian_approximation", "limited-memory")
set_optimizer_attribute(model_stress, "print_level", 5)
set_optimizer_attribute(model_stress, "tol", 1e-2)
set_optimizer_attribute(model_stress, "max_iter", 100)
JuMP.optimize!(model_stress)
```
```
This is Ipopt version 3.14.16, running with linear solver MUMPS 5.7.3.

Number of nonzeros in equality constraint Jacobian...:        0
Number of nonzeros in inequality constraint Jacobian.:        0
Number of nonzeros in Lagrangian Hessian.............:        0

Total number of variables............................:        4
                     variables with only lower bounds:        0
                variables with lower and upper bounds:        4
                     variables with only upper bounds:        0
Total number of equality constraints.................:        0
Total number of inequality constraints...............:        0
        inequality constraints with only lower bounds:        0
   inequality constraints with lower and upper bounds:        0
        inequality constraints with only upper bounds:        0

iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
   0  1.8353610e-01 0.00e+00 4.78e-01   0.0 0.00e+00    -  0.00e+00 0.00e+00   0
   1  1.7094021e-01 0.00e+00 1.89e-01  -0.7 7.81e-02    -  9.91e-01 1.00e+00f  1
   2  1.5999992e-01 0.00e+00 4.11e-02  -1.6 9.21e-02    -  9.92e-01 1.00e+00f  1
   3  1.3042926e-01 0.00e+00 5.16e-02  -2.5 1.95e-01    -  9.44e-01 1.00e+00f  1
   4  1.1644991e-01 0.00e+00 1.86e-01  -3.3 1.08e-01    -  9.94e-01 1.00e+00f  1
   5  1.1594185e-01 0.00e+00 1.18e-01  -2.7 7.78e-02    -  1.00e+00 5.00e-01f  2
   6  1.1499322e-01 0.00e+00 2.88e-02  -3.5 2.69e-02    -  1.00e+00 1.00e+00f  1
   7  1.1408559e-01 0.00e+00 7.05e-02  -5.2 1.03e-02    -  1.00e+00 9.87e-01f  1
   8  1.0542880e-01 0.00e+00 4.74e-01  -5.6 1.23e-01    -  1.00e+00 6.60e-01f  1
   9  1.0529379e-01 0.00e+00 4.93e-01  -6.5 1.44e-01    -  1.00e+00 7.32e-03f  1
iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr  ls
  10  9.8336529e-02 0.00e+00 5.16e-01  -7.1 4.26e-01    -  1.00e+00 3.06e-02f  1
  11  9.8333861e-02 0.00e+00 3.03e-01 -11.0 2.88e-04    -  4.22e-01 2.83e-02f  1
  12  9.8307038e-02 0.00e+00 1.76e-01  -6.7 5.17e-04    -  3.48e-01 6.18e-01f  1
  13  9.8301848e-02 0.00e+00 1.67e-02  -9.0 2.54e-04    -  1.00e+00 6.16e-01f  1
  14  9.8299099e-02 0.00e+00 4.75e-02  -9.7 9.02e-05    -  1.00e+00 1.00e+00f  1
  15  9.8294189e-02 0.00e+00 1.34e-03 -11.0 2.32e-04    -  1.00e+00 1.00e+00f  1
  16  9.8294180e-02 0.00e+00 9.36e-04 -11.0 6.49e-06    -  1.00e+00 1.00e+00f  1

Number of Iterations....: 16

                                   (scaled)                 (unscaled)
Objective...............:   9.8294180295543834e-02    9.8294180295543834e-02
Dual infeasibility......:   9.3577021153362150e-04    9.3577021153362150e-04
Constraint violation....:   0.0000000000000000e+00    0.0000000000000000e+00
Variable bound violation:   9.9796192976064901e-09    9.9796192976064901e-09
Complementarity.........:   1.0000514737138981e-11    1.0000514737138981e-11
Overall NLP error.......:   9.3577021153362150e-04    9.3577021153362150e-04


Number of objective function evaluations             = 22
Number of objective gradient evaluations             = 17
Number of equality constraint evaluations            = 0
Number of inequality constraint evaluations          = 0
Number of equality constraint Jacobian evaluations   = 0
Number of inequality constraint Jacobian evaluations = 0
Number of Lagrangian Hessian evaluations             = 0
Total seconds in IPOPT                               = 0.807

EXIT: Optimal Solution Found.
```

## Visualize Ipopt Solution for Minimizing Stress
Following our intuition, Ipopt finds a solution that moves the ellipse in the bottom left corner of the grid to minimize the average stress on the free boundary.

```julia
x_ipopt_stress = value.(all_variables(model_stress))
field_values_stress = eval_fields_stress(x_ipopt_stress)
field_values_displacement = eval_fields_displacement(x_ipopt_stress)

input_plot_fx = reshape(field_values_stress[:fx], 51, 51)
input_plot_fy = reshape(field_values_stress[:fy], 51, 51)

prediction_plot_ux = reshape(field_values_displacement[:ux], 51, 51)
prediction_plot_uy = reshape(field_values_displacement[:uy], 51, 51)
prediction_plot_stress = reshape(field_values_stress[:s], 51, 51)
displacement_norm = sqrt.( (field_values_displacement[:ux].^2 + field_values_displacement[:uy]).^2 )
prediction_plot_d = permutedims(reshape(displacement_norm, 51, 51))

fig = Figure(size = (1200,300))

ax1 = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y", title="Force")
hm1 = heatmap!(ax1, input_plot_fx)
Colorbar(fig[1,2], hm1)

ax3 = CairoMakie.Axis(fig[1, 3]; xlabel="x", ylabel="y", title="Von-Mises Stress")
hm3 = heatmap!(ax3, prediction_plot_stress)
Colorbar(fig[1,4], hm3)

ax4 = CairoMakie.Axis(fig[1,5]; xlabel="x", ylabel="y", title="Norm Displacement")
hm4 = heatmap!(ax4, prediction_plot_d)
Colorbar(fig[1,6], hm4)

fig
```
```{figure} optimize_stress.png
:alt: linear-elasticity-minimize_stress
:width: 1200px

<small>Force, stress, and displacement fields corresponding to a minimum stress solution.</small>
```

# Multi-Objective Optimization
To conclude this tutorial, we shift focus to building a coupled optimization model that incorporates both <span class="product">Tesseracts</span>. We now elect to use a multi-objective algorithm that will trade-off minimizing the average stress with maximizing the average displacement on the free boundary.

## Formulate Optimization Problem to Minimize Stress and Maximize Displacement
We now define a coupled model where we use Plasmosis.jl to build both stress and displacement models that we couple by means of a shared objective function. The objective function in this case is a vector function that contains both stress and displacement objectives in correspondence of the *same* ellipse parameters. This vector function can be communicated to a multi-objective algorithm.

```julia
function build_coupled_model()
    # generate definition
    lower_bounds = [0.335, 0.335, 0.0386, 0.0234]
    upper_bounds = [0.664, 0.664, 0.329, 179.98]
    n_inputs = 4
    n_outputs = 1
    surrogate_definition = GenericSurrogate(n_inputs, n_outputs, lower_bounds, upper_bounds)

    # build optimization formulation
    model = Model()
    @variable(model, lower_bounds[i] <= x[i=1:n_inputs] <= upper_bounds[i])

    # build displacement model
    f_eval_forward_displacement(x::Vector{Float64}) = eval_forward_displacement(x)
    f_eval_gradient_displacement(x::Vector{Float64}) = eval_gradient_displacement(x)
    surrogate_evaluator_displacement = SurrogateEvaluator(f_eval_forward_displacement, f_eval_gradient_displacement)
    build_formulation(
        model,
        x,
        ComposedFormulation(),
        surrogate_definition,
        surrogate_evaluator_displacement;
        name=:displacement
    )

    # build stress model
    f_eval_forward_stress(x::Vector{Float64}) = eval_forward_stress(x)
    f_eval_gradient_stress(x::Vector{Float64}) = eval_gradient_stress(x)
    surrogate_evaluator_stress = SurrogateEvaluator(f_eval_forward_stress, f_eval_gradient_stress)
    build_formulation(
        model,
        x,
        ComposedFormulation(),
        surrogate_definition,
        surrogate_evaluator_stress;
        name=:stress
    )

    mean_displacement = model.ext[:displacement][:outputs][1]
    mean_stress = model.ext[:stress][:outputs][1]

    # define multi-objective function. we use negative displacement to indicate we actually want to maximize it.
    @objective(model, Min, [mean_stress/1e5, -mean_displacement])
    return model
end
```

## Optimize with a Multi-Objective Epsilon-Constraint Algorithm
We build our coupled optimization model and use an epsilon-constraint algorithm (from MultiObjectiveAlgorithms.jl) to explore the Pareto front for minimum stress and maximum displacement. Note that we are using a local optimizer (Ipopt), so it is an approximation of the true pareto frontier. It would be possible in this case to use a simple multi-start algorithm to evaluate different initial points to search multiple minima. More high-dimensional problems could use various global search strategies in combination with Ipopt as a local optimizer.

```julia
multi_model_eps = build_coupled_model()
x0 = [0.5, 0.5, 0.1, 45.0]
set_start_value.(all_variables(multi_model_eps), x0)

ipopt_optimizer = optimizer_with_attributes(
    Ipopt.Optimizer,
    "hessian_approximation" => "limited-memory",
    "print_level" => 5,
    "tol" => 1e-2,
    "max_iter" => 100
)
set_optimizer(multi_model_eps, () -> MOA.Optimizer(ipopt_optimizer))
set_attribute(multi_model_eps, MOA.Algorithm(), MOA.EpsilonConstraint())
set_attribute(multi_model_eps, MOA.SolutionLimit(), 6)
JuMP.optimize!(multi_model_eps)
```

## Visualize Multi-Objective Results
The epsilon-constraint algorithm produces a pareto-front of optimization solutions that trade-off displacement and stress objectives.

```julia
# fill objective value vectors
eps_stress = []
eps_displacement = []
for i = 1:result_count(multi_model_eps)
    x_eps = value.(all_variables(multi_model_eps), result=i)
    stress = eval_forward_stress(x_eps)
    displacement = eval_forward_displacement(x_eps)
    push!(eps_stress, stress)
    push!(eps_displacement, displacement)
end

# get best stress, displacement, and the utopia point
best_stress_point = [eval_forward_stress(x_ipopt_stress), eval_forward_displacement(x_ipopt_stress)]
best_displacement_point = [eval_forward_stress(x_ipopt_displacement), eval_forward_displacement(x_ipopt_displacement)]
ideal_point = [eval_forward_stress(x_ipopt_stress), eval_forward_displacement(x_ipopt_displacement)]

# plot pareto frontier
fontsize_theme = Theme(fontsize = 24)
set_theme!(fontsize_theme)
fig = Figure(size = (1200,900))
ax1 = CairoMakie.Axis(fig[1, 1]; xlabel="Mean Stress", ylabel="Mean Displacement")
scatter!(ax1, [best_stress_point[1]], [best_stress_point[2]], markersize=42, label="Optimize Stress", marker=:diamond)
scatter!(ax1, [best_displacement_point[1]], [best_displacement_point[2]], markersize=42, label="Optimize Displacement", marker=:diamond)
scatter!(ax1, [ideal_point[1]], [ideal_point[2]], markersize=42, label="Utopia Point", marker=:cross)
scatter!(ax1, eps_stress[1:end], eps_displacement[1:end], markersize=32, label="Multi-Objective-Algorithm")
axislegend(ax1; position=:rb)

fig
```

```{figure} multi_objective.png
:alt: linear-elasticity-pareto-frontier
:width: 1200px

<small>Pareto Front for Maximizing Displacement and Minimizing Stress.</small>
```

## Visualize a Select Pareto Point
We visualize one of the intermediate Pareto solutions (the fifth solution in particular) to observe a potential tradeoff. The solution for this point places the ellipse near the center right and stretches the major axis. This induces considerable displacement on the free boundary with only a moderate amount of stress.

```julia
result = 5 # grab the fifth optimization result

x_ipopt_multi_eps = value.(all_variables(multi_model_eps), result=result)
field_values_stress = eval_fields_stress(x_ipopt_multi_eps)
field_values_displacement = eval_fields_displacement(x_ipopt_multi_eps)

input_plot_fx = reshape(field_values_stress[:fx], 51, 51)
input_plot_fy = reshape(field_values_stress[:fy], 51, 51)

prediction_plot_ux = reshape(field_values_displacement[:ux], 51, 51)
prediction_plot_uy = reshape(field_values_displacement[:uy], 51, 51)
prediction_plot_stress = reshape(field_values_stress[:s], 51, 51)
displacement_norm = sqrt.( (field_values_displacement[:ux].^2 .+ field_values_displacement[:uy]).^2 )
prediction_plot_d = permutedims(reshape(displacement_norm, 51, 51))

fig = Figure(size = (1200,300))

ax1 = CairoMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y", title="Force")
hm1 = heatmap!(ax1, input_plot_fx)
Colorbar(fig[1,2], hm1)

ax3 = CairoMakie.Axis(fig[1, 3]; xlabel="x", ylabel="y", title="Von-Mises Stress")
hm3 = heatmap!(ax3, prediction_plot_stress)
Colorbar(fig[1,4], hm3)

ax4 = CairoMakie.Axis(fig[1,5]; xlabel="x", ylabel="y", title="Norm Displacement")
hm4 = heatmap!(ax4, prediction_plot_d)
Colorbar(fig[1,6], hm4)

fig
```

```{figure} pareto_point.png
:alt: linear-elasticity-pareto-point
:width: 1200px

<small>Selected Intermediate Pareto Point Solution.</small>
```
