module StressSurrogate

using Lux
using JLD2, CodecZlib
using ForwardDiff

### load model

model = Chain(
    Dense(5202 => 1000, swish),
    Dense(1000 => 1000, swish),
    Dense(1000 => 2601)
)

@load joinpath(@__DIR__, "../mlp_linear_elasticity_stress.jld2") trained_parameters trained_states
@load joinpath(@__DIR__, "../mlp_linear_elasticity_stress_auxilliary.jld2") data_limits coordinates

# define constants
force_value = 1e6
SURF_AREA = 0.04

function get_data_limits()
    return data_limits
end

function get_coordinates()
    return coordinates
end

function eval_surrogate(load_field)
    prediction = Lux.apply(model, load_field, trained_parameters, trained_states)[1]
    return prediction
end

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

function calc_mean_stress_from_field(prediction)
    # average von-mises stress over x = 1 and y = 1
    prediction = reshape(prediction, 51, 51)
    boundary_stress = [prediction[:,end]; prediction[end,:]]
    return sum(boundary_stress) / length(boundary_stress)
end

function eval_forward(x)
    load_field = generate_field(x)
    predicted_field = eval_surrogate(load_field)
    average_stress = calc_mean_stress_from_field(predicted_field)
    return average_stress
end

function eval_gradient(x)
	return ForwardDiff.gradient(eval_forward, x)
end

end
