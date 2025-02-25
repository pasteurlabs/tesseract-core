using Flux, ProgressMeter
using DataFrames, CSV
using MLDataUtils, Statistics
using BSON

# load data

function load_data()
    df = DataFrame(CSV.File(joinpath(@__DIR__, "dataset.csv")))
    x_data = Matrix(transpose(Float64.(df.x)))
    y_data = Matrix(transpose(Float64.(df.y)))
    return x_data, y_data
end

function split_train_test(x_data, y_data; split=0.85)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=split)
    return (x_train, y_train), (x_test, y_test)
end

# loss

function loss(model, x, y)
    return mean(abs2.(model(x) .- y))
end

# initialize new architecture
# NOTE: juliacall is failing when i use type signature here
function initialize_mlp(
    layer_nodes, #::Vector{Int64},
    activation, #::Function
)
    layers = []
    input = f64(Dense(1 => layer_nodes[1]))
    push!(layers, input)
    for j in 1:(length(layer_nodes) - 1)
        push!(layers, f64(Dense(layer_nodes[j] => layer_nodes[j + 1], activation)))
    end
    output = f64(Dense(layer_nodes[end] => 1))
    push!(layers, output)

    model = Chain(layers...)
    return model
end

# train model

function train_mlp!(
    model::Flux.Chain, x_train::AbstractArray, y_train::AbstractArray; n_epochs::Int64=1000
)
    data = [(x_train, y_train)]
    opt = Flux.setup(Adam(), model)
    @showprogress for epoch in 1:n_epochs
        Flux.train!(loss, model, data, opt)
    end
end

# accuracy

function mse(model::Flux.Chain, x_test::AbstractArray, y_test::AbstractArray)
    return sum((y_test .- model(x_test)) .^ 2) ./ length(y_test)
end

# export

function save_state(model::Flux.Chain, path::String)
    state_to_save = Flux.state(model)
    BSON.@save path state_to_save
    return nothing
end

function load_state(model::Flux.Chain, path::String)
    try
        loaded_state = BSON.load(path)[:state_to_save]
        Flux.loadmodel!(model, loaded_state)
    catch DimensionMismatch
        error(
            "Failed to load model state; it has inconsistent dimensions with the defined input architecture.",
        )
    end
    return nothing
end
