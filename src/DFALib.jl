# DFALib.jl
module DFALib

using LinearAlgebra, Random, Printf
using FLoops, StaticArrays
using Base.Threads: @threads
export DFANet, feedback!


include("Activations.jl")
export Identity, Sigmoid, ReLU, ReLU6, LReLU, CELU, Softmax
export identity, identity′, sigmoid, sigmoid′, relu, relu′, relu6, relu6′, lrelu, lrelu′, celu, celu′, softmax, softmax′


include("WeightInitializers.jl")
export GlorotUniform, GlorotNormal, HeUniform, HeNormal, LeCunUniform, LeCunNormal, Orthogonal, RandomNormal, RandomUniform, Ones, Zeros


abstract type Layer end
abstract type LayerConstructor end
include("Layers/Dense.jl")
include("Layers/Conv.jl")
export Dense, Conv


mutable struct DFANet{T<:AbstractFloat}
    layers::Vector{Layer}

    function DFANet{T}(layers::Vector{Layer}) where {T<:AbstractFloat}
        !isempty(layers) || throw(ArgumentError("Network must have at least one layer"))
        return new{T}(layers)
    end
end

function Base.show(io::IO, nn::DFANet)
    println(io, "DFANet(")
    for (i, layer) in enumerate(nn.layers)
        print(io, "  [$i] ")
        show(io, layer)
        i < length(nn.layers) && println(io)
    end
    print(io, "\n)")
end

# Constructor with progress printing
function DFANet(rng::AbstractRNG, ::Type{T}, constructors::LayerConstructor...; verbose=false) where {T<:AbstractFloat}
    verbose && println("Building network...")
    layers = Vector{Layer}(undef, length(constructors))
    insz = constructors[1].insz
    final_size = constructors[end].outsz

    for (i, constructor) in enumerate(constructors)
        verbose && print("  Layer $i/$(length(constructors))... ")
        layers[i], insz = constructor(rng, T, insz, final_size)
        verbose && println("✓")
    end
    insz == final_size || throw(ArgumentError("Final layer output size $(insz) doesn't match expected $(final_size)"))

    layers[end].B .= Matrix{T}(I, final_size, final_size)
    verbose && println("Network built successfully!")
    return DFANet{T}(layers)
end

function DFANet(::Type{T}, constructors::LayerConstructor...; verbose=false) where {T<:AbstractFloat}
    DFANet(Random.default_rng(), T, constructors...; verbose=verbose)
end

function DFANet(rng::AbstractRNG, constructors::LayerConstructor...; verbose=false)
    DFANet(rng, Float64, constructors...; verbose=verbose)
end

function DFANet(constructors::LayerConstructor...; verbose=false)
    DFANet(Random.default_rng(), Float64, constructors...; verbose=verbose)
end

# Forward pass with optional progress tracking
function (nn::DFANet{T})(_x::AbstractArray{T}; track_progress=false) where {T<:AbstractFloat}
    x = _x
    if track_progress
        println("Forward pass:")
        for (i, layer) in enumerate(nn.layers)
            print("  Layer $i/$(length(nn.layers))... ")
            x = layer(x)
            @printf("Output shape: %s ✓\n", size(x))
        end
    else
        for layer in nn.layers
            x = layer(x)
        end
    end
    return x
end

# Feedback with progress tracking
function feedback!(nn::DFANet{T}, e::AbstractArray{T}, η::T; track_progress=false) where {T<:AbstractFloat}
    if track_progress
        println("Feedback pass:")
        @threads for i in 1:length(nn.layers)
            print("  Layer $i/$(length(nn.layers))... ")
            feedback!(nn.layers[i], e, η)
            println("✓")
        end
    else
        @threads for i in 1:length(nn.layers)
            feedback!(nn.layers[i], e, η)
        end
    end
end

# Utility function to get total number of parameters
function count_parameters(nn::DFANet)
    return sum(layer -> sum(length, (layer.W, layer.b, layer.B)), nn.layers)
end

end # module
