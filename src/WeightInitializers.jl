# WeightInitializers.jl

# Helper functions for calculating fan_in/fan_out
fan_in(dims::Int...) = dims[2]  # For Dense layers: (out, in)
fan_out(dims::Int...) = dims[1]

# Glorot/Xavier initialization
function GlorotUniform(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat}
    limit = sqrt(6 / (fan_in(dims...) + fan_out(dims...)))
    return (rand(rng, T, dims...) .- 0.5) .* 2limit
end

function GlorotNormal(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat}
    std = sqrt(2 / (fan_in(dims...) + fan_out(dims...)))
    return randn(rng, T, dims...) .* std
end

# He/Kaiming initialization
function HeUniform(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat}
    limit = sqrt(6 / fan_in(dims...))
    return (rand(rng, T, dims...) .- 0.5) .* 2limit
end

function HeNormal(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat}
    std = sqrt(2 / fan_in(dims...))
    return randn(rng, T, dims...) .* std
end

# LeCun initialization
function LeCunUniform(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat}
    limit = sqrt(3 / fan_in(dims...))
    return (rand(rng, T, dims...) .- 0.5) .* 2limit
end

function LeCunNormal(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat}
    std = sqrt(1 / fan_in(dims...))
    return randn(rng, T, dims...) .* std
end

# Orthogonal initialization
function Orthogonal(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat}
    if length(dims) != 2
        throw(ArgumentError("Orthogonal initialization only works with 2D arrays"))
    end
    A = randn(rng, T, dims...)
    Q, R = qr(A)
    return Array(Q) .* sign.(diag(R))
end

# Basic initializers
RandomNormal(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat} = randn(rng, T, dims...)
RandomUniform(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat} = rand(rng, T, dims...)
Ones(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat} = ones(T, dims...)
Zeros(rng::AbstractRNG, ::Type{T}, dims::Int...) where {T<:AbstractFloat} = zeros(T, dims...)
