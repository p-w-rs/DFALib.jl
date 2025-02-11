# Dense.jl
struct Dense{T<:AbstractFloat} <: Layer
    W::Matrix{T}
    b::Vector{T}
    B::Matrix{T}
    σ::Function
    cache::Dict{Symbol,AbstractArray{T}}

    function Dense(W::Matrix{T}, b::Vector{T}, B::Matrix{T}, σ::Function, cache::Dict{Symbol,AbstractArray{T}}) where {T<:AbstractFloat}
        size(W, 1) == length(b) || throw(DimensionMismatch("Weight matrix and bias vector dimensions must match"))
        size(B, 1) == size(W, 1) || throw(DimensionMismatch("Feedback matrix B must have same number of rows as W"))
        return new{T}(W, b, B, σ, cache)
    end
end

function Base.show(io::IO, l::Dense)
    print(io, "Dense(", size(l.W, 2), " → ", size(l.W, 1), ", ", l.σ, ")")
end

function (l::Dense{T})(x::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    size(x, 1) == size(l.W, 2) || throw(DimensionMismatch("Input dimension $(size(x,1)) doesn't match expected $(size(l.W,2))"))
    l.cache[:x] = x
    y, l.cache[:d] = l.σ(l.W * x .+ l.b)
    return y
end

function (l::Dense{T})(x::AbstractArray{T}) where {T<:AbstractFloat}
    # Auto flatten input (i.e. accept the output of a Conv layer)
    return l(reshape(x, :, size(x)[end]))
end

function feedback!(l::Dense{T}, e::AbstractArray{T}, η::T) where {T<:AbstractFloat}
    size(e, 1) == size(l.B, 2) || throw(DimensionMismatch("Error signal dimension doesn't match B matrix"))
    ΔW = l.B * e .* l.cache[:d] * l.cache[:x]'
    Δb = vec(sum(l.B * e .* l.cache[:d], dims=2))
    l.W .-= η .* ΔW
    l.b .-= η .* Δb
end




struct DenseConstructor <: LayerConstructor
    insz::Int
    outsz::Int
    σ::Function
    initW::Function
    initb::Function
    initB::Function
end

function Dense(insz::Int, outsz::Int, σ::Function; initW=RandomNormal, initb=Ones, initB=RandomNormal)
    return DenseConstructor(insz, outsz, σ, initW, initb, initB)
end

function (c::DenseConstructor)(rng::AbstractRNG, ::Type{T}, insz::Union{Int,Tuple{Vararg{Int}}}, finlsz::Int) where {T<:AbstractFloat}
    prod(insz) == c.insz || throw(ArgumentError("Input size $(insz) doesn't match expected $(c.insz)"))
    W = c.initW(rng, T, c.outsz, c.insz)
    b = c.initb(rng, T, c.outsz)
    B = c.initB(rng, T, c.outsz, finlsz)
    return Dense(W, b, B, c.σ, Dict{Symbol,AbstractArray{T}}()), c.outsz
end
