# Activations.jl

# identity(x::AbstractArray{T}) where {T<:AbstractFloat} = x already defined in Base
@inline identity′(x::AbstractArray{T}) where {T<:AbstractFloat} = ones(T, size(x)...)
@inline Identity(x::AbstractArray{T}) where {T<:AbstractFloat} = x, identity′(x)

@inline sigmoid(x::AbstractArray{T}) where {T<:AbstractFloat} = one(T) ./ (one(T) .+ exp.(-x))
@inline sigmoid′(x::AbstractArray{T}) where {T<:AbstractFloat} = sigmoid(x) .* (one(T) .- sigmoid(x))
@inline Sigmoid(x::AbstractArray{T}) where {T<:AbstractFloat} = sigmoid(x), sigmoid′(x)

@inline relu(x::AbstractArray{T}) where {T<:AbstractFloat} = max.(zero(T), x)
@inline relu′(x::AbstractArray{T}) where {T<:AbstractFloat} = x .> zero(T)
@inline ReLU(x::AbstractArray{T}) where {T<:AbstractFloat} = relu(x), relu′(x)

@inline relu6(x::AbstractArray{T}) where {T<:AbstractFloat} = min.(max.(zero(T), x), T(6))
@inline relu6′(x::AbstractArray{T}) where {T<:AbstractFloat} = (zero(T) .< x .< T(6))
@inline ReLU6(x::AbstractArray{T}) where {T<:AbstractFloat} = relu6(x), relu6′(x)

@inline lrelu(x::AbstractArray{T}) where {T<:AbstractFloat} = max.(T(0.01) .* x, x)
@inline lrelu′(x::AbstractArray{T}) where {T<:AbstractFloat} = (x .> zero(T)) .+ T(0.01) .* (x .<= zero(T))
@inline LReLU(x::AbstractArray{T}) where {T<:AbstractFloat} = lrelu(x), lrelu′(x)

@inline celu(x::AbstractArray{T}) where {T<:AbstractFloat} = max.(zero(T), x) .+ min.(zero(T), one(T) .- exp.(min.(zero(T), x)))
@inline celu′(x::AbstractArray{T}) where {T<:AbstractFloat} = (x .> zero(T)) .+ (x .<= zero(T)) .* exp.(min.(zero(T), x))
@inline CELU(x::AbstractArray{T}) where {T<:AbstractFloat} = celu(x), celu′(x)

@inline function softmax(x::AbstractArray{T}) where {T<:AbstractFloat}
    tmp = exp.(x .- maximum(x, dims=1))
    return tmp ./ sum(tmp, dims=1)
end
@inline softmax′(x::AbstractArray{T}) where {T<:AbstractFloat} = softmax(x) .* (one(T) .- softmax(x))
@inline Softmax(x::AbstractArray{T}) where {T<:AbstractFloat} = softmax(x), softmax′(x)
