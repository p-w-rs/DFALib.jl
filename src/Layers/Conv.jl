# Conv.jl
# Not even close to working yet

struct Conv{T<:AbstractFloat} <: Layer
    K::Array{T,3}        # W x H x C_out
    b::Vector{T}         # one bias for each output channel
    B::Array{T,4}        # 1 x final_size x (W_out*H_out*C_out)

    x::Array{T,4}        # W x H x C_in x BatchSize
    d::Array{T,4}        # 1 x BatchSize
    y::Array{T,4}        # Wout x Hout x C_out x BatchSize

    padding::Int
    stride::Int
    dilation::Int
    σ::Function
end

function Base.show(io::IO, l::Conv)
    # TODO
end

function (l::Conv{T})(x::AbstractArray{T,4}) where {T<:AbstractFloat}
    for i in 1:size(K, 3)
        crosscor!(l.y[:, :, i, :], x, l.K[:, :, i], l.padding, l.stride, l.dilation)
    end
end

function (l::Conv{T})(x::AbstractArray{T,3}) where {T<:AbstractFloat}
    # Convert input to W X H X C X N
    l(reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)))
end

function (l::Conv{T})(x::AbstractMatrix{T}) where {T<:AbstractFloat}
    # Convert input to W X H X C X N
    l(reshape(x, size(x, 1), size(x, 2), 1, 1))
end

function (l::Conv{T})(x::AbstractVector{T}) where {T<:AbstractFloat}
    # Convert input to W X H X C X N
    l(reshape(x, size(l.K, 1), size(l.K, 2), 1, 1))
end

function feedback!(l::Conv{T}, e::AbstractArray{T}, η::T) where {T<:AbstractFloat}
    # TODO
end




struct ConvConstructor <: LayerConstructor
    kernel::Tuple{Int,Int}
    channels::Pair{Int,Int}
    σ::Function
    padding::Int
    stride::Int
    dilation::Int
    initW::Function
    initb::Function
    initB::Function
end

function Conv(kernel::Tuple{Int,Int}, channels::Pair{Int,Int}, σ::Function; padding::Int=0, stride::Int=1, dilation::Int=1, initW=RandomNormal, initb=Ones, initB=RandomNormal)
    return ConvConstructor(kernel, channels, σ, padding, stride, dilation, initW, initb, initB)
end

function (c::ConvConstructor)(rng::AbstractRNG, ::Type{T}, insz::Union{Int,Tuple{Vararg{Int}}}, finlsz::Int) where {T<:AbstractFloat}
    if insz isa Tuple
        # Standard convolution
        c.kernel[1] <= insz[1] || throw(ArgumentError("Input width $(insz[1]) is smaller than kernel width $(c.kernel[1])"))
        c.kernel[2] <= insz[2] || throw(ArgumentError("Input height $(insz[2]) is smaller than kernel height $(c.kernel[2])"))
        insz[3] == c.channels[1] || throw(ArgumentError("Input channels $(insz[3]) doesn't match expected $(c.channels[1])"))
    else
        # 1D convolution either treat input either horizontally or vertically depending on the kernel size either way it will come in as an AbstractVecOrMat
        # so no difference in implementation
        (min(c.kernel) == 1 && max(c.kernel) <= insz) || throw(ArgumentError("Input size $(insz) doesn't match kernel size $(c.kernel)"))
    end
    return Conv(), outsz
end




### Cross Correlation Primitives ###
function calc_output_size(
    input_size::Tuple{Int,Int}, kernel_size::Tuple{Int,Int},
    padding::Int, stride::Int, dilation::Int
)::Tuple{Int,Int}
    H_in, W_in = input_size
    K_h, K_w = kernel_size

    effective_k_h = K_h + (K_h - 1) * (dilation - 1)
    effective_k_w = K_w + (K_w - 1) * (dilation - 1)

    H_out = div(H_in + 2 * padding - effective_k_h, stride) + 1
    W_out = div(W_in + 2 * padding - effective_k_w, stride) + 1

    return (H_out, W_out)
end

function allocate(
    ::Type{T}, A::Array{T,4}, K::Array{T,2},
    padding::Int, stride::Int, dilation::Int
)::Array{T,4} where {T<:AbstractFloat}
    H_in, W_in, C, N = size(A)
    K_h, K_w = size(K)
    H_out, W_out = calc_output_size((H_in, W_in), (K_h, K_w), padding, stride, dilation)
    return zeros(T, H_out, W_out, 1, N)
end

# Helper function for kernel computation with SIMD
@inline function compute_kernel(
    A::Array{T,4}, K::SMatrix{KH,KW,T},
    i::Int, j::Int, C::Int, n::Int,
    H_in::Int, W_in::Int,
    padding::Int, stride::Int, dilation::Int
)::T where {T,KH,KW}
    acc = zero(T)
    start_h = i * stride - padding
    start_w = j * stride - padding

    @fastmath @inbounds for c in 1:C
        @simd for ki in 1:KH
            @simd for kj in 1:KW
                in_h = start_h + (ki - 1) * dilation
                in_w = start_w + (kj - 1) * dilation

                if 1 <= in_h <= H_in && 1 <= in_w <= W_in
                    acc += A[in_h, in_w, c, n] * K[ki, kj]
                end
            end
        end
    end
    return acc
end

function crosscor!(
    result::Array{T,4}, A::Array{T,4}, K::Array{T,2},
    padding::Int, stride::Int, dilation::Int
)::Nothing where {T<:AbstractFloat}
    H_in, W_in, C, N = size(A)
    K_h, K_w = size(K)
    H_out, W_out = size(result)[1:2]

    # Convert kernel to static array for better SIMD
    static_K = SMatrix{K_h,K_w,T}(K)

    # Parallelize all independent computations
    @floop ThreadedEx() for idx in CartesianIndices((H_out, W_out, N))
        i, j, n = Tuple(idx)

        result[i, j, 1, n] = compute_kernel(
            A, static_K, i, j, C, n,
            H_in, W_in, padding, stride, dilation
        ) + b
    end
    return nothing
end
