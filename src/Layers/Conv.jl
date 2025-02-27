# Conv.jl
# Not even close to working yet

struct Conv{T<:AbstractFloat} <: Layer
    K::Array{T,3}        # W x H x C_out
    b::Vector{T}         # one bias for each output channel
    B::Array{T,3}        # 1 x final_size x (W_out*H_out*C_out)

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
    # TODO
end




struct PaddedView{T,N} <: AbstractArray{T,4}
    img::Array{T,N}
    i::Int          # Starting row in padded coordinates
    j::Int          # Starting column in padded coordinates
    padding::Int    # Padding size
    dilation::Int   # Dilation factor
    H::Int          # Original image height
    W::Int          # Original image width
    C::Int          # Number of channels
    B::Int          # Batch size
    kernel_height::Int
    kernel_width::Int
end

# Define the size of each patch
Base.size(pv::PaddedView) = (pv.kernel_height, pv.kernel_width, pv.C, pv.B)

# Correctly index into the image based on patch position (m, n)
function Base.getindex(pv::PaddedView{T,N}, m::Int, n::Int, c::Int, b::Int) where {T,N}
    # Compute position in padded coordinates
    row_padded = pv.i + (m - 1) * pv.dilation
    col_padded = pv.j + (n - 1) * pv.dilation
    # Check if the position is within the original image bounds
    if (pv.padding + 1 <= row_padded <= pv.padding + pv.H) &&
       (pv.padding + 1 <= col_padded <= pv.padding + pv.W)
        row_orig = row_padded - pv.padding
        col_orig = col_padded - pv.padding
        return pv.img[row_orig, col_orig, c, b]
    else
        return zero(T)
    end
end

# The convolution iterator function
function convolution_iterator(img, kernel_size::Tuple{Int,Int}, padding::Int,
    dilation::Int, stride::Int)
    kernel_height, kernel_width = kernel_size
    H, W, C, B = size(img)
    # Calculate the range of starting positions
    i_max = H + 2 * padding - (kernel_height - 1) * dilation
    j_max = W + 2 * padding - (kernel_width - 1) * dilation
    i_range = 1:stride:max(1, i_max)
    j_range = 1:stride:max(1, j_max)
    # Return an iterator over PaddedView objects
    return (PaddedView(img, i, j, padding, dilation, H, W, C, B,
        kernel_height, kernel_width)
            for i in i_range, j in j_range)
end
