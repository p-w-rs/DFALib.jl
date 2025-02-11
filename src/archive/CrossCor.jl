module CrossCorr

using FLoops, StaticArrays

export crosscor!, allocate, calc_output_size

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
        )
    end
    return nothing
end

end # module
