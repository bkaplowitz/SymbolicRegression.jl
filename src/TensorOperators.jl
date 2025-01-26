module TensorOperators

using ..SymbolicRegression
using DynamicExpressions

# Helper functions for tensor operations
function roll(x::AbstractArray{T,N}, shift::Int; dims::Int) where {T,N}
    inds = map(d -> d == dims ? circshift(1:size(x, d), shift) : (1:size(x, d)), 1:N)
    return x[inds...]
end

# Basic tensor operators with proper type signatures
function grad_along_x(x::AbstractArray{T,3}) where {T<:Real}
    return (roll(x, -1; dims=2) .- roll(x, 1; dims=2)) ./ T(2)
end

function grad_along_y(x::AbstractArray{T,3}) where {T<:Real}
    return (roll(x, -1; dims=3) .- roll(x, 1; dims=3)) ./ T(2)
end

function laplacian(x::AbstractArray{T,3}) where {T<:Real}
    return grad_along_x(grad_along_x(x)) .+ grad_along_y(grad_along_y(x))
end

function divergence(fx::AbstractArray{T,3}, fy::AbstractArray{T,3}) where {T<:Real}
    return grad_along_x(fx) .+ grad_along_y(fy)
end

function curl_2d(fx::AbstractArray{T,3}, fy::AbstractArray{T,3}) where {T<:Real}
    return grad_along_x(fy) .- grad_along_y(fx)
end

# Export the operators
export grad_along_x, grad_along_y, laplacian, divergence, curl_2d

end # module
