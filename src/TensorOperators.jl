module TensorOperators

using ..SymbolicRegression
using DynamicExpressions

# Helper functions for tensor operations
function roll(x::AbstractArray{T,N}, shift::Int; dims::Int) where {T,N}
    inds = map(d -> d == dims ? circshift(1:size(x, d), shift) : (1:size(x, d)), 1:N)
    return x[inds...]
end

# Basic tensor operators with proper type signatures and default behavior
function grad_along_x(x)
    if x isa AbstractArray{<:Real,3}
        return (roll(x, -1; dims=2) .- roll(x, 1; dims=2)) ./ 2
    else
        return convert(typeof(x), NaN)
    end
end

function grad_along_y(x)
    if x isa AbstractArray{<:Real,3}
        return (roll(x, -1; dims=3) .- roll(x, 1; dims=3)) ./ 2
    else
        return convert(typeof(x), NaN)
    end
end

function laplacian(x)
    if x isa AbstractArray{<:Real,3}
        return grad_along_x(grad_along_x(x)) .+ grad_along_y(grad_along_y(x))
    else
        return convert(typeof(x), NaN)
    end
end

function divergence(fx, fy)
    if fx isa AbstractArray{<:Real,3} && fy isa AbstractArray{<:Real,3}
        return grad_along_x(fx) .+ grad_along_y(fy)
    else
        return convert(promote_type(typeof(fx), typeof(fy)), NaN)
    end
end

function curl_2d(fx, fy)
    if fx isa AbstractArray{<:Real,3} && fy isa AbstractArray{<:Real,3}
        return grad_along_x(fy) .- grad_along_y(fx)
    else
        return convert(promote_type(typeof(fx), typeof(fy)), NaN)
    end
end

# Export the operators
export grad_along_x, grad_along_y, laplacian, divergence, curl_2d

end # module
