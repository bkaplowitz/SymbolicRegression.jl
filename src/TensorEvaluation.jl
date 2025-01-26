module TensorEvaluation

using ..SymbolicRegression
using DynamicExpressions
using ..CoreModule: DATA_TYPE, LOSS_TYPE

# Type aliases for clarity
const Tensor{T} = AbstractArray{T,3} where {T<:Real}
const Matrix{T} = AbstractArray{T,2} where {T<:Real}
const Vector{T} = AbstractArray{T,1} where {T<:Real}

"""
    eval_tensor_tree(tree::Node, X::AbstractArray{T,N}, options::Options) where {T,N}

Evaluate a tree on tensor input data. Handles both scalar and tensor operations.
Returns (output::AbstractArray, completed::Bool)
"""
function eval_tensor_tree(tree::Node, X::AbstractArray{T,N}, options::Options) where {T,N}
    if tree.degree == 0
        if tree.constant
            return fill(convert(T, tree.val), size(X)[1:(N - 1)]...), true
        else
            return X[:, :, tree.feature, :], true
        end
    end

    # Get operator
    op = if tree.degree == 1
        options.operators.unaops[tree.op]
    else
        options.operators.binops[tree.op]
    end

    # Evaluate children
    if tree.degree == 1
        l, completed = eval_tensor_tree(tree.l, X, options)
        !completed && return (similar(X, size(X)[1:(N - 1)]...), false)

        try
            return op.(l), true
        catch e
            return (similar(X, size(X)[1:(N - 1)]...), false)
        end
    else
        l, completed_l = eval_tensor_tree(tree.l, X, options)
        r, completed_r = eval_tensor_tree(tree.r, X, options)

        !completed_l || !completed_r && return (similar(X, size(X)[1:(N - 1)]...), false)

        try
            return op.(l, r), true
        catch e
            return (similar(X, size(X)[1:(N - 1)]...), false)
        end
    end
end

"""
    check_tensor_operator_compatibility(op::Function, input_types::Tuple, output_type::Type)

Check if an operator is compatible with given input and output tensor types.
"""
function check_tensor_operator_compatibility(
    op::Function, input_types::Tuple, output_type::Type
)
    try
        test_inputs = map(T -> rand(T, 2, 2, 2), input_types)
        result = op(test_inputs...)
        return eltype(result) == output_type && size(result, 1) == 2
    catch
        return false
    end
end

end # module
