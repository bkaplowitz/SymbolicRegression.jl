using Test
using SymbolicRegression
using Random
using Statistics

# First import TensorOperators
include("../src/TensorOperators.jl")
using .TensorOperators

# Create wrapper functions that handle both scalar and tensor inputs
function wrapped_grad_x(x)
    if x isa AbstractArray{<:Real,3}
        return grad_along_x(x)
    else
        return convert(typeof(x), NaN)  # Return NaN for non-tensor inputs
    end
end

function wrapped_grad_y(x)
    if x isa AbstractArray{<:Real,3}
        return grad_along_y(x)
    else
        return convert(typeof(x), NaN)
    end
end

function wrapped_laplacian(x)
    if x isa AbstractArray{<:Real,3}
        return laplacian(x)
    else
        return convert(typeof(x), NaN)
    end
end

function wrapped_divergence(x, y)
    if x isa AbstractArray{<:Real,3} && y isa AbstractArray{<:Real,3}
        return divergence(x, y)
    else
        return convert(promote_type(typeof(x), typeof(y)), NaN)
    end
end

function wrapped_curl_2d(x, y)
    if x isa AbstractArray{<:Real,3} && y isa AbstractArray{<:Real,3}
        return curl_2d(x, y)
    else
        return convert(promote_type(typeof(x), typeof(y)), NaN)
    end
end

@testset "tensor shapes" begin
    # Create options with tensor operators
    options = Options(;
        binary_operators=[+, *, wrapped_divergence, wrapped_curl_2d],
        unary_operators=[wrapped_grad_x, wrapped_grad_y, wrapped_laplacian],
    )
    @extend_operators options

    # Test data
    X = randn(Float32, 10, 32, 32)  # batch × width × height
    y = laplacian(X) .+ grad_along_x(X)  # Example PDE to discover

    # Basic operator tests
    @test size(grad_along_x(X)) == size(X)
    @test size(grad_along_y(X)) == size(X)
    @test size(laplacian(X)) == size(X)

    # Test divergence and curl with vector fields
    fx = randn(Float32, 10, 32, 32)
    fy = randn(Float32, 10, 32, 32)
    @test size(divergence(fx, fy)) == size(X)
    @test size(curl_2d(fx, fy)) == size(X)
end

@testset "Basic tensor operations" begin
    # Create options with tensor operators
    options = Options(;
        binary_operators=[+, *, wrapped_divergence, wrapped_curl_2d],
        unary_operators=[wrapped_grad_x, wrapped_grad_y, wrapped_laplacian],
        populations=8,  # Smaller population for testing
        maxsize=10,    # Limit expression size for testing
        parsimony=0.1,  # Encourage simpler expressions
    )
    @extend_operators options

    # Generate synthetic PDE data
    nx, ny, nt = 32, 32, 10
    dx = dy = 2π / 32
    dt = 0.1

    # Create grid
    x = reshape(range(0, 2π - dx, nx), 1, nx, 1)
    y = reshape(range(0, 2π - dy, ny), 1, 1, ny)
    t = reshape(range(0, dt * (nt - 1), nt), nt, 1, 1)

    # Generate true solution: u_t = ∇²u
    u = @. sin(2x) * cos(3y)  # Initial condition
    # Calculate true time derivative using Laplacian
    dudt = laplacian(u)  # Target: ∂u/∂t = ∇²u

    # Keep tensor structure: [batch, width, height]
    X = reshape(u, nt, nx, ny)  # Maintain 3D structure
    y = reshape(dudt, nt, nx, ny)

    # Reshape for symbolic regression while preserving tensor structure
    X_tensor = reshape(X, 1, :)  # 1 feature × (nt*nx*ny) samples
    y_flat = vec(y)              # (nt*nx*ny) samples

    # Run equation search
    hall_of_fame = equation_search(
        X_tensor,
        y_flat;
        niterations=20,  # Small number for testing
        options=options,
        parallelism=:serial,
    )

    # Get best equation
    dominating = calculate_pareto_frontier(X_tensor, y_flat, hall_of_fame, options)
    best = last(dominating)

    # The true equation should be y = laplacian(x)
    # Test if we recovered something close to the true equation
    @test best.loss < 0.1

    # Test prediction
    predicted = best.tree(X_tensor, options)
    @test cor(predicted, y_flat) > 0.9  # Strong correlation with true solution

    # Print the discovered equation
    println("Discovered equation: ", string_tree(best.tree, options))
end
