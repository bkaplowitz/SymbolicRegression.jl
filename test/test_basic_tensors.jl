using Test
using SymbolicRegression
using Random
using Statistics

# First import TensorOperators
include("../src/TensorOperators.jl")
using .TensorOperators

# Create wrapper functions that handle both scalar and tensor inputs

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
        binary_operators=[+, *],
        unary_operators=[laplacian],
        populations=20,
        maxsize=15,
        parsimony=0.1,  # Increase parsimony to encourage simpler solutions
        weights=nothing,  # Don't use weights for complexity
        should_simplify=false,  # Prevent simplification that might remove laplacian
        constraints=[(laplacian => 1)],  # Require exactly one laplacian
        seed=0,  # Make test deterministic
    )
    @extend_operators options

    # Generate synthetic PDE data for Laplace equation ∇²u = 0
    nx, ny, nt = 16, 16, 5  # Smaller grid for faster testing
    dx = dy = 2π / 16

    # Create grid
    x = reshape(range(0, 2π - dx, nx), 1, nx, 1)
    y = reshape(range(0, 2π - dy, ny), 1, 1, ny)

    # Solution to Laplace equation: u = sin(kx)cos(ky)
    # This satisfies ∇²u = -k²(sin(kx)cos(ky)) - k²(sin(kx)cos(ky)) = -2k²u
    k = 1  # Wave number
    u_exact = @. sin(k * x) * cos(k * y)
    u = repeat(u_exact; outer=(nt, 1, 1))

    # For this solution, ∇²u = -2k²u
    dudt = @. -2.0 * k^2 * u  # Target is proportional to u

    # Test that our laplacian operator gives expected results
    laplacian_u = laplacian(u)
    @test maximum(abs.(laplacian_u .- dudt)) < 1e-5  # Should match our analytical solution

    # Reshape for symbolic regression
    X_tensor = reshape(u, 1, :)
    y_flat = vec(dudt)

    # Run equation search with more iterations
    hall_of_fame = equation_search(
        X_tensor,
        y_flat;
        niterations=50,  # Increase iterations
        options=options,
        parallelism=:serial,
    )

    # Get best equation
    dominating = calculate_pareto_frontier(X_tensor, y_flat, hall_of_fame, options)
    best = last(dominating)

    # Print the discovered equation
    println("Discovered equation: ", string_tree(best.tree, options))

    # The true equation should be y = laplacian(x)
    @test best.loss < 1e-5  # Should be very close to exact solution

    # Test that laplacian is ACTUALLY used in the solution
    has_laplacian = any(get_tree(best.tree)) do node
        node.degree == 1 && options.operators.unaops[node.op] == laplacian
    end
    @test has_laplacian "Solution must use the Laplacian operator"

    # Test prediction
    predicted = best.tree(X_tensor, options)
    @test maximum(abs.(predicted .- y_flat)) < 1e-5  # Should match exactly
end
