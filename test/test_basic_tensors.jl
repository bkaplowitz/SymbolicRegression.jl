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
    # Create options with tensor operators - reduce operator set to encourage Laplacian use
    options = Options(;
        binary_operators=[+, *],
        unary_operators=[laplacian],  # Use laplacian directly
        populations=20,
        maxsize=15,
        parsimony=0.01,
        constraints=[(laplacian => 1)],  # Use constraints with direct operator
    )
    @extend_operators options

    # Generate synthetic PDE data
    nx, ny, nt = 32, 32, 10
    dx = dy = 2π / 32
    dt = 0.1

    # Create grid
    x = reshape(range(0, 2π - dx, nx), 1, nx, 1)
    y = reshape(range(0, 2π - dy, ny), 1, 1, ny)

    # Generate initial condition that will show Laplacian behavior clearly
    u_init = @. sin(2x) * sin(3y)  # Initial condition with clear spatial variation
    u = repeat(u_init; outer=(nt, 1, 1))  # Repeat for each time step

    # Calculate true time derivative using laplacian
    dudt = laplacian(u)  # Target: ∂u/∂t = ∇²u

    # Reshape for symbolic regression
    X_tensor = reshape(u, 1, :)  # 1 feature × (nt*nx*ny) samples
    y_flat = vec(dudt)          # (nt*nx*ny) samples

    # Run equation search
    hall_of_fame = equation_search(
        X_tensor, y_flat; niterations=50, options=options, parallelism=:serial
    )

    # Get best equation
    dominating = calculate_pareto_frontier(X_tensor, y_flat, hall_of_fame, options)
    best = last(dominating)

    # The true equation should be y = laplacian(x)
    @test best.loss < 0.1

    # Test prediction
    predicted = best.tree(X_tensor, options)
    @test cor(predicted, y_flat) > 0.9

    # Print the discovered equation
    println("Discovered equation: ", string_tree(best.tree, options))

    # Test that laplacian is used in the solution
    @test any(get_tree(best.tree)) do node
        node.degree == 1 && options.operators.unaops[node.op] == laplacian
    end
end
