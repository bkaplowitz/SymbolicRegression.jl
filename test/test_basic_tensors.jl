using Test
using SymbolicRegression
using Random
using Statistics

# First import TensorOperators
include("../src/TensorOperators.jl")
using .TensorOperators

@testset "tensor shapes" begin
    # Create options with tensor operators
    options = Options(;
        binary_operators=[+, *, divergence, curl_2d],
        unary_operators=[grad_along_x, grad_along_y, laplacian],
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
        binary_operators=[+, *, divergence, curl_2d],
        unary_operators=[grad_along_x, grad_along_y, laplacian],
        populations=8,  # Smaller population for testing
        maxsize=10,    # Limit expression size for testing
        parsimony=0.1,  # Encourage simpler expressions
    )
    @extend_operators options

    # Generate synthetic PDE data
    # We'll use a simple 2D diffusion equation: ∂u/∂t = ∇²u
    # with a known solution u(x,y,t) = sin(x)cos(y)exp(-2t)

    nx, ny, nt = 32, 32, 10
    dx = dy = 2π / 32
    dt = 0.1

    # Create grid
    x = reshape(range(0, 2π - dx, nx), 1, nx, 1)
    y = reshape(range(0, 2π - dy, ny), 1, 1, ny)
    t = reshape(range(0, dt * (nt - 1), nt), nt, 1, 1)

    # Generate true solution
    u = @. sin(x) * cos(y) * exp(-2t)

    # Calculate true time derivative (target)
    dudt = @. -2 * sin(x) * cos(y) * exp(-2t)

    # Test basic operators match analytical solutions
    lap_u = laplacian(u) / (dx * dy)  # Scale by grid spacing
    analytical_lap = @. -2 * sin(x) * cos(y) * exp(-2t)  # ∇²[sin(x)cos(y)] = -2sin(x)cos(y)
    @test maximum(abs.(lap_u .- analytical_lap)) < 0.1  # Allow for numerical differences

    # Now try to discover the PDE
    # The true equation is: ∂u/∂t = ∇²u
    # So dudt = laplacian(u)

    # Prepare data for symbolic regression
    # Reshape data to match expected format: features × samples
    X_flat = reshape(u, :, 1)  # Flatten spatial dimensions into samples
    y_flat = vec(dudt)         # Target should be a vector

    # Run equation search
    hall_of_fame = equation_search(
        X_flat,
        y_flat;
        niterations=20,  # Small number for testing
        options=options,
        parallelism=:serial,
    )

    # Get best equation
    best = hall_of_fame[end]

    # Test if we recovered something close to the true equation
    # The true equation should have low loss
    @test best.loss < 0.1

    # Test prediction
    predicted = best.tree(X_flat, options)
    @test cor(predicted, y_flat) > 0.9  # Strong correlation with true solution
end
