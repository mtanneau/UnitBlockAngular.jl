import Tulip

@testset "LinearSolver" begin
    # TODO: Tulip constructor

    Aref = [
        1.0 0.0 1.1 1.2 1.3 1.4;
        0.0 1.0 2.1 2.2 2.3 2.4;
        1.0 0.0 3.1 3.2 3.3 3.4;
        0.0 0.0 1.0 0.0 1.0 0.0;
        0.0 0.0 0.0 1.0 0.0 1.0
    ]

    m0, n0, n, R = 3, 2, 4, 2
    M, N = m0 + R, n0 + n
    B0 = Aref[1:3, 1:2]
    B  = Aref[1:3, 3:end]
    blocks = [1, 2, 1, 2]

    A = UnitBlockAngularMatrix(B0, B, R, blocks)

    ls = UnitBlockAngularFactor(A)

    θ = ones(N)
    regP = 1e-2 .* ones(N)
    regD = 1e-2 .* ones(M)
    Tulip.TLA.update_linear_solver!(ls, θ, regP, regD)

    dx = zeros(N)
    dy = zeros(M)
    ξp = ones(M)
    ξd = ones(N)
    Tulip.TLA.solve_augmented_system!(dx, dy, ls, ξp, ξd)

    lsref = Tulip.TLA.DenseLinearSolver(Aref)
    Tulip.TLA.update_linear_solver!(lsref, θ, regP, regD)
    dxref = zeros(N)
    dyref = zeros(M)
    Tulip.TLA.solve_augmented_system!(dxref, dyref, lsref, ξp, ξd)

    @test dx ≈ dxref
    @test dy ≈ dyref

end