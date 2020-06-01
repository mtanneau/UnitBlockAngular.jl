import Tulip

using SparseArrays

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
    Tulip.KKT.update!(ls, θ, regP, regD)

    dx = zeros(N)
    dy = zeros(M)
    ξp = ones(M)
    ξd = ones(N)
    Tulip.KKT.solve!(dx, dy, ls, ξp, ξd)

    lsref = Tulip.KKT.Dense_SymPosDef(Aref)
    Tulip.KKT.update!(lsref, θ, regP, regD)
    dxref = zeros(N)
    dyref = zeros(M)
    Tulip.KKT.solve!(dxref, dyref, lsref, ξp, ξd)

    @test dx ≈ dxref
    @test dy ≈ dyref

end

@testset "Optimize" begin
    Aref = sparse([
        1.0 -1.0  0.0  0.0 1.1 1.2 1.3 1.4;
        0.0  0.0  1.0 -1.0 2.1 2.2 2.3 2.4;
        0.0  0.0  0.0  0.0 1.0 0.0 1.0 0.0;
        0.0  0.0  0.0  0.0 0.0 1.0 0.0 1.0
    ])
    m0, n0, n, R = 2, 4, 4, 2

    obj = [10000, 10000, 10000, 10000, 10, 20, 100, 200.0]
    lvar = zeros(8)
    uvar = fill(Inf, 8)

    lcon = ones(4)
    ucon = ones(4)
    connames = fill("", 4)
    varnames = fill("", 8)
    
    m = Tulip.Model{Float64}()

    m.params.OutputLevel = 1
    m.params.Presolve = 0

    m.params.MatrixOptions = Tulip.TLA.MatrixOptions(UnitBlockAngularMatrix; m0=m0, n0=n0, n=n, R=R)
    m.params.KKTOptions = Tulip.KKT.SolverOptions(UnitBlockAngularFactor)
    Tulip.load_problem!(m.pbdata, "Test", true, obj, 0.0, Aref, lcon, ucon, lvar, uvar, connames, varnames)
    Tulip.optimize!(m)
end