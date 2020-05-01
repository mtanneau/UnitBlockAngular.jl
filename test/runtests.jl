using LinearAlgebra
using Test

using UnitBlockAngular

@testset "UnitBlockAngular" begin

Aref = [
    1.0 0.0 1.1 1.2 1.3 1.4;
    0.0 1.0 2.1 2.2 2.3 2.4;
    1.0 0.0 3.1 3.2 3.3 3.4;
    0.0 0.0 1.0 0.0 1.0 0.0;
    0.0 0.0 0.0 1.0 0.0 1.0
]

m0, n0, n, R = 3, 2, 4, 2
B0 = Aref[1:3, 1:2]
B  = Aref[1:3, 3:end]
blocks = [1, 2, 1, 2]

A = UnitBlockAngularMatrix{Float64}(3, 2, 4, 2);
A.B0 .= B0
A.B  .= B
A.blockidx .= blocks

@test size(A) == (m0+R, n0+n)
@test A == Aref

A = UnitBlockAngularMatrix(B0, B, R, blocks)
@test size(A) == (m0+R, n0+n)
@test A == Aref

x, xref = ones(n0 + n), ones(n0 + n)
y, yref = zeros(m0 + R), zeros(m0 + R)

@testset "Matrix-vector" begin
    mul!(yref, Aref, xref)
    mul!(y, A, x)
    @test y ≈ yref

    mul!(yref, Aref, xref, 100.0, 10.0)
    mul!(y, A, x, 100.0, 10.0)
    @test y ≈ yref

    y .= 1
    yref .= 1
    mul!(xref, Aref', yref)
    mul!(x, A', y)
    @test x ≈ xref

    y .= 1
    yref .= 1
    mul!(xref, Aref', yref, 100.0, 10.0)
    mul!(x, A', y, 100.0, 10.0)
    @test y ≈ yref
end

end  # testset

# Factorization
include("linear_solver.jl")