# TODO: replace this by a linear solver
import Tulip

import LinearAlgebra.LAPACK


"""
    UnitBlockAngularFactor{Tv}


"""
struct UnitBlockAngularFactor{Tv<:Real} <: Tulip.KKT.AbstractKKTSolver{Tv}
    
    A::UnitBlockAngularMatrix{Tv}

    θ::Vector{Tv}     # Diagonal scaling
    regP::Vector{Tv}  # Primal regularization
    regD::Vector{Tv}  # Dual   regularization
    
    D::Vector{Tv}  # Diagonal elements of Cholesky factor (n x 1)
    L::Matrix{Tv}  # Lower block of Cholesky factor (m0 x n)
    C::Matrix{Tv}  # Dense Schur complement (m0 x m0)

    # Local copies, needed to compute rank-k updates of the form C += M*Θ*M'
    _B0::Matrix{Tv}
    _B::Matrix{Tv}
    _L::Matrix{Tv}

    function UnitBlockAngularFactor(A::UnitBlockAngularMatrix{Tv}) where{Tv<:Real}
        return new{Tv}(
            A, Vector{Tv}(undef, A.N, ), Vector{Tv}(undef, A.N), Vector{Tv}(undef, A.M),
            Vector{Tv}(undef, A.R), Matrix{Tv}(undef, A.m0, A.R), Matrix{Tv}(undef, A.m0, A.m0),
            Matrix{Tv}(undef, A.m0, A.n0), Matrix{Tv}(undef, A.m0, A.n), Matrix{Tv}(undef, A.m0, A.R)
        )
    end
end

Tulip.KKT.backend(::UnitBlockAngularFactor) = "UnitBlockAngular"
Tulip.KKT.linear_system(::UnitBlockAngularFactor) = "Normal equations"

function Tulip.KKT.update!(
    ls::UnitBlockAngularFactor{Tv},
    θ::AbstractVector{Tv},
    regP::AbstractVector{Tv},
    regD::AbstractVector{Tv}
) where{Tv<:Real}

    m0 = ls.A.m0
    n0 = ls.A.n0
    n  = ls.A.n
    R  = ls.A.R

    # Copy scaling
    ls.θ .= θ
    ls.regP .= regP
    ls.regD .= regD

    # I. Compute normal equations
    # I.1 - Schur complement `Φ = Σ Bi * Θi * Bi'`
    θ_ = one(Tv) ./ (ls.θ .+ ls.regP)

    # Linking block
    @views θ0 = θ_[1:(ls.A.n0)]
    copyto!(ls._B0, ls.A.B0)
    rmul!(ls._B0, Diagonal(sqrt.(θ0)))
    BLAS.syrk!('U', 'N', one(Tv), ls._B0, zero(Tv), ls.C)  # C = B0 * Θ0 * B0'

    # Columns block
    @views θ1 = θ_[(1+ls.A.n0):end]
    copyto!(ls._B, ls.A.B)
    rmul!(ls._B , Diagonal(sqrt.(θ1)))
    BLAS.syrk!('U', 'N', one(Tv), ls._B, one(Tv), ls.C)  # C += Σ Bi * Θi * Bi'
    
    # I.2 Lower block
    ls.L .= zero(Tv)
    @inbounds for j in 1:n
        z = θ1[j]
        k = ls.A.blockidx[j]
        @inbounds for i in 1:m0
            ls.L[i, k] += z * ls.A.B[i, j]
        end
    end

    # I.2 Compute diagonal
    # Di = Σ { θj | block[j] == i }
    ls.D .= 0
    @inbounds for (i, θj) in zip(ls.A.blockidx, θ1)
        ls.D[i] += θj
    end

    # I.3 Apply dual regularizations
    ls.D .+= ls.regD[(ls.A.m0+1):end]

    
    # II. Compute the factorization
    ls._L .= ls.L
    rmul!(ls._L, Diagonal(sqrt.(inv.(ls.D))))
    BLAS.syrk!('U', 'N', -one(Tv), ls._L, one(Tv), ls.C)  # C -= (Di)⁻¹ (Biθi) (Biθi)'
    
    # 6. Factorize C
    # Dual regularizations on C
    @inbounds for i in 1:m0
        ls.C[i, i] += ls.regD[i]
    end
    LAPACK.potrf!('U', ls.C)

    # Done
    return nothing
end

function Tulip.KKT.solve!(
    dx, dy,
    ls::UnitBlockAngularFactor{Tv},
    ξp, ξd
) where{Tv<:Real}
    m0 = ls.A.m0
    R = ls.A.R
    # TODO: dimension checks

    # 1. Setup right-hand side for normal equations
    dy .= ξp 
    mul!(dy, ls.A, ξd ./ (ls.θ .+ ls.regP), one(Tv), one(Tv))
    @views dy0 = dy[1:m0]
    @views dy1 = dy[(m0+1):end]

    dy1 ./= ls.D
    mul!(dy0, ls.L, dy1, -one(Tv), one(Tv))

    # 2. Solve Schur complement
    LAPACK.potrs!('U', ls.C, dy0)

    # 3. Recover solution to dy1
    dy1 .*= ls.D
    mul!(dy1, ls.L', dy0, -one(Tv), one(Tv))
    dy1 ./= ls.D

    # 4. Recover solution to augmented system
    dx .= (ls.A' * dy - ξd) ./ (ls.θ .+ ls.regP)

    return nothing
end