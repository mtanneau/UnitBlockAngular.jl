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
    kkt::UnitBlockAngularFactor{Tv},
    θ::AbstractVector{Tv},
    regP::AbstractVector{Tv},
    regD::AbstractVector{Tv}
) where{Tv<:Real}

    m0 = kkt.A.m0
    n0 = kkt.A.n0
    n  = kkt.A.n
    R  = kkt.A.R

    # Copy scaling
    kkt.θ .= θ
    kkt.regP .= regP
    kkt.regD .= regD

    # I. Compute normal equations
    # I.1 - Schur complement `Φ = Σ Bi * Θi * Bi'`
    θ_ = one(Tv) ./ (kkt.θ .+ kkt.regP)

    # Linking block
    @views θ0 = θ_[1:(kkt.A.n0)]
    copyto!(kkt._B0, kkt.A.B0)
    rmul!(kkt._B0, Diagonal(sqrt.(θ0)))
    BLAS.syrk!('U', 'N', one(Tv), kkt._B0, zero(Tv), kkt.C)  # C = B0 * Θ0 * B0'

    # Columns block
    @views θ1 = θ_[(1+kkt.A.n0):end]
    copyto!(kkt._B, kkt.A.B)
    rmul!(kkt._B , Diagonal(sqrt.(θ1)))
    BLAS.syrk!('U', 'N', one(Tv), kkt._B, one(Tv), kkt.C)  # C += Σ Bi * Θi * Bi'
    
    # I.2 Lower block
    kkt.L .= zero(Tv)
    @inbounds for j in 1:n
        z = θ1[j]
        k = kkt.A.blockidx[j]
        @inbounds for i in 1:m0
            kkt.L[i, k] += z * kkt.A.B[i, j]
        end
    end

    # I.2 Compute diagonal
    # Di = Σ { θj | block[j] == i }
    kkt.D .= 0
    @inbounds for (i, θj) in zip(kkt.A.blockidx, θ1)
        kkt.D[i] += θj
    end

    # I.3 Apply dual regularizations
    kkt.D .+= kkt.regD[(kkt.A.m0+1):end]


    # II. Compute the factorization
    kkt._L .= kkt.L
    rmul!(kkt._L, Diagonal(sqrt.(inv.(kkt.D))))
    BLAS.syrk!('U', 'N', -one(Tv), kkt._L, one(Tv), kkt.C)  # C -= (Di)⁻¹ (Biθi) (Biθi)'
    
    # 6. Factorize C
    # Dual regularizations on C
    @inbounds for i in 1:m0
        kkt.C[i, i] += kkt.regD[i]
    end
    _, ic = LAPACK.potrf!('U', kkt.C)
    ic == 0 || throw(PosDefException(ic))

    # Done
    return nothing
end

function Tulip.KKT.solve!(
    dx, dy,
    kkt::UnitBlockAngularFactor{Tv},
    ξp, ξd
) where{Tv<:Real}
    m0 = kkt.A.m0
    R = kkt.A.R
    # TODO: dimension checks

    # 1. Setup right-hand side for normal equations
    dy .= ξp 
    mul!(dy, kkt.A, ξd ./ (kkt.θ .+ kkt.regP), one(Tv), one(Tv))
    @views dy0 = dy[1:m0]
    @views dy1 = dy[(m0+1):end]

    dy1 ./= kkt.D
    mul!(dy0, kkt.L, dy1, -one(Tv), one(Tv))

    # 2. Solve Schur complement
    LAPACK.potrs!('U', kkt.C, dy0)

    # 3. Recover solution to dy1
    dy1 .*= kkt.D
    mul!(dy1, kkt.L', dy0, -one(Tv), one(Tv))
    dy1 ./= kkt.D

    # 4. Recover solution to augmented system
    mul!(dx, kkt.A', dy)
    dx .-= ξd
    dx ./= kkt.θ .+ kkt.regP

    return nothing
end