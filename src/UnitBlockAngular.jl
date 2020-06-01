module UnitBlockAngular

export UnitBlockAngularMatrix, UnitBlockAngularFactor

using LinearAlgebra

# Base Interface
import Base:
    size, getindex

import LinearAlgebra.mul!

"""
    UnitBlockAngularMatrix{Tv}

Unit block-angular matrix.

A unit block-angular matrix is a matrix of the form
```
A = [ B0  B1  B2  ...  Br ]
    |  0  e'              |
    |  0      e'          |
    |  0          ...     |
    [  0               e' ]
```
where `Bk` has size `m x nk`.
"""
struct UnitBlockAngularMatrix{Tv<:Real} <: AbstractMatrix{Tv}

    m0::Int  # number of linking constraints
    n0::Int  # Number of linking columns
    n::Int   # number of non-linking columns

    M::Int  # Total number of constraints
    N::Int  # Total number of columns
    R::Int  # Number of blocks

    B0::Matrix{Tv}         # Linking block
    B::Matrix{Tv}          # Lower blocks, concatenated
    blockidx::Vector{Int}  # Block index of each column in `B`

    function UnitBlockAngularMatrix(
        B0::AbstractMatrix{Tv},
        B::AbstractMatrix{Tv},
        R::Int, blockidx::Vector{Int}
    ) where{Tv<:Real}

        # Sanity checks
        m, n = size(B)
        m0, n0 = size(B0)
        m == m0 || throw(DimensionMismatch(
            "B has $m rows, but B0 has $(m0)"
        ))
        n == length(blockidx) || throw(DimensionMismatch(
            "B has $n columns, but blockidx has $(length(blockidx))"
        ))

        if n > 0
            imin, imax = extrema(blockidx[1:n])
            (0 <= imin <= imax <= R) || throw(DimensionMismatch(
                "Specified $R blocks but indices are in range [$imin, $imax]"
            ))
        end

        A = new{Tv}(m0, n0, n, m0 + R, n0+n, R,
            Matrix{Tv}(undef, m0, n0),
            Matrix{Tv}(undef, m, n),
            Vector{Int}(undef, n)
        )

        A.B0 .= B0
        A.B  .= B
        A.blockidx .= blockidx
        return A
    end

    function UnitBlockAngularMatrix{Tv}(m0, n0, n, R) where{Tv<:Real}
        return new{Tv}(m0, n0, n, m0+R, n0+n, R, Matrix{Tv}(undef, m0, n0), Matrix{Tv}(undef, m0, n), Vector{Int}(undef, n))
    end
end

# Base matrix interface
size(A::UnitBlockAngularMatrix) = (A.M, A.N)

function getindex(A::UnitBlockAngularMatrix{Tv}, i::Integer, j::Integer) where Tv<:Real
    
    # Sanity check
    ((1 <= i <= A.M) && (1 <= j <= A.N)) || throw(BoundsError())
 
    if (i <= A.m0) && (j <= A.n0)
        # Element is in linking block
        return A.B0[i, j]
    
    elseif (i <= A.m0) && (j > A.n0)
        # Element is in regular block
        return A.B[i, j-A.n0]
    
    elseif (i > A.m0) && (j <= A.n0)
        return zero(Tv)
    
    elseif (i > A.m0) && (j > A.n0)
        # Check if column j belongs to block i
        if A.blockidx[j-A.n0] == i - A.m0
            return oneunit(Tv)
        else
            return zero(Tv)
        end
    end 
end

# Matrix-vector products
function LinearAlgebra.mul!(
    y::AbstractVector{Tv},
    A::UnitBlockAngularMatrix{Tv},
    x::AbstractVector{Tv},
    α::Tv, β::Tv
) where{Tv<:Real}
    # Dimensions checks
    m, n = size(A)
    n == length(x) || throw(DimensionMismatch("A has size $((m, n)) but x has length $(length(x))"))
    m == length(y) || throw(DimensionMismatch("A has size $((m, n)) but y has length $(length(y))"))
    
    # Compute `y0 = α * A * x + β * y0`
    # `y0 = α * B0 * x0 + α B * [x1 ... xR] + β y0`
    if iszero(β)
        # We do this explicity in case `y` has not been initialized
        # Otherwise, we may have some NaN values going around
        y .= zero(Tv)
    end

    @views x0 = x[1:A.n0]
    @views x1 = x[(A.n0+1):end]
    @views y0 = y[1:A.m0]
    @views y1 = y[(A.m0+1):end]
    mul!(y0, A.B0, x0, α, β)
    mul!(y0, A.B, x1, α,  one(Tv))

    # Compute the remaining part
    y1 .*= β
    @inbounds for (j, xj) in zip(A.blockidx, x1)
        y1[j] += α * xj
    end

    return y
end

mul!(y::AbstractVector{Tv}, A::UnitBlockAngularMatrix{Tv}, x::AbstractVector{Tv}) where{Tv<:Real} = mul!(y, A, x, one(Tv), zero(Tv))

function mul!(
    x::AbstractVector{Tv},
    At::Union{
        LinearAlgebra.Transpose{Tv, UnitBlockAngularMatrix{Tv}},
        LinearAlgebra.Adjoint{Tv, UnitBlockAngularMatrix{Tv}}
    },
    y::AbstractVector{Tv},
    α::Tv,
    β::Tv
) where{Tv<:Real}
    A = At.parent

    m, n = size(A)
    n == length(x) || throw(DimensionMismatch(
        "A has size $(size(A)) but x has size $(length(x))")
    )
    m == length(y) || throw(DimensionMismatch(
        "A has size $(size(A)) but y has size $(length(y))")
    )

    # `x = B' * y0`
    @views x0 = x[1:A.n0]
    @views x1 = x[(A.n0+1):end]
    @views y0 = y[1:A.m0]
    @views y1 = y[(A.m0+1):end]
    mul!(x0, A.B0', y0, α, β)
    mul!(x1, A.B' , y0, α, β)

    # `∀i, x[i] += y[blockidx[i]]`
    @inbounds for (i, j) in enumerate(A.blockidx)
        x1[i] += α * y1[j]
    end

    return x
end

mul!(
    x::AbstractVector{Tv},
    At::Union{
        LinearAlgebra.Transpose{Tv, UnitBlockAngularMatrix{Tv}},
        LinearAlgebra.Adjoint{Tv, UnitBlockAngularMatrix{Tv}}
    },
    y::AbstractVector{Tv}
) where{Tv<:Real} = mul!(x, At, y, one(Tv), zero(Tv))

import Tulip.TLA.construct_matrix

function construct_matrix(
    ::Type{UnitBlockAngularMatrix}, M::Int, N::Int,
    aI::Vector{Int}, aJ::Vector{Int}, aV::Vector{Tv};
    m0::Int, n0::Int, n::Int, R::Int
) where{Tv<:Real}
    A = UnitBlockAngularMatrix{Tv}(m0, n0, n, R)
    A.B .= 0
    A.B0 .= 0
    A.blockidx .= 0
    # TODO: may be more efficient to first sort indices so that
    # A is accessed in column-major order.
    for(i, j, v) in zip(aI, aJ, aV)
        if 1 <= i <= m0
            # Linking constraints
            if 1 <= j <= n0
                # Update B0
                A.B0[i, j] = v
            elseif n0 < j <= n0 + n
                # Update B
                A.B[i, j - n0] = v
            else
                error("Invalid column index $j")
            end
        elseif m0 < i <= m0 + R
            # Convexity constraint
            n0 < j || error("Non-zero coefficient ($i, $j) (should be zero)")
            isone(v) || error("Invalid coefficient A[$i, $j]=$v (should be one)")

            A.blockidx[j - n0] = i - m0
        else
            error("Invalid row index $i in matrix with $m0 linking and $R convexity constraints")
        end
    end
    return A
end


include("KKTSolver.jl")

end # module
