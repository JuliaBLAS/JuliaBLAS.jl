using SIMD
# using CpuId
import Base.Cartesian: @nexprs
using StaticArrays: MMatrix

@inline micro_ker!(blk, args...) = blk.generic ? _generic_micro_ker!(blk,args...) : _simd_micro_ker!(blk,args...)

struct Block{T1,T2,T3,T4}
    Ac::T1
    Bc::T2
    AB::T3
    C::T4
    mc::Int
    kc::Int
    nc::Int
    mr::Int
    nr::Int
    inc1A::Int
    inc2A::Int
    inc1B::Int
    inc2B::Int
    inc1C::Int
    inc2C::Int
    generic::Bool
end
function Block(A::X, B::W, C::Z, generic) where {X, W, Z}
    mr=8; nr=6
    m, n = size(C)
    mc = min(512, mr*cld(m,mr))
    nc = min(516, nr*cld(n,nr))
    kc = min(1024, size(A,2))
    T = promote_type(eltype(X), eltype(W), eltype(Z))
    Ac = Matrix{T}(undef, mr, mc*kc÷mr)
    Bc = Matrix{T}(undef, nr, kc*nc÷nr)
    AB = zero(MMatrix{mr, nr , T})
    Block{typeof(Ac),typeof(Bc),typeof(AB),typeof(C)}(Ac, Bc, AB, C, mc, kc, nc, mr, nr,
                                                      strides(A)..., strides(B)..., strides(C)...,
                                                      generic)
end


function mymuladd!(C, A, B, generic::Bool=false)
    mymuladd!(C, A, B, Block(A, B, C, generic))
end

function mymuladd!(C, A, B, blk::Block{T1,T2,T3,T4}=Block(A, B, C, false)) where {T1,T2,T3,T4}
    m,  k = size(A); _k, n = size(B)
    @assert k == _k
    _m, _n = size(C)
    @assert m == _m && n == _n
    mb, _mc = cld(m, blk.mc), m % blk.mc
    nb, _nc = cld(n, blk.nc), n % blk.nc
    kb, _kc = cld(k, blk.kc), k % blk.kc
    for j in 1:nb # Loop 5
        nc = (j!=nb || _nc==0) ? blk.nc : _nc
        for l in 1:kb # Loop 4
            kc = (l!=kb || _kc==0) ? blk.kc : _kc
            #_β = l==1 ? β : 1.0
            offsetB = blk.kc*(l-1)*blk.inc1B + blk.nc*(j-1)*blk.inc2B
            pack_B!(blk, B, kc, nc, offsetB)
            for i in 1:mb # Loop 3
                mc = (i!=mb || _mc==0) ? blk.mc : _mc
                offsetA = blk.mc*(i-1)*blk.inc1A + blk.kc*(l-1)*blk.inc2A
                offsetC = blk.mc*(i-1)*blk.inc1C + blk.nc*(j-1)*blk.inc2C
                pack_A!(blk, A, mc, kc, offsetA)
                macro_ker!(blk, C, mc, nc, kc, offsetC)
            end # Loop 3
        end # Loop 4
    end # Loop 5
    C
end

@inline function pack_MRxK!(blk::Block{T1,T2,T3,T4}, A, k::Int,
                            offsetA::Int, offsetAc::Int) where {T1,T2,T3,T4}
    @inbounds for j in 1:k
        for i in 1:blk.mr
            blk.Ac[offsetAc+i] = A[offsetA + (i-1)*blk.inc1A + 1]
        end
        offsetAc += blk.mr
        offsetA  += blk.inc2A
    end
    return nothing
end

function pack_A!(blk::Block{T1,T2,T3,T4}, A, mc::Int, kc::Int,
                 offsetA::Int) where {T1,T2,T3,T4}
    mp, _mr = divrem(mc, blk.mr)
    offsetAc = 0
    for i in 1:mp
        pack_MRxK!(blk, A, kc, offsetA, offsetAc)
        offsetAc += kc*blk.mr
        offsetA  += blk.mr*blk.inc1A
    end
    if _mr > 0
        @inbounds for j in 1:kc
            for i in 1:_mr
                blk.Ac[offsetAc+i] = A[offsetA + (i-1)*blk.inc1A + 1]
            end
            for i in _mr+1:blk.mr
                blk.Ac[offsetAc+i] = zero(eltype(A))
            end
            offsetAc += blk.mr
            offsetA  += blk.inc2A
        end
    end
    return nothing
end

@inline function pack_KxNR!(blk::Block{T1,T2,T3,T4}, B, k::Int,
                            offsetB::Int, offsetBc::Int) where {T1,T2,T3,T4}
    @inbounds for i = 1:k
        for j = 1:blk.nr
            blk.Bc[offsetBc+j] = B[offsetB + (j-1)*blk.inc2B + 1]
        end
        offsetBc += blk.nr
        offsetB  += blk.inc1B
    end
    return nothing
end

function pack_B!(blk::Block{T1,T2,T3,T4}, B,
                 kc::Int, nc::Int, offsetB::Int) where {T1,T2,T3,T4}
    np, _nr = divrem(nc, blk.nr)
    offsetBc = 0
    for j in 1:np
        pack_KxNR!(blk, B, kc, offsetB, offsetBc)
        offsetBc += kc*blk.nr
        offsetB  += blk.nr*blk.inc2B
    end
    if _nr > 0
        @inbounds for i in 1:kc
            for j in 1:_nr
                blk.Bc[offsetBc+j] = B[offsetB + (j-1)*blk.inc2B + 1]
            end
            for j in _nr+1:blk.nr
                blk.Bc[offsetBc+j] = zero(eltype(B))
            end
            offsetBc += blk.nr
            offsetB  += blk.inc1B
        end
    end
    return nothing
end

@inline function macro_ker!(blk::Block{T1,T2,T3,T4}, C, mc::Int, nc::Int, kc::Int,
                            offsetC::Int) where {T1,T2,T3,T4}
    mp, _mr = cld(mc, blk.mr), mc % blk.mr
    np, _nr = cld(nc, blk.nr), nc % blk.nr
    for j in 1:np
        nr = (j!=np || _nr==0) ? blk.nr : _nr
        for i in 1:mp
            mr = (i!=mp || _mr==0) ? blk.mr : _mr
            offsetA = (i-1)*kc*blk.mr
            offsetB = (j-1)*kc*blk.nr
            if mr == blk.mr && nr==blk.nr
                micro_ker!(blk, kc, offsetA, offsetB, offsetC+(i-1)*blk.mr*blk.inc1C + (j-1)*blk.nr*blk.inc2C, Val(true))
            else
                micro_ker!(blk, kc, offsetA, offsetB, 0, Val(false))
                _axpy!(C, 1, blk.AB, mr, nr, offsetC+(i-1)*blk.mr*blk.inc1C + (j-1)*blk.nr*blk.inc2C,
                       0, 1, blk.mr)
            end
        end
    end
    return nothing
end

@inline function _generic_micro_ker!(blk::Block{T1,T2,T3,T4}, kc::Int,
                                     offsetA::Int, offsetB::Int, offsetC::Int,
                                     ::Val{loadC}) where {T1,T2,T3,T4,loadC}
    fill!(blk.AB, zero(eltype(blk.AB)))
    @inbounds for k in 1:kc
        for j in 1:blk.nr, i in 1:blk.mr
            blk.AB[i + (j-1)*blk.mr] += blk.Ac[offsetA+i] * blk.Bc[offsetB+j]
        end
        offsetA += blk.mr
        offsetB += blk.nr
    end
    if loadC
        @inbounds for j in 1:blk.nr, i in 1:blk.mr
            blk.C[offsetC+(i-1)*blk.inc1C+(j-1)*blk.inc2C+1] += blk.AB[i + (j-1)*blk.mr]
        end
    end
    return nothing
end

function kernel_quote(T1, MR, NR, loadC)
    quote
        @inbounds begin
            pA, pAB = pointer(blk.Ac), pointer(blk.AB)
            T = eltype($T1); siz = sizeof(T)
            VT = Vec{$MR, T}
            if $loadC
                pC = pointer(blk.C)
                @nexprs $NR i -> ab_i = vload(VT, pC + (offsetC+(i-1)*blk.inc2C)siz)
            else
                @nexprs $NR i -> ab_i = zero(VT)
            end
            for k in 1:kc
                a1 = vload(VT, pA + (offsetA+(k-1)blk.mr)siz)
                @nexprs $NR i -> begin
                    b_i = VT(blk.Bc[offsetB+(k-1)blk.nr+i])
                    ab_i = muladd(a1, b_i, ab_i)
                end
            end
            if $loadC
                @nexprs $NR i -> vstore(ab_i, pC + (offsetC+(i-1)*blk.inc2C)siz)
            else
                @nexprs $NR i -> vstore(ab_i, pAB + (i-1)blk.mr*siz)
            end
            return nothing
        end
    end
end

@generated function _simd_micro_ker!(blk::Block{T1,T2,T3,T4}, kc::Int,
                                     offsetA::Int, offsetB::Int, offsetC::Int,
                                     ::Val{loadC}) where {T1,T2,T3,T4,loadC}
    expr = kernel_quote(T1, 8, 6, loadC)
    quote
        $(Expr(:meta, :inline))
        @assert blk.mr == 8 && blk.nr == 6
        $expr
    end
end

@inline function _axpy!(Y, α, X, m::Int, n::Int,
                        offsetY::Int, offsetX::Int, inc1X::Int, inc2X::Int)
    inc1Y, inc2Y = stride(Y, 1), stride(Y, 2)
    @inbounds for j in 1:n, i in 1:m
        Y[offsetY+(i-1)*inc1Y+(j-1)*inc2Y+1] += α*X[offsetX+(i-1)*inc1X+(j-1)*inc2X+1]
    end
    return nothing
end
