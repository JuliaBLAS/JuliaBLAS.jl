using SIMD
using Base.Cartesian: @nexprs

struct Block{T1,T2,T3,T4,G}
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
end

const Ac = Vector{UInt8}(undef, 110592)
const Bc = Vector{UInt8}(undef, 6266880)
const AB = Vector{UInt8}(undef, 12*4*8)

function Block(A::X, B::W, C::Z, generic) where {X, W, Z}
    global Ac, Bc, AB
    mr=12; nr=4
    m, n = size(C)
    mc = 72
    kc = 192
    nc = 4080
    T = promote_type(eltype(X), eltype(W), eltype(Z))
    siz = sizeof(T)
    _Ac = unsafe_wrap(Array, Ptr{T}(pointer(Ac)), length(Ac)÷siz)
    _Bc = unsafe_wrap(Array, Ptr{T}(pointer(Bc)), length(Bc)÷siz)
    _AB = unsafe_wrap(Array, Ptr{T}(pointer(AB)), length(AB)÷siz)
    Block{typeof(_Ac),typeof(_Bc),typeof(_AB),typeof(C),generic}(_Ac, _Bc, _AB, C, mc, kc, nc, mr, nr,
                                                      strides(A)..., strides(B)..., strides(C)...,)
end

"""
    addmul!(C, A, B, blk::Block{T1,T2,T3,T4,G}=Block(A, B, C, false)) -> C

`addmul!` computs ``C = AB + C``, where ``A``, ``B``, and ``C`` are matrices.
"""
function addmul!(C, A, B, blk::Block{T1,T2,T3,T4,G}=Block(A, B, C, false)) where {T1,T2,T3,T4,G}
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

@inline function pack_MRxK!(blk::Block{T1,T2,T3,T4,G}, A, k::Int,
                            offsetA::Int, offsetAc::Int) where {T1,T2,T3,T4,G}
    @inbounds for j in 1:k
        for i in 1:blk.mr
            blk.Ac[offsetAc+i] = A[offsetA + (i-1)*blk.inc1A + 1]
        end
        offsetAc += blk.mr
        offsetA  += blk.inc2A
    end
    return nothing
end

function pack_A!(blk::Block{T1,T2,T3,T4,G}, A, mc::Int, kc::Int,
                 offsetA::Int) where {T1,T2,T3,T4,G}
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

@inline function pack_KxNR!(blk::Block{T1,T2,T3,T4,G}, B, k::Int,
                            offsetB::Int, offsetBc::Int) where {T1,T2,T3,T4,G}
    @inbounds for i = 1:k
        for j = 1:blk.nr
            blk.Bc[offsetBc+j] = B[offsetB + (j-1)*blk.inc2B + 1]
        end
        offsetBc += blk.nr
        offsetB  += blk.inc1B
    end
    return nothing
end

function pack_B!(blk::Block{T1,T2,T3,T4,G}, B,
                 kc::Int, nc::Int, offsetB::Int) where {T1,T2,T3,T4,G}
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

@inline function macro_ker!(blk::Block{T1,T2,T3,T4,G}, C, mc::Int, nc::Int, kc::Int,
                            offsetC::Int) where {T1,T2,T3,T4,G}
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

@inline function micro_ker!(blk::Block{T1,T2,T3,T4,true}, kc::Int,
                            offsetA::Int, offsetB::Int, offsetC::Int,
                            ::Val{loadC}) where {T1,T2,T3,T4,G,loadC}
    @inbounds begin
        fill!(blk.AB, zero(eltype(blk.AB)))
        for k in 1:kc
            for j in 1:blk.nr, i in 1:blk.mr
                blk.AB[i + (j-1)*blk.mr] += blk.Ac[offsetA+i] * blk.Bc[offsetB+j]
            end
            offsetA += blk.mr
            offsetB += blk.nr
        end
        if loadC
            for j in 1:blk.nr, i in 1:blk.mr
                blk.C[offsetC+(i-1)*blk.inc1C+(j-1)*blk.inc2C+1] += blk.AB[i + (j-1)*blk.mr]
            end
        end
    end
    return nothing
end

@inline function micro_ker!(blk::Block{T1,T2,T3,T4,false}, kc::Int,
                            offsetA::Int, offsetB::Int, offsetC::Int,
                            ::Val{loadC}) where {T1,T2,T3,T4,G,loadC}
    width = 4
    nr = 4
    mr = 12÷width
    @inbounds begin
        pA, pAB = pointer(blk.Ac), pointer(blk.AB)
        T = eltype(T1)
        siz = sizeof(T)
        VT = Vec{width, T}
        pC = pointer(blk.C)
        @nexprs 4 i -> #= nr =#
            @nexprs 3 j -> #= mr =#
                ab_i_j = loadC ? vload(VT, pC + (offsetC+(i-1)*blk.inc2C+(j-1)*width)siz) :
                                 zero(VT)
        for k in 1:kc
            @nexprs 3 j -> #= mr =# a_j = vload(VT, pA + (offsetA+blk.mr*(k-1)+(j-1)width)siz)
            @nexprs 4 i -> begin # nr
                b_i = VT(blk.Bc[offsetB+(k-1)blk.nr+i])
                @nexprs 3 j -> #= mr =# ab_i_j = muladd(a_j, b_i, ab_i_j)
            end
        end
        @nexprs 4 i -> #= nr =# @nexprs 3 j -> begin # mr
            ptr = loadC ? pC + (offsetC+(i-1)*blk.inc2C+(j-1)*width)siz :
                          pAB + ((i-1)blk.mr + (j-1)*width)*siz
            vstore(ab_i_j, ptr)
        end
    end
    return nothing
end

@inline function _axpy!(Y, α, X, m::Int, n::Int,
                        offsetY::Int, offsetX::Int, inc1X::Int, inc2X::Int)
    inc1Y, inc2Y = stride(Y, 1), stride(Y, 2)
    @inbounds for j in 1:n, i in 1:m
        Y[offsetY+(i-1)*inc1Y+(j-1)*inc2Y+1] += α*X[offsetX+(i-1)*inc1X+(j-1)*inc2X+1]
    end
    return nothing
end
