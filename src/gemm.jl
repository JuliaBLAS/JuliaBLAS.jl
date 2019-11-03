using SIMD
using Base.Cartesian: @nexprs

struct Block{W,MR,NR,G,TA,TB,TAB,TC}
    Ac::TA
    Bc::TB
    AB::TAB
    C::TC
    mc::Int
    kc::Int
    nc::Int
    inc1A::Int
    inc2A::Int
    inc1B::Int
    inc2B::Int
    inc1C::Int
    inc2C::Int
end

const Ac = Vector{UInt8}(undef, 110592)
const Bc = Vector{UInt8}(undef, 6266880)
const AB = Vector{UInt8}(undef, 1 << 12)

function Block(A::X, B::Y, C::Z, ::Val{Width}, ::Val{MR}, ::Val{NR}, ::Val{Generic}) where {X,Y,Z,Width,MR,NR,Generic}
    global Ac, Bc, AB
    m, n = size(C)
    mc = 72
    kc = 192
    nc = 4080
    T = promote_type(eltype(X), eltype(Y), eltype(Z))
    siz = sizeof(T)
    _Ac = unsafe_wrap(Array, Ptr{T}(pointer(Ac)), length(Ac)÷siz)
    _Bc = unsafe_wrap(Array, Ptr{T}(pointer(Bc)), length(Bc)÷siz)
    _AB = unsafe_wrap(Array, Ptr{T}(pointer(AB)), length(AB)÷siz)
    Block{Width,MR,NR,Generic,typeof(_Ac),typeof(_Bc),typeof(_AB),typeof(C)}(_Ac, _Bc, _AB, C, mc, kc, nc, strides(A)..., strides(B)..., strides(C)...,)
end
getmr(blk::Block{W,MR,NR}) where {W,MR,NR} = MR
getnr(blk::Block{W,MR,NR}) where {W,MR,NR} = NR

"""
    addmul!(C, A, B, blk::Block=Block(A, B, C, false)) -> C

`addmul!` computes ``C = AB + C``, where ``A``, ``B``, and ``C`` are matrices.
"""
function addmul!(C, A, B; width=Val(4), mr=Val(12), nr=Val(4), generic=Val(false))
    blk = Block(A, B, C, width, mr, nr, generic)
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

@inline function pack_MRxK!(blk::Block, A, k::Int, offsetA::Int, offsetAc::Int)
    mr = getmr(blk)
    @inbounds for j in 1:k
        for i in 1:mr
            blk.Ac[offsetAc+i] = A[offsetA + (i-1)*blk.inc1A + 1]
        end
        offsetAc += mr
        offsetA  += blk.inc2A
    end
    return nothing
end

function pack_A!(blk::Block, A, mc::Int, kc::Int, offsetA::Int)
    mr = getmr(blk)
    mp, _mr = divrem(mc, mr)
    offsetAc = 0
    for i in 1:mp
        pack_MRxK!(blk, A, kc, offsetA, offsetAc)
        offsetAc += kc*mr
        offsetA  += mr*blk.inc1A
    end
    if _mr > 0
        @inbounds for j in 1:kc
            for i in 1:_mr
                blk.Ac[offsetAc+i] = A[offsetA + (i-1)*blk.inc1A + 1]
            end
            for i in _mr+1:mr
                blk.Ac[offsetAc+i] = zero(eltype(A))
            end
            offsetAc += mr
            offsetA  += blk.inc2A
        end
    end
    return nothing
end

@inline function pack_KxNR!(blk::Block, B, k::Int, offsetB::Int, offsetBc::Int)
    nr = getnr(blk)
    @inbounds for i = 1:k
        for j = 1:nr
            blk.Bc[offsetBc+j] = B[offsetB + (j-1)*blk.inc2B + 1]
        end
        offsetBc += nr
        offsetB  += blk.inc1B
    end
    return nothing
end

function pack_B!(blk::Block, B, kc::Int, nc::Int, offsetB::Int)
    nr = getnr(blk)
    np, _nr = divrem(nc, nr)
    offsetBc = 0
    for j in 1:np
        pack_KxNR!(blk, B, kc, offsetB, offsetBc)
        offsetBc += kc*nr
        offsetB  += nr*blk.inc2B
    end
    if _nr > 0
        @inbounds for i in 1:kc
            for j in 1:_nr
                blk.Bc[offsetBc+j] = B[offsetB + (j-1)*blk.inc2B + 1]
            end
            for j in _nr+1:nr
                blk.Bc[offsetBc+j] = zero(eltype(B))
            end
            offsetBc += nr
            offsetB  += blk.inc1B
        end
    end
    return nothing
end

@inline function macro_ker!(blk::Block, C, mc::Int, nc::Int, kc::Int, offsetC::Int)
    mr′, nr′ = getmr(blk), getnr(blk)
    mp, _mr = cld(mc, mr′), mc % mr′
    np, _nr = cld(nc, nr′), nc % nr′
    for j in 1:np
        nr = (j!=np || _nr==0) ? nr′ : _nr
        for i in 1:mp
            mr = (i!=mp || _mr==0) ? mr′ : _mr
            offsetA = (i-1)*kc*mr′
            offsetB = (j-1)*kc*nr′
            if mr == mr′ && nr == nr′
                micro_ker!(blk, kc, offsetA, offsetB, offsetC+(i-1)*mr′*blk.inc1C + (j-1)*nr′*blk.inc2C, Val(true))
            else
                micro_ker!(blk, kc, offsetA, offsetB, 0, Val(false))
                _axpy!(C, 1, blk.AB, mr, nr, offsetC+(i-1)*mr′*blk.inc1C + (j-1)*nr′*blk.inc2C,
                       0, 1, mr′)
            end
        end
    end
    return nothing
end

# generic
@inline function micro_ker!(blk::Block{<:Any,<:Any,<:Any,true}, kc::Int, offsetA::Int, offsetB::Int, offsetC::Int, ::Val{loadC}) where loadC
    mr, nr = getmr(blk), getnr(blk)
    @inbounds begin
        fill!(blk.AB, zero(eltype(blk.AB)))
        for k in 1:kc
            for j in 1:nr, i in 1:mr
                blk.AB[i + (j-1)*mr] += blk.Ac[offsetA+i] * blk.Bc[offsetB+j]
            end
            offsetA += mr
            offsetB += nr
        end
        if loadC
            for j in 1:nr, i in 1:mr
                blk.C[offsetC+(i-1)*blk.inc1C+(j-1)*blk.inc2C+1] += blk.AB[i + (j-1)*mr]
            end
        end
    end
    return nothing
end

# SIMD
@inline @generated function micro_ker!(blk::Block{Width,MR,NR,false,T1}, kc::Int, # TODO: mixed-precision
                            offsetA::Int, offsetB::Int, offsetC::Int,
                            ::Val{loadC}) where {Width,MR,NR,Generic,T1,loadC}
    mr = MR÷Width
    quote
        @inbounds begin
            pA, pAB = pointer(blk.Ac), pointer(blk.AB)
            T = eltype(T1)
            siz = sizeof(T)
            VT = Vec{$Width, T}
            pC = pointer(blk.C)
            @nexprs $NR i ->
                @nexprs $mr j ->
                    ab_i_j = loadC ? vload(VT, pC + (offsetC+(i-1)*blk.inc2C+(j-1)*$Width)siz) :
                                     zero(VT)
            for k in 1:kc
                @nexprs $mr j -> a_j = vload(VT, pA + (offsetA+$MR*(k-1)+(j-1)*$Width)siz)
                @nexprs $NR i -> begin
                    b_i = VT(blk.Bc[offsetB+(k-1)*$NR+i])
                    @nexprs 3 j -> ab_i_j = muladd(a_j, b_i, ab_i_j)
                end
            end
            @nexprs $NR i ->
                @nexprs $mr j -> begin
                    ptr = loadC ? pC + (offsetC+(i-1)*blk.inc2C+(j-1)*$Width)siz :
                                  pAB + ((i-1)*$MR + (j-1)*$Width)*siz
                    vstore(ab_i_j, ptr)
                end
        end
        return nothing
    end #quote
end

@inline function _axpy!(Y, α, X, m::Int, n::Int,
                        offsetY::Int, offsetX::Int, inc1X::Int, inc2X::Int)
    inc1Y, inc2Y = stride(Y, 1), stride(Y, 2)
    @inbounds for j in 1:n, i in 1:m
        Y[offsetY+(i-1)*inc1Y+(j-1)*inc2Y+1] += α*X[offsetX+(i-1)*inc1X+(j-1)*inc2X+1]
    end
    return nothing
end
