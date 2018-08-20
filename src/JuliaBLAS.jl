module JuliaBLAS

using SIMD, CpuId

struct Block{T,MC,KC,NC,MR,NR}
    Ac::T
    Bc::T
    AB::T
    Cc::T
    #mc::Int
    #kc::Int
    #nc::Int
    #mr::Int
    #nr::Int
end
function Block(A::X, B::W, C::Z; mr=8, nr=4, mc=32, kc=32, nc=32) where {X, W, Z}# TODO: smarter block size
    T = promote_type(eltype(X), eltype(W), eltype(Z))
    Ac = Vector{T}(undef, kc*mc)
    Bc = Vector{T}(undef, kc*nc)
    AB = valloc(T, 8, mr*nr)
    Cc = valloc(T, 8, mr*nr)
    Block{Vector{T}, mc, kc, nc, mr, nr}(Ac, Bc, AB, Cc)
end

function mymul!(C, A, B, blk::Block{T,MC,KC,NC,MR,NR}=Block(A, B, C)) where {T,MC,KC,NC,MR,NR}
    m,  k = size(A); _k, n = size(B)
    @assert k == _k
    _m, _n = size(C)
    @assert m == _m && n == _n
    mb, _mc = cld(m, MC), m % MC
    nb, _nc = cld(n, NC), n % NC
    kb, _kc = cld(k, KC), k % KC
    for j in 1:nb # Loop 5
        nc = (j!=nb || _nc==0) ? NC : _nc
        for l in 1:kb # Loop 4
            kc = (l!=kb || _kc==0) ? KC : _kc
            #_β = l==1 ? β : 1.0
            offsetB = offsetM(B, KC*(l-1), NC*(j-1))
            pack_B!(blk, B, kc, nc, offsetB)
            for i in 1:mb # Loop 3
                mc = (i!=mb || _mc==0) ? MC : _mc
                offsetA = offsetM(A, MC*(i-1), KC*(l-1))
                offsetC = offsetM(C, MC*(i-1), NC*(j-1))
                pack_A!(blk, A, mc, kc, offsetA)
                macro_ker!(blk, C, mc, nc, kc, offsetC)
            end # Loop 3
        end # Loop 4
    end # Loop 5
    C
end

@inline function offsetM(A, i, j)
    inc1A, inc2A = stride(A, 1), stride(A, 2)
    return i*inc1A + j*inc2A
end

@inline function pack_MRxK!(blk::Block{T,MC,KC,NC,MR,NR}, A, k::Int, offsetA::Int, offsetAc::Int) where {T,MC,KC,NC,MR,NR}
    inc1A, inc2A = stride(A, 1), stride(A, 2)
    for j in 1:k
        for i in 1:MR
            blk.Ac[offsetAc+i] = A[offsetA + (i-1)*inc1A + 1]
        end
        offsetAc += MR
        offsetA  += inc2A
    end
    return nothing
end

function pack_A!(blk::Block{T,MC,KC,NC,MR,NR}, A, mc::Int, kc::Int, offsetA::Int) where {T,MC,KC,NC,MR,NR}
    mp, _mr = divrem(mc, MR)
    inc1A, inc2A = stride(A, 1), stride(A, 2)
    offsetAc = 0
    for i in 1:mp
        pack_MRxK!(blk, A, kc, offsetA, offsetAc)
        offsetAc += kc*MR
        offsetA  += MR*inc1A
    end
    if _mr > 0
        for j in 1:kc
            for i in 1:_mr
                blk.Ac[offsetAc+i] = A[offsetA + (i-1)*inc1A + 1]
            end
            for i in _mr+1:MR
                blk.Ac[offsetAc+i] = zero(eltype(A))
            end
            offsetAc += MR
            offsetA  += inc2A
        end
    end
    return nothing
end

@inline function pack_KxNR!(blk::Block{T,MC,KC,NC,MR,NR}, B, k::Int, offsetB::Int, offsetBc::Int) where {T,MC,KC,NC,MR,NR}
    inc1B, inc2B = stride(B, 1), stride(B, 2)
    for i = 1:k
        for j = 1:NR
            blk.Bc[offsetBc+j] = B[offsetB + (j-1)*inc2B + 1]
        end
        offsetBc += NR
        offsetB  += inc1B
    end
    return nothing
end

function pack_B!(blk::Block{T,MC,KC,NC,MR,NR}, B, kc::Int, nc::Int, offsetB::Int) where {T,MC,KC,NC,MR,NR}
    np, _nr = divrem(nc, NR)
    inc1B, inc2B = stride(B, 1), stride(B, 2)
    offsetBc = 0
    for j in 1:np
        pack_KxNR!(blk, B, kc, offsetB, offsetBc)
        offsetBc += kc*NR
        offsetB  += NR*inc2B
    end
    if _nr > 0
        for i in 1:kc
            for j in 1:_nr
                blk.Bc[offsetBc+j] = B[offsetB + (j-1)*inc2B + 1]
            end
            for j in _nr+1:NR
                blk.Bc[offsetBc+j] = zero(eltype(B))
            end
            offsetBc += NR
            offsetB  += inc1B
        end
    end
    return nothing
end

function macro_ker!(blk::Block{T,MC,KC,NC,MR,NR}, C, mc::Int, nc::Int, kc::Int, offsetC::Int) where {T,MC,KC,NC,MR,NR}
    mp, _mr = cld(mc, MR), mc % MR
    np, _nr = cld(nc, NR), nc % NR
    inc1C, inc2C = stride(C, 1), stride(C, 2)
    for j in 1:np
        nr = (j!=np || _nr==0) ? NR : _nr
        for i in 1:mp
            mr = (i!=mp || _mr==0) ? MR : _mr
            offsetA = (i-1)*kc*MR
            offsetB = (j-1)*kc*NR
            if mr == MR && nr==NR
                micro_ker!(C, blk, kc, offsetA, offsetB, offsetC+(i-1)*MR*inc1C + (j-1)*NR*inc2C, inc1C, inc2C, true)
            else
                micro_ker!(blk.Cc, blk, kc, offsetA, offsetB, 0, 1, MR, false)
                _axpy!(C, 1, blk.AB, mr, nr, offsetC+(i-1)*MR*inc1C + (j-1)*NR*inc2C, 0, 1, MR)
            end
        end
    end
    return nothing
end

@inline function micro_ker!(C, blk::Block{T,MC,KC,NC,MR,NR}, kc::Int, offsetA, offsetB, offsetC, inc1C, inc2C, loadC) where {T,MC,KC,NC,MR,NR}
    fill!(blk.AB, zero(eltype(blk.AB)))
    for k in 1:kc
        for j in 1:NR
            for i in 1:MR
                blk.AB[i + (j-1)*MR] += blk.Ac[offsetA+i] * blk.Bc[offsetB+j]
            end
        end
        offsetA += MR
        offsetB += NR
    end
    #for j in 1:NR
    #    for i in 1:MR
    #        C[offsetC+(i-1)*inc1C+(j-1)*inc2C+1] = zero(eltype(C))
    #    end
    #end
    if loadC
        for j in 1:NR
            for i in 1:MR
                C[offsetC+(i-1)*inc1C+(j-1)*inc2C+1] += blk.AB[i + (j-1)*MR]
            end
        end
    end
    return nothing
end

function _axpy!(Y, α, X, m::Int, n::Int, offsetY::Int, offsetX::Int, inc1X, inc2X)
    inc1Y, inc2Y = stride(Y, 1), stride(Y, 2)
    for j in 1:n
        for i in 1:m
            Y[offsetY+(i-1)*inc1Y+(j-1)*inc2Y+1] += α*X[offsetX+(i-1)*inc1X+(j-1)*inc2X+1]
        end
    end
    return nothing
end

export mymul!

end # module
