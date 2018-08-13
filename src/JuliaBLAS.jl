module JuliaBLAS

using SIMD, CpuId

struct Block{abType, cType}
    Ac::abType
    Bc::abType
    Cc::cType
    mc::Int
    kc::Int
    nc::Int
    mr::Int
    nr::Int
end
function Block(A, B, C)
    mr, nr = 8, 8
    mc, kc, nc = 64, 64, 64
    Ac = similar(A, kc*mc)
    Bc = similar(B, kc*nc)
    Cc = reshape(valloc(eltype(C), 8, mr*nr), mr, nr)
    fill!(Ac, 0); fill!(Bc, 0)
    Block(Ac, Bc, Cc,
          mc, kc, nc, mr, nr)
end

function mymul!(C, A, B, blk=Block(A, B, C))
    m,  n = size(A); _n, k = size(B)
    @assert n == _n
    _m, _k = size(C)
    @assert m == _m && k == _k
    mb, _mc = cld(m, blk.mc), m % blk.mc
    nb, _nc = cld(n, blk.nc), n % blk.nc
    kb, _kc = cld(k, blk.kc), k % blk.kc
    for j in 1:nb # Loop 5
        nc = (j!=nb || _nc==0) ? blk.nc : _nc
        for l in 1:kb # Loop 4
            kc = (l!=kb || _kc==0) ? blk.kc : _kc
            #_β = l==1 ? β : 1.0
            offsetB = offsetM(B, blk.kc*(l-1), blk.nc*(j-1))
            pack_B!(blk, B, kc, nc, offsetB)
            for i in 1:mb # Loop 3
                mc = (i!=mb || _mc==0) ? blk.mc : _mc
                offsetA = offsetM(A, blk.mc*(i-1), blk.kc*(l-1))
                offsetC = offsetM(C, blk.mc*(i-1), blk.nc*(j-1))
                pack_A!(blk, A, mc, kc, offsetA)
                macro_ker!(blk, C, mc, nc, kc, offsetC)
            end # Loop 3
        end # Loop 4
    end # Loop 5
    C
end

function offsetM(A, i, j)
    inc1A, inc2A = stride(A, 1), stride(A, 2)
    return i*inc1A + j*inc2A
end

function pack_MRxK!(blk::Block, A, k::Int, offsetA::Int, offsetAc::Int)
    inc1A, inc2A = stride(A, 1), stride(A, 2)
    for j in 1:k
        for i in 1:blk.mr
            blk.Ac[offsetAc+i] = A[offsetA + (i-1)*inc1A + 1]
        end
        offsetAc += blk.mr
        offsetA  += inc2A
    end
    return nothing
end

function pack_A!(blk::Block, A, mc::Int, kc::Int, offsetA::Int)
    mp, _mr = divrem(mc, blk.mr)
    inc1A, inc2A = stride(A, 1), stride(A, 2)
    offsetAc = 0
    for i in 1:mp
        pack_MRxK!(blk, A, kc, offsetA, offsetAc)
        offsetAc += kc*blk.mr
        offsetA  += blk.mr*inc1A
    end
    if _mr > 0
        for j in 1:kc
            for i in 1:_mr
                blk.Ac[offsetAc+i] = A[offsetA + (i-1)*inc1A + 1]
            end
            for i in _mr:blk.mr
                blk.Ac[offsetAc+i] = zero(eltype(A))
            end
        end
        offsetAc += blk.mr
        offsetA  += inc2A
    end
    return nothing
end

function pack_KxNR!(blk::Block, B, k::Int, offsetB::Int, offsetBc::Int)
    inc1B, inc2B = stride(B, 1), stride(B, 2)
    for i = 1:k
        for j = 1:blk.nr
            blk.Bc[offsetBc+j] = B[offsetB + (j-1)*inc2B + 1]
        end
        offsetBc += blk.nr
        offsetB  += inc1B
    end
    return nothing
end

function pack_B!(blk::Block, B, kc::Int, nc::Int, offsetB::Int)
    np, _nr = divrem(nc, blk.nr)
    inc1B, inc2B = stride(B, 1), stride(B, 2)
    offsetBc = 0
    for j in 1:np
        pack_KxNR!(blk, B, kc, offsetB, offsetBc)
        offsetBc += kc*blk.nr
        offsetB  += blk.nr*inc2B
    end
    if _nr > 0
        for i in 1:kc
            for j in 1:_nr
                blk.Bc[offsetBc+j] = B[offsetB + j*inc2B]
            end
            for j in _nr:blk.nr
                blk.Bc[offsetBc+j] = zero(eltype(B))
            end
        end
        offsetBc += blk.nr
        offsetB  += inc1B
    end
    return nothing
end

function macro_ker!(blk, C, mc::Int, nc::Int, kc::Int, offsetC::Int)
    mp, _mr = cld(mc, blk.mr), mc % blk.mr
    np, _nr = cld(nc, blk.nr), nc % blk.nr
    for j in 1:np
        nr = (j!=np || _nr==0) ? blk.nr : _nr
        for i in 1:mp
            mr = (i!=mp || _mr==0) ? blk.mr : _mr
            offsetA = i*kc*blk.mr
            offsetB = j*kc*blk.nr
            offsetC = offsetM(C, (i-1)*blk.mr, (j-1)*blk.mr)
            if mr == blk.mr && nr==blk.nr
                micro_ker!(C, blk, kc, offsetA, offsetB, offsetC)
            else
                micro_ker!(blk.Cc, blk, kc, offsetA, offsetB, 0)
                _axpy!(C, 1, blk.Cc, mr, nr, offsetC, 0)
            end
        end
    end
    return nothing
end

function micro_ker!(C, blk::Block, kc::Int, offsetA, offsetB, offsetC)
    fill!(blk.Cc, zero(eltype(blk.Cc)))
    inc1C, inc2C = stride(C, 1), stride(C, 2)
    for k in 1:kc
        for j in 1:blk.nr
            for i in 1:blk.mr
                blk.Cc[i, j] += blk.Ac[i] * blk.Bc[j]
            end
        end
        offsetA += blk.mr
        offsetB += blk.nr
    end
    if pointer(blk.Cc) != pointer(C)
        for j in 1:blk.mr, i in 1:blk.nr
            C[offsetC+(i-1)*inc1C+(j-1)*inc2C+1] += blk.Cc[i, j]
        end
    end
    return nothing
end

function _axpy!(Y, α, X, m::Int, n::Int, offsetY::Int, offsetX::Int)
    inc1Y, inc2Y = stride(Y, 1), stride(Y, 2)
    inc1X, inc2X = stride(X, 1), stride(X, 2)
    if α != 1.0
        for j = 1:n
            for i = 1:m
                Y[offsetY+(i-1)*inc1Y+(j-1)*inc2Y+1] = Y[(i-1)*inc1Y+(j-1)*inc2Y+1] + α*X[offsetX+(i-1)*inc1X+(j-1)*inc2X+1]
            end
        end
    else
        for j = 1:n
            for i = 1:m
                Y[offsetY+(i-1)*inc1Y+(j-1)*inc2Y+1] = Y[(i-1)*inc1Y+(j-1)*inc2Y+1] + X[offsetX+(i-1)*inc1X+(j-1)*inc2X+1]
            end
        end
    end
    return nothing
end

export mymul!

end # module
