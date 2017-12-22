using SIMD

struct BlockInfo{T}
    mc::Int
    kc::Int
    A::BLASVec{T}
    B::BLASVec{T}
    prefetch::BLASVec{T}
end

@fastmath function blocking(::Type{T}, m::Int, n::Int, k::Int) where T
    P = T<:Complex ? 2 : 1
    SIZE = sizeof(T)
    s = n * P
    l2 = c2 * 3÷5 # half cache
    kc = div((l2 - m*main_nr*SIZE), (m*SIZE + main_nr*SIZE))
    mc = m
    minKc = 320 ÷ P
    a = 2 * (main_nr*main_mr*SIZE + main_nr * line) + 512
    if (kc < minKc || kc * (main_mr*SIZE + main_nr*SIZE) + a  > c1)
        kc = (c1 - a) ÷ (main_mr*SIZE + main_nr*SIZE)
        kc = normalize_chunk_size(Val{main_nr}, kc, k)
        df = SIZE*main_nr + SIZE * kc
        mc = div((l2 - kc * main_nr * SIZE), df)
        mc = normalize_chunk_size(Val{main_nr}, mc, m)
    else
        kc = normalize_chunk_size(Val{main_mr}, kc, k)
    end
    a_length = mc * kc
    b_length = kc * n
    mem = allocate(T, a_length + b_length + prefetchshift)
    A = BLASVec{T}(mem, mc, kc, true)
    B = BLASVec{T}(mem + a_length, kc, n, false)
    prefetch = BLASVec{T}(mem + a_length + b_length, div(prefetchshift,SIZE), 1, false)
    return BlockInfo{T}(mc, kc, A, B, prefetch)
end

@fastmath function normalize_chunk_size(::Type{Val{subChunk}}, chunk::Int, length::Int) where subChunk
    if chunk >= length
        return length
    end
    #c1, c2 = unsafe_divrem(length, chunk)
    c1, c2 = divrem(length, chunk)
    # count = length ÷ chunk + (length % chunk != 0)
    count = c1 + (c2!=0 ? 1 : 0)
    c1, c2 = divrem(length, count)
    # new_ch = length ÷ count + (length % count != 0)
    new_ch = c1 + (c2!=0 ? 1 : 0)
    r = rem(new_ch, subChunk)
    if r != 0
        new_new_ch = new_ch + subChunk - r
        if new_new_ch <= chunk
            return new_new_ch
        end
    end
    return new_ch
end
