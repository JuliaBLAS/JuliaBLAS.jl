using SIMD

# Only real cases for now
function pack_b_nano!(::Type{Val{N}}, ::Type{Val{Conj}}, elemstride::Int, length::Int, stride::Int, from::StridedMatrix{T}, to::BLASVec{T}) where {N, Conj, T}
    #SIZEF = sizeof(F)
    #SIZET = sizeof(T)
    #s = P * N
    #isComplex = P == 2
    #pf = T==F ? from : reinterpret(Ptr{T}, from)
    pf = pointer(from)
    pt = pointer(to.vec)
    # special case
    if elemStride==1 && !Conj
        @simd for i in 1:length
            Base.unsafe_copyto!(pt, pf, N)
            to += s*SIZET
            pf += stride*SIZEF
        end
        return nothing;
    end

    pt = isComplex ? reinterpret(Ptr{real(T)}, to) : to
    for j in 1:length
        @simd for i in 1:N
            fi = unsafe_load(pf, (i-1)*elemStride+1)
            if isComplex
                index = 2(i-1) + 1
                unsafe_store!(pt, fi.re, index)
                Conj ? unsafe_store!(pt, -fi.im, index+1) : unsafe_store!(pt, fi.im, index+1)
            else
                unsafe_store!(pt, fi, i)
            end
        end
        pf += stride*SIZEF
        pt += N*SIZET
    end
    return nothing
end

@inline function pack_a!(Type{Val{N}}, A::StridedMatrix{T}, stridemat::StridedMatrix{C}, kernel) where {N,T,C}
    ex = Expr(:block)
    for i in 0:M*N-1
        push!( ex.args, :(kernel(Val{N}, Val{P}, Val{Conj}, length(A), elemstride, stride, stridemat, A)) )
    end
    ex
end

function pack_a_nano!(::Type{Val{N}}, ::Type{Val{P}}, ::Type{Val{Conj}}, ::Type{Val{Elemstride}}, length::Int, stride::Int, from::Ptr{F}, to::Ptr{T}) where {N, P, Conj, Elemstride, F, T}
    SIZEF = sizeof(F)
    SIZET = sizeof(real(T))
    s = N*P
    isComplex = P == 2

    if elemStride==1 && N > 1
        pf = reinterpret(Ptr{real(T)}, from)
        pt = reinterpret(Ptr{real(T)}, to)
        for i in 1:length
            if Conj && isComplex
                unsafe_copy!(pt, pf, N)
                for i in N+1:4:2N
                    vf = -vload(Vec{4,real(T)}, pf+SIZET*(i-1))
                    vstore(vf, pt+SIZET*(i-1))
                end
            else
                unsafe_copy!(pt, pf, s)
            end
            pt += s*SIZET
            pf += stride*SIZEF
        end
    else
        !isComplex && return pack_b_nano(Val{N},Val{1},Val{false},Val{elemStride},length,stride,from,to)
        pt = reinterpret(Ptr{real(T)}, to)
        for j in 1:length
            @simd for i in 1:N
                fi = unsafe_load(from, (i-1)*elemStride+1)
                unsafe_store!(pt, fi.re, i)
                Conj ? unsafe_store!(pt, -fi.im, N+i) : unsafe_store!(pt, fi.im, N+i)
            end
            from += stride*SIZEF
            pt += s*SIZET
        end
    end
    return nothing
end

#function pack_b_nano!(::Type{Val{N}}, ::Type{Val{P}}, ::Type{Val{Conj}}, ::Type{Val{Elemstride}}, length::Int, stride::Int, from::Ptr{F}, to::Ptr{T}) where {N, P, Conj, Elemstride, F, T}
#    SIZEF = sizeof(F)
#    SIZET = sizeof(T)
#    s = P * N
#    isComplex = P == 2
#    pf = T==F ? from : reinterpret(Ptr{T}, from)
#    # special case
#    if elemStride==1 && !Conj && s > 1
#        @simd for i in 1:length
#            unsafe_copy!(to, pf, s)
#            to += s*SIZET
#            pf += stride*SIZEF
#        end
#        return nothing;
#    end
#
#    pt = isComplex ? reinterpret(Ptr{real(T)}, to) : to
#    for j in 1:length
#        @simd for i in 1:N
#            fi = unsafe_load(pf, (i-1)*elemStride+1)
#            if isComplex
#                index = 2(i-1) + 1
#                unsafe_store!(pt, fi.re, index)
#                Conj ? unsafe_store!(pt, -fi.im, index+1) : unsafe_store!(pt, fi.im, index+1)
#            else
#                unsafe_store!(pt, fi, i)
#            end
#        end
#        pf += stride*SIZEF
#        pt += N*SIZET
#    end
#    return nothing
#end
#
#@inline function pack_a!(Type{Val{N}}, A::StridedMatrix{T}, stridemat::StridedMatrix{C}, kernel) where {N,T,C}
#    ex = Expr(:block)
#    for i in 0:M*N-1
#        push!( ex.args, :(kernel(Val{N}, Val{P}, Val{Conj}, length(A), elemstride, stride, stridemat, A)) )
#    end
#    ex
#end
#
#function pack_a_nano!(::Type{Val{N}}, ::Type{Val{P}}, ::Type{Val{Conj}}, ::Type{Val{Elemstride}}, length::Int, stride::Int, from::Ptr{F}, to::Ptr{T}) where {N, P, Conj, Elemstride, F, T}
#    SIZEF = sizeof(F)
#    SIZET = sizeof(real(T))
#    s = N*P
#    isComplex = P == 2
#
#    if elemStride==1 && N > 1
#        pf = reinterpret(Ptr{real(T)}, from)
#        pt = reinterpret(Ptr{real(T)}, to)
#        for i in 1:length
#            if Conj && isComplex
#                unsafe_copy!(pt, pf, N)
#                for i in N+1:4:2N
#                    vf = -vload(Vec{4,real(T)}, pf+SIZET*(i-1))
#                    vstore(vf, pt+SIZET*(i-1))
#                end
#            else
#                unsafe_copy!(pt, pf, s)
#            end
#            pt += s*SIZET
#            pf += stride*SIZEF
#        end
#    else
#        !isComplex && return pack_b_nano(Val{N},Val{1},Val{false},Val{elemStride},length,stride,from,to)
#        pt = reinterpret(Ptr{real(T)}, to)
#        for j in 1:length
#            @simd for i in 1:N
#                fi = unsafe_load(from, (i-1)*elemStride+1)
#                unsafe_store!(pt, fi.re, i)
#                Conj ? unsafe_store!(pt, -fi.im, N+i) : unsafe_store!(pt, fi.im, N+i)
#            end
#            from += stride*SIZEF
#            pt += s*SIZET
#        end
#    end
#    return nothing
#end
