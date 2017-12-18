module JuliaBLAS

using SIMD#, CpuId

export mul!

# platform info
const prefetchshift = 512
#const alignment = sizeof(Float64)

#const main_nr       = 6
#const main_mr       = 4
#const c1, c2, c3    = cachesize()
#const line          = cachelinesize()

# Wrap ‘llvm.prefetch‘ Intrinsic
# https://llvm.org/docs/LangRef.html#llvm-prefetch-intrinsic

# address is the address to be prefetched, rw is the specifier determining if the fetch
# should be for a read (0) or write (1), and locality is a temporal locality specifier
# ranging from (0) - no locality, to (3) - extremely local keep in cache. The cache type
# specifies whether the prefetch is performed on the data (1) or instruction (0) cache.
# The rw, locality and cache type arguments must be constant integers.
@inline function _prefetch(address::Ptr{T}, rw::Integer, locality::Integer, cachetype::Integer) where T
    Base.llvmcall((""" declare void @llvm.prefetch(i8*, i32, i32, i32) """,
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.prefetch(i8* %ptr, i32 %1, i32 %2, i32 %3)
                   ret void
                   """),
                  Void, Tuple{UInt64, Int32, Int32, Int32},
                  UInt64(address), Int32(rw), Int32(locality), Int32(cachetype))
end

@inline _prefetch_r(ptr::Ptr{T}) where T = _prefetch(ptr, 0, 3, 1)
@inline _prefetch_w(ptr::Ptr{T}) where T = _prefetch(ptr, 1, 3, 1)

@inline function prefetch_r(::Type{Val{M}}, ::Type{Val{N}}, ::Type{Val{Rem}}, ::Type{Val{Shift}}, ptr::Ptr{T}, ld) where {M,N,Rem,Shift,T}
    for n in 0:N-1
        for m in 0:(M÷64 + (M % 64 >= Rem) - 1)
            _prefetch_r(ptr + M * 64 + Shift + ld * N)
        end
    end
end

check_alignment(x::Integer) = (x & -x) > (x - 1) && x >= sizeof(Ptr{Void})

posix_memalign(pptr::Ref{Ptr{Void}}, alignment::Integer, size::Integer) = ccall(:posix_memalign, Cint, (Ptr{Ptr{Void}}, Csize_t, Csize_t), pptr, alignment, size)

mutable struct BLASVec{T}
    vec::Vector{T}
end

function BLASVec{T}(::Uninitialized, m::Integer) where T
    mem = Ref{Ptr{Void}}()
    alignment = sizeof(Float64)
    @assert check_alignment(alignment)
    # TODO error handling
    return_code = posix_memalign(mem, alignment, m*sizeof(T))
    ptr = Ptr{T}(mem[])
    x = BLASVec(unsafe_wrap(Vector{T}, ptr, m, false))
    finalizer(x->Base.Libc.free(pointer(x.vec)), x)
end

include("kernel.jl")
include("blocking.jl")

end
