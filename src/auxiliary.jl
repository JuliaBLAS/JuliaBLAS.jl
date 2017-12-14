using CpuId

const prefetchShift = 512
const main_nr       = 6
const main_mr       = 4
const c1, c2, c3    = cachesize()
const line          = cachelinesize()

function _prefetch_r(ptr::Ptr{T}) where T
    Base.llvmcall((""" declare void @llvm.prefetch(i8*, i32, i32, i32) """,
                   """
                   %ptr = inttoptr i64 %0 to i8*
                   call void @llvm.prefetch(i8* %ptr, i32 0, i32 3, i32 1)
                   ret void
                   """),
                  Void, Tuple{UInt64}, UInt64(ptr))
end

function prefetch_r(::Type{Val{M}}, ::Type{Val{N}}, ::Type{Val{Rem}}, ::Type{Val{Shift}}, ptr::Ptr{T}, ld) where {M,N,Rem,Shift,T}
    for n in 0:N-1
        for m in 0:(M÷64 + (M % 64 >= Rem) - 1)
            _prefetch_r(ptr + M * 64 + Shift + ld * N)
        end
    end
end

#const platformAlignment = max(sizeof(LinAlg.BlasReal.a), sizeof(LinAlg.BlasReal.b))

#posix_memalign(memptr::Ref{Ptr{Void}}, alignment::T, size::T) where {T<:Integer} = ccall(:posix_memalign, Cint, (Ptr{Ptr{Void}}, Csize_t, Csize_t), memptr, alignment, size)

#is_good_dynamic_alignment(x) = (x & -x) > (x - 1) && x >= sizeof(Ptr{Void})

#function aligned_allocate(bytes, a)
#    @assert is_good_dynamic_alignment(a)
#    result = Ref{Ptr{Void}}()
#    ret_code = posix_memalign(result, a, bytes)
#    if ret_code == Base.Libc.ENOMEM
#        return nothing
#    elseif ret_code == Base.Libc.EINVAL
#        error("Alignment is not power of two")
#    else
#        return result[]
#    end
#end

#allocate(bytes)    = aligned_allocate(bytes, platformAlignment)
#deallocate(p::Ptr) = Base.Libc.free(p)

#struct BLASVec{T<:LinAlg.BlasFloat} <: AbstractArray{T, 1}
#    ptr::Ptr{T}
#    len::LinAlg.BlasInt
#end

#import Base:getindex, isassigned, length, size

#getindex(v::BLASVec{T}, i) where T = unsafe_load(v.ptr, i)
# isassigned(v::BLASVec{T}, i) where T = !(i>v.len) && (i<0)
#length(v::BLASVec{T}) where T = length(v.len)
#size(v::BLASVec{T}) where T = (v.len,)
#function size(v::BLASVec{T}, r) where T
#    if r == 1
#        return v.len
#    elseif r > 1
#        return 1
#    else
#        error("dimension out of range!")
#    end
#end

#function llvm_memset!(dest::AbstractVecOrMat{Float64}, src::UInt8)
#    Base.llvmcall((""" declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) """,
#                   """
#                      %4 = bitcast double* %0 to i8*
#                      tail call void @llvm.memset.p0i8.i64(i8* nocapture %4, i8 %1, i64 %2, i32 8, i1 false)
#                      ret void
#                   """),
#                  Void, Tuple{Ptr{Float64}, UInt8, Int64}, pointer(dest), src, sizeof(dest))
#end
