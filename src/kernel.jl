using SIMD

# compute MC×KC * KC×N
function gebp!(C::StridedMatrix{T}, block::BlockInfo{T}) where T
    for j in 0:length(block.B)÷block.kc
        for i in 0:block.mc÷main_mr-1
            #_ker!(Val{main_mr}, Val{main_nr}, main_kc, pointer(_A.vec, 1+i*main_mr), pointer(_B.vec, 1+j*main_kc), pointer(_C.vec, 1+i*main_mr+j*main_nr))
            #_ker!(Val{main_mr}, Val{main_nr}, block.kc, block.A.ptr + (1+i*main_mr)*sizeof(T), block.B.ptr + (1+j*block.kc)*sizeof(T), pointer(C, 1+i*main_mr+j*main_nr))
            ker!(block.kc, block.A.ptr + (i*main_mr)*sizeof(T), block.B.ptr + (j*block.kc)*sizeof(T), pointer(C, 1+i*main_mr+j*main_nr))
        end
    end
end

# working micro-kernel
function ker!(len::Int, A::Ptr{T}, B::Ptr{T}, C::Ptr{T}) where T
    M0 = zero(SIMD.Vec{4,Float64})
    M1 = zero(SIMD.Vec{4,Float64})
    M2 = zero(SIMD.Vec{4,Float64})
    M3 = zero(SIMD.Vec{4,Float64})
    M4 = zero(SIMD.Vec{4,Float64})
    M5 = zero(SIMD.Vec{4,Float64})
    M6 = zero(SIMD.Vec{4,Float64})
    M7 = zero(SIMD.Vec{4,Float64})
    M8 = zero(SIMD.Vec{4,Float64})
    M9 = zero(SIMD.Vec{4,Float64})
    M10 = zero(SIMD.Vec{4,Float64})
    M11 = zero(SIMD.Vec{4,Float64})
    for i = 1:len
        #@assert UInt(A)%16 == 0 "A is $(UInt(A)) at $i"
        A0 = vloada(SIMD.Vec{4,Float64}, A + 0)
        A1 = vloada(SIMD.Vec{4,Float64}, A + 32)
        B0 = (SIMD.Vec{4,Float64})(unsafe_load(B, 0len + 1))
        B1 = (SIMD.Vec{4,Float64})(unsafe_load(B, 1len + 1))
        M0 = muladd(A0, B0, M0)
        M1 = muladd(A1, B0, M1)
        M2 = muladd(A0, B1, M2)
        M3 = muladd(A1, B1, M3)
        B2 = (SIMD.Vec{4,Float64})(unsafe_load(B, 2len + 1))
        B3 = (SIMD.Vec{4,Float64})(unsafe_load(B, 3len + 1))
        M4 = muladd(A0, B2, M4)
        M5 = muladd(A1, B2, M5)
        M6 = muladd(A0, B3, M6)
        M7 = muladd(A1, B3, M7)
        B4 = (SIMD.Vec{4,Float64})(unsafe_load(B, 4len + 1))
        B5 = (SIMD.Vec{4,Float64})(unsafe_load(B, 5len + 1))
        M8 = muladd(A0, B4, M8)
        M9 = muladd(A1, B4, M9)
        M10 = muladd(A0, B5, M10)
        M11 = muladd(A1, B5, M11)
        A += 64
        B += 8
    end
    vstore(M0, C + 0)
    vstore(M1, C + 32)
    vstore(M2, C + 64)
    vstore(M3, C + 96)
    vstore(M4, C + 128)
    vstore(M5, C + 160)
    vstore(M6, C + 192)
    vstore(M7, C + 224)
    vstore(M8, C + 256)
    vstore(M9, C + 288)
    vstore(M10, C + 320)
    vstore(M11, C + 352)
end

@inline @generated function _ker!(::Type{Val{MV}}, ::Type{Val{N}}, len::Int, A::Ptr{T}, B::Ptr{T}, C::Ptr{T}) where {MV,N,T}
    # AVX256
    VL = (256÷8)÷sizeof(T)
    VT = Vec{VL, T}
    M = MV÷VL
    # wrap everything into a @inbounds block
    outerex = Expr( :macrocall, Symbol("@inbounds"), :LineNumberNode, Expr(:block) )
    ex = outerex.args[3]
    # initialize Mi... which will be later stored in the C matrix
    for i in 0:M*N-1
        push!( ex.args, :($(Symbol(:M, i)) = zero($VT)) )
    end

    # main loop
    push!( ex.args, Expr(:for, :(i = 1:len), Expr(:block)) )
    mainloop = ex.args[end].args[2].args
    # the main loop computes many rank-1 updates on Mi...
    # Mi... are very small and able to fit in SIMD registers
    # e.g. YMM0-YMM11, in the case of 8 × N × 6 multiplication
    #push!( mainloop, :(prefetch_r( Val{$(sizeof(VT)*M)}, Val{1}, Val{8}, Val{prefetchshift}, A, 0 )) )
    # load Ai...
    for i in 0:M-1
        push!( mainloop, :($(Symbol(:A, i)) = vloada($VT, A+$(i*sizeof(VT)))) )
    end

    for u in 0:(N÷2+N%2-1)
        um = 2u:(2u+2>N ? 2u : 2u+1)
        # load Bi
        for n in um
            push!( mainloop, :($(Symbol(:B, n)) = $VT(unsafe_load(B, $n*len + 1))) )
        end
        # muladd the outer product
        for n in um
            for m in 0:M-1
                push!( mainloop, :($(Symbol(:M, m + n*M)) = muladd($(Symbol(:A, m)), $(Symbol(:B, n)), $(Symbol(:M, m + n*M)))) )
            end
        end
    end
    push!(mainloop, :(A += $(sizeof(VT)*M)), :(B += $(sizeof(T))))

    # store Mi... into C
    for i in 0:M*N-1
        push!( ex.args, :(vstore($(Symbol(:M, i)), C + $(sizeof(VT) * i))) )
    end
    ex
end

# kernel for small matrix multiplication
# 4N x 4N (Float64) or 8N x 8N (Float32) where N > 2
@generated function _mul!(::Type{Val{S}}, C::StridedMatrix{T}, A::StridedMatrix{T},
                          B::StridedMatrix{T}, ::Type{Val{G}} = Val{false},
                          α::T = zero(T), β::T = zero(T)) where {T <: LinAlg.BlasReal, S, G}
    # AVX256
    VL = (256÷8)÷sizeof(T)
    VT = Vec{VL, T}
    N, r  = divrem(S, VL)
    @assert r == 0 "Only works for $(VL)N × $(VL)N square matrix!"
    outerex = Expr( :macrocall, Symbol("@inbounds"), :LineNumberNode, Expr(:block) )
    ex = outerex.args[3]
    # prefetch Ai0...
    for i in 0:N-1
        push!( ex.args, :($(Symbol(:A, i)) = vload($VT, pointer(A, 1+$VL*$i))) )
    end

    # main loop
    push!( ex.args, Expr(:for, :(i = 1:S), Expr(:block)) )
    mainloop = ex.args[end].args[2].args
    push!( mainloop, :(Bi = B[1, i]) )
    # take advantage of the prefetch & initialize Mi... which will be later stored in the C matrix
    for i in 0:N-1
        push!( mainloop, :($(Symbol(:Mi, i)) = Bi * $(Symbol(:A, i))) )
    end

    push!( mainloop, Expr(:for, :(u = 2:S), Expr(:block)) )
    innerloop = mainloop[end].args[2].args
    # enter the inner loop
    # update Mi... by the rest of Bi column
    push!( innerloop, :(Bu = B[u, i]), :(offset = (u-1)*S) )
    # separate loads from calculation for better cache locality
    for i in 0:N-1
        push!( innerloop, :($(Symbol(:Au, i)) = vload($VT, pointer(A, 1+$VL*$i + offset))) )
    end
    for i in 0:N-1
        push!( innerloop, :($(Symbol(:Mi, i)) += Bu * $(Symbol(:Au, i))) )
    end

    push!( mainloop, :(offset = (i-1)*S) )
    # load Ci column if necessary
    if G
        for i in 0:N-1
            push!( mainloop, :($(Symbol(:Ci, i)) = vload($VT, pointer(C, 1+$VL*$i + offset))))
            push!( mainloop, :($(Symbol(:Mi, i)) = α * $(Symbol(:Mi, i)) + β * $(Symbol(:Ci, i))))
        end
    end
    # store Mi... in the Ci column
    for i in 0:N-1
        push!( mainloop, :(vstore($(Symbol(:Mi, i)), pointer(C, 1 + $VL * $i + offset))) )
    end
    outerex
end
