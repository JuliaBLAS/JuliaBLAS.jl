using SIMD

@inline @generated function _ker!(::Type{Val{MV}}, ::Type{Val{N}}, len, A::Ptr{T}, B::Ptr{T}, C::Ptr{T}) where {MV,N,T}
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
    push!( mainloop, :(prefetch_r( Val{$(sizeof(VT)*M)}, Val{1}, Val{8}, Val{prefetchshift}, A, 0 )) )
    # load Ai...
    for i in 0:M-1
        push!( mainloop, :($(Symbol(:A, i)) = vload($VT, A+$(i*sizeof(VT)))) )
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
