using SIMD

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
