using SIMD
import Base:A_mul_B!

@generated function A_mul_B!(::Type{Val{S}}, C::StridedMatrix{T}, A::StridedMatrix{T}, B::StridedMatrix{T}) where {T <: Float64, S}
    VL = 4
    VT = Vec{VL, Float64}
    N, r  = divrem(S, VL)
    @assert r == 0 "Only works for 4N × 4N square matrix!"
    outerex = Expr( :macrocall, Symbol("@inbounds"), :LineNumberNode, Expr(:block) )
    ex = outerex.args[3]
    for i in 0:N-1
        push!( ex.args, Expr(:(=), Symbol(:A, i), :(vload($VT, pointer(A, 1+$VL*$i)))) )
    end

    push!( ex.args, Expr(:for, :(i = 1:S), Expr(:block)) )
    mainloop = ex.args[end].args[2].args
    push!( mainloop, :(Bi = B[1, i]) )
    for i in 0:N-1
        push!( mainloop, Expr(:(=), Symbol(:Mi, i), Expr(:call, :*, :Bi, Symbol(:A, i)) ) )
    end

    push!( mainloop, Expr(:for, :(u = 2:S), Expr(:block)) )
    innerloop = mainloop[end].args[2].args
    push!( innerloop, :(Bu = B[u, i]), :(offset = (u-1)*S) )
    for i in 0:N-1
        push!( innerloop, Expr( :(=),  Symbol(:Au, i), :(vload($VT, pointer(A, 1+$VL*$i + offset))) ) )
        push!( innerloop, Expr( :(+=), Symbol(:Mi, i), Expr(:call, :*, :Bu, Symbol(:Au, i)) ) )
    end

    push!( mainloop, :(offset = (i-1)*S) )
    for i in 0:N-1
        push!( mainloop, Expr(:call, :vstore, Symbol(:Mi, i), :(pointer(C, 1 + $VL * $i + offset)) ) )
    end
    outerex
end

