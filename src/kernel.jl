using SIMD

@inline @generated function micro_ker!(C, blk::Block{T,MC,KC,NC,MR,NR},
                            kc::Int, offsetA::Int, offsetB::Int, offsetC::Int,
                            inc1C::Int, inc2C::Int,
                            ::Val{loadC}) where {T,MC,KC,NC,MR,NR,loadC}
    quote
        @inbounds begin
            VT = Vec{$MR,Float64}
            if $loadC
                @nexprs $NR i -> ab_i = vload(VT, pointer(C, offsetC+(i-1)*inc2C+1))
            else
                @nexprs $NR i -> ab_i = zero(VT)
            end
            for k in 1:kc
                a1 = vload(VT, blk.Ac, offsetA+(k-1)MR+1)
                @nexprs $NR i -> begin
                    b_i = VT(blk.Bc[offsetB+(k-1)NR+i])
                    ab_i = muladd(a1, b_i, ab_i)
                end
            end
            if $loadC
                @nexprs $NR i -> vstore(ab_i, pointer(C, offsetC+(i-1)*inc2C+1)) # TODO
            else
                @nexprs $NR i -> vstore(ab_i, pointer(blk.AB, (i-1)MR+1))
            end
            return nothing
        end
    end
end
