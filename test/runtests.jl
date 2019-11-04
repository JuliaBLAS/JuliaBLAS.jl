using Test, LinearAlgebra
using JuliaBLAS: Block, addmul!

function matmul_test(name, datagen, generic_only = false)
    @testset "$name" begin
        siz = (3, 5, 9, 13, 32, 200)
        for m in siz, n in siz, k in siz
            C = datagen(m, n)
            A = datagen(m, k)
            B = datagen(k, n)
            @test true # ensure base functionality doesn't error out before here

            if !generic_only
                ABC = A*B + C
                addmul!(C, A, B, generic=Val(false), mr=Val(12), nr=Val(4))
                @test C ≈ ABC
                ABC = A*B + C
                addmul!(C, A, B, generic=Val(false), mr=Val(8), nr=Val(6))
                @test C ≈ ABC
                ABC = A*B + C
                addmul!(C, A, B, generic=Val(false), mr=Val(4), nr=Val(4))
                @test C ≈ ABC
            end

            ABC = A*B + C
            addmul!(C, A, B, generic=Val(true), mr=Val(12), nr=Val(4))
            @test C ≈ ABC
            ABC = A*B + C
            addmul!(C, A, B, generic=Val(true), mr=Val(8), nr=Val(6))
            @test C ≈ ABC
            ABC = A*B + C
            addmul!(C, A, B, generic=Val(true), mr=Val(4), nr=Val(4))
            @test C ≈ ABC
        end
    end
end

matmul_test.([
              Int64,
              Int32,
              UInt64,
              UInt32,
              Float64,
              Float32,
             ],
              randn)

# Weird type, with weird math
#struct Stringish
#    str::String
#end
#
#Stringish(x) = Stringish(string(x))
#
#Base.zero(::Stringish) = Stringish("")
#Base.one(::Stringish) = Stringish("")
#Base.:+(s1::Stringish, s2::Stringish) = Stringish(s1.str * s2.str)
#Base.:*(s1::Stringish, s2::Stringish) = Stringish("[$(s1.str),$(s2.str)]")
#
#Base.rand(::Type{Stringish}, (m, n)::Tuple{Int64, Int64}) = [Stringish(rand(0:9)) for ii in 1:m, jj in 1:n]
#Base.:≈(x::Stringish, y::Stringish) = x.str == y.str
##################################################################
#
#
#datagen(T) = (m,n) -> rand(T, (m,n))
#
#@testset "Matrix Multiplication Tests" begin
#    for T in [
#              #Int128,
#              #Int16,
#              #Int32,
#              Int64,
#              #Int8,
#              #UInt128,
#              #UInt16,
#              #UInt32,
#              UInt64,
#              #UInt8,
#              #Float16,
#              #Float32,
#              Float64,
#              #Bool
#             ]
#
#        matmul_test(string(T), datagen(T))
#        #T = Complex{T}
#        #matmul_test(string(T), datagen(T))
#    end
#
#
#    #for T in [BigFloat,]
#    #    matmul_test(string(T), datagen(T), true)
#    #    T = Complex{T}
#    #    matmul_test(string(T), datagen(T), true)
#    #end
#
#    matmul_test(string(Stringish), datagen(Stringish), true)
#
#    matmul_test(string(BigInt), (m,n) -> rand(-big"2"^256 : big"2"^256, (m,n)), true)
#end
