using Test, JuliaBLAS, BenchmarkTools

@testset "BLAS level3 kernel" begin
    for T in (Float64, Float32)
        len = 512
        M = T <: Float32 ? 8*2 : 8
        N = T <: Float32 ? 6*2 : 6
        A = rand(T, M, len)
        B = rand(T, len, N)
        C = Matrix{T}(uninitialized, M, N)
        to = @belapsed A_mul_B!($C, $A, $B)
        println("--- OpenBLAS T=$T Len=$len ---\n t = $to")
        tj = @belapsed JuliaBLAS._ker!($(Val{M}), $(Val{N}), $len, $(pointer(A)), $(pointer(B)), $(pointer(C)))
        println("--- JuliaBLAS T=$T Len=$len ---\n t = $tj")
        @test to/tj > (T<:Float32 ? 2.5 : 4.5)
        @test C ≈ A*B
    end
end

@testset "Small Matrix Multiplication" begin
    for T in (Float64, Float32)
        inc = T <: Float32 ? 8 : 4
        for i in inc:inc:inc*8
            A = rand(T, i, i)
            B = rand(T, i, i)
            C = copy(A)
            JuliaBLAS._mul!(Val{i}, C, A, B, Val{true}, T(2), T(3))
            @test C ≈ 2*A*B + 3*A
            JuliaBLAS._mul!(Val{i}, C, A, B)
            @test C ≈ A*B
        end
        @test_throws AssertionError JuliaBLAS._mul!(Val{5}, rand(T, 5, 5), rand(T, 5, 5), rand(T, 5, 5))
    end
end
