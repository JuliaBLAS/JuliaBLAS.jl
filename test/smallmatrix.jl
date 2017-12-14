using JuliaBLAS, Test

@testset "Small Matrix Multiplication" begin
    for T in (Float64, Float32)
        for i in 8:8:8*15
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
