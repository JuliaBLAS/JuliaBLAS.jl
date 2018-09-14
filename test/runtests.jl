using JuliaBLAS, Test, LinearAlgebra

@testset "Matrix Multiplication Tests" begin
    siz = (3, 5, 9, 13, 32, 200)
    for m in siz, n in siz, k in siz
        C = rand(m, n)
        A = rand(m, k)
        B = rand(k, n)
        ABC = A*B + C
        mymuladd!(C, A, B, false)
        @test C ≈ ABC
        ABC = A*B + C
        mymuladd!(C, A, B, true)
        @test C ≈ ABC
    end
end
