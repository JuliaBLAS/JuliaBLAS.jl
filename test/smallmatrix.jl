using JuliaBLAS

@testset "Small Matrix Multiplication" begin
    for i in 4:4:52
        A = rand(i, i)
        B = rand(i, i)
        C = similar(A)
        A_mul_B!(Val{i}, C, A, B)
        @test C ≈ A*B
    end
end

@test_throws AssertionError A_mul_B!(Val{5}, rand(5,5), rand(5,5), rand(5,5))
