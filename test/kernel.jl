using Test, JuliaBLAS

@testset "Small Matrix Multiplication" begin
    for T in (Float64, Float32)
        for i in 8:8:8*15
            M = 8
            N = 6
            A = rand(T, M, i)
            B = rand(T, i, N)
            C = Matrix{T}(uninitialized, M, N)
            JuliaBLAS._ker!(Val{M}, Val{N}, i, pointer(A), pointer(B), pointer(C))
            @test C ≈ A*B
        end
    end
end
