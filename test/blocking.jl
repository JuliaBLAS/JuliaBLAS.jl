using Test, JuliaBLAS

# detailed explanations https://apfel.mathematik.uni-ulm.de/~lehn/sghpc/day08/page02.html
@testset "Packing Matrix A" begin
    M = 14; K = 15
    MC = 8; KC = 12
    MR = 4
    A = Matrix(reshape(1:Float64(M*K), M, K))
    _A = zeros(MC*KC)
    JuliaBLAS.pack_A(MC, KC, A, 0, 1, M, _A)
    # Test A_{1,1} packing
    @test reshape(_A, MR, MC*KC÷MR) == hcat(A[1:MC÷2,1:KC], A[MC÷2+1:MC,1:KC])
end
