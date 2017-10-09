using JuliaBLAS

@testset "Check allocation" begin
    block = JuliaBLAS.blocking{Float64}(256,256,256)
    JuliaBLAS.deallocate(block)
end

@testset "Copy" begin
    from = rand(Float64,10,5)
    to   = similar(from)
    JuliaBLAS.pack_b_nano{10,1,false,Float64,Float64}(5, 10, 1, pointer(from), pointer(to))
end
