module JuliaBLAS

include("gemm.jl")
include("kernel.jl")

export mymul!, Block

end
