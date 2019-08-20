module JuliaBLAS

using SIMD
import Base.Cartesian: @nexprs
import Hwloc

include("tune.jl")
include("gemm.jl")

export addmul!

end
