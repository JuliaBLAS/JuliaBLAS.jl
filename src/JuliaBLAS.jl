module JuliaBLAS

using SIMD

include("auxiliary.jl")
include("kernel.jl")
include("smallmatrix.jl")
#include("blocking.jl")
#include("copy.jl")

end # module
