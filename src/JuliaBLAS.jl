module JuliaBLAS

using SIMD

include("smallmatrix.jl")
include("auxiliary.jl")
include("blocking.jl")
include("copy.jl")

end # module
