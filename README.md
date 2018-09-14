# JuliaBLAS
[![Build Status](https://travis-ci.org/YingboMa/JuliaBLAS.jl.svg?branch=master)](https://travis-ci.org/YingboMa/JuliaBLAS.jl)
[![codecov](https://codecov.io/gh/YingboMa/JuliaBLAS.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/JuliaBLAS.jl)

# It is still WIP!!!!

```julia
using BenchmarkTools, JuliaBLAS, Printf, LinearAlgebra, Plots

BLAS.set_num_threads(1)
N = 10
obtimes = zeros(N); jbtimes = zeros(N)
xs = (1:N) .* 120
for (i,siz) in enumerate(xs)
    A,B,C = (zeros(siz,siz) for i in 1:3)
    blk = Block(A,B,C)
    jbtimes[i] = @belapsed mymul!($C,$A,$B,$blk)
    obtimes[i] = @belapsed mul!($C,$A,$B)
end

for (i,x) in enumerate(xs)
    jflops = 2*x^3/jbtimes[i]
    oflops = 2*x^3/obtimes[i]
    slowdown = (oflops-jflops)/oflops * 100
    @printf("M = K = N = %4d, JuliaBLAS vs OpenBLAS: slowdown %3.3lf%%\n", x, slowdown)
end

plot(xs, 2 .*xs.^3 ./ jbtimes .* 1e-6 , lab="JuliaBLAS")
plot!(xs, 2 .*xs.^3 ./ obtimes .* 1e-6, lab="OpenBLAS", ylabel="GFLOPS", xlabel="M=N=K")
```

```julia
M = K = N =  120, JuliaBLAS vs OpenBLAS: slowdown 28.058%
M = K = N =  240, JuliaBLAS vs OpenBLAS: slowdown 32.375%
M = K = N =  360, JuliaBLAS vs OpenBLAS: slowdown 34.085%
M = K = N =  480, JuliaBLAS vs OpenBLAS: slowdown 38.516%
M = K = N =  600, JuliaBLAS vs OpenBLAS: slowdown 38.761%
M = K = N =  720, JuliaBLAS vs OpenBLAS: slowdown 38.467%
M = K = N =  840, JuliaBLAS vs OpenBLAS: slowdown 36.468%
M = K = N =  960, JuliaBLAS vs OpenBLAS: slowdown 38.749%
M = K = N = 1080, JuliaBLAS vs OpenBLAS: slowdown 38.187%
M = K = N = 1200, JuliaBLAS vs OpenBLAS: slowdown 38.351%

julia> 2 .*xs.^3 ./ jbtimes .* 1e-6
10-element Array{Float64,1}:
 27407.90673698402
 30717.303703341593
 31469.327422042486
 30490.630001007696
 29575.922351154375
 29946.511192693895
 29918.317034093627
 28376.272602551897
 28139.917251739385
 28070.7358501358

julia> 2 .*xs.^3 ./ obtimes .* 1e-6
10-element Array{Float64,1}:
 38097.33781623766
 45422.88230268778
 47742.28009909469
 49591.35738583521
 48295.66066842759
 48667.50080971681
 47091.79911029914
 46327.62356082969
 45524.539831406495
 45533.19069868112
```

# Acknowledgement
Based on [ulmBLAS](http://apfel.mathematik.uni-ulm.de/~lehn/ulmBLAS/),
[BLISlab](https://github.com/flame/blislab/) and some initial code by Andreas
Noack.
