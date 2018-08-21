# JuliaBLAS
[![Build Status](https://travis-ci.org/YingboMa/JuliaBLAS.jl.svg?branch=master)](https://travis-ci.org/YingboMa/JuliaBLAS.jl)
[![codecov](https://codecov.io/gh/YingboMa/JuliaBLAS.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YingboMa/JuliaBLAS.jl)

# Acknowledgement
Based on [ulmBLAS](http://apfel.mathematik.uni-ulm.de/~lehn/ulmBLAS/),
[BLISlab](https://github.com/flame/blislab/) and some initial code by Andreas
Noack.

```julia
using BenchmarkTools, JuliaBLAS, Printf, LinearAlgebra, Plots

BLAS.set_num_threads(1)
N = 25
obtimes = zeros(N); jbtimes = zeros(N)
xs = (1:N) .* 120
for (i,siz) in enumerate(xs)
    A,B,C = (zeros(siz,siz) for i in 1:3)
    blk = Block(A,B,C)
    jbtimes[i] = @belapsed mymul!($C,$A,$B,$blk)
    obtimes[i] = @belapsed mul!($C,$A,$B)
end

for i in 1:N
    jflops = 2*(100i)^3/jbtimes[i]
    oflops = 2*(100i)^3/obtimes[i]
    slowdown = (oflops-jflops)/oflops * 100
    @printf("M = K = N = %4d, JuliaBLAS vs OpenBLAS: slowdown %3.3lf%%\n", 100i, slowdown)
end

plot(xs, 2 .*xs.^3 ./ jbtimes, lab="JuliaBLAS")
plot!(xs, 2 .*xs.^3 ./ obtimes, lab="OpenBLAS", ylabel="FLOPS", xlabel="M=N=K")
```

```
M = K = N =  100, JuliaBLAS vs OpenBLAS: slowdown 27.024%
M = K = N =  200, JuliaBLAS vs OpenBLAS: slowdown 32.761%
M = K = N =  300, JuliaBLAS vs OpenBLAS: slowdown 34.067%
M = K = N =  400, JuliaBLAS vs OpenBLAS: slowdown 38.329%
M = K = N =  500, JuliaBLAS vs OpenBLAS: slowdown 38.237%
M = K = N =  600, JuliaBLAS vs OpenBLAS: slowdown 39.130%
M = K = N =  700, JuliaBLAS vs OpenBLAS: slowdown 36.771%
M = K = N =  800, JuliaBLAS vs OpenBLAS: slowdown 52.681%
M = K = N =  900, JuliaBLAS vs OpenBLAS: slowdown 37.367%
M = K = N = 1000, JuliaBLAS vs OpenBLAS: slowdown 42.804%
M = K = N = 1100, JuliaBLAS vs OpenBLAS: slowdown 39.971%
M = K = N = 1200, JuliaBLAS vs OpenBLAS: slowdown 39.246%
M = K = N = 1300, JuliaBLAS vs OpenBLAS: slowdown 37.637%
M = K = N = 1400, JuliaBLAS vs OpenBLAS: slowdown 39.914%
M = K = N = 1500, JuliaBLAS vs OpenBLAS: slowdown 40.089%
M = K = N = 1600, JuliaBLAS vs OpenBLAS: slowdown 37.827%
M = K = N = 1700, JuliaBLAS vs OpenBLAS: slowdown 42.383%
M = K = N = 1800, JuliaBLAS vs OpenBLAS: slowdown 53.632%
M = K = N = 1900, JuliaBLAS vs OpenBLAS: slowdown 40.037%
M = K = N = 2000, JuliaBLAS vs OpenBLAS: slowdown 38.386%
M = K = N = 2100, JuliaBLAS vs OpenBLAS: slowdown 42.812%
M = K = N = 2200, JuliaBLAS vs OpenBLAS: slowdown 38.957%
M = K = N = 2300, JuliaBLAS vs OpenBLAS: slowdown 46.964%
M = K = N = 2400, JuliaBLAS vs OpenBLAS: slowdown 50.340%
M = K = N = 2500, JuliaBLAS vs OpenBLAS: slowdown 38.731%
```
