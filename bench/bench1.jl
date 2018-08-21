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
