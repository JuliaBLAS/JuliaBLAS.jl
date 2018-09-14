using BenchmarkTools, JuliaBLAS, Printf, LinearAlgebra, Plots

BLAS.set_num_threads(1)
N = 10
obtimes = zeros(N); jbtimes = zeros(N)
xs = (1:N) .* 120
for (i,siz) in enumerate(xs)
    A,B,C = (zeros(siz,siz) for i in 1:3)
    blk = Block(A,B,C)
    jbtimes[i] = @belapsed addmul!($C,$A,$B,$blk)
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
