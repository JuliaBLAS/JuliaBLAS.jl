using BenchmarkTools, LinearAlgebra, Plots
gr()
import JuliaBLAS: addmul!, Block

BLAS.set_num_threads(1)
mnks = 48*(1:2:30)
obtimes = zeros(length(mnks)); jbtimes = zeros(length(mnks))
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.8

for (i,siz) in enumerate(mnks)
    A,B,C = (zeros(siz,siz) for i in 1:3)
    blk = Block(A,B,C,false)
    jbtimes[i] = @belapsed addmul!($C,$A,$B,$blk)
    obtimes[i] = @belapsed mul!($C,$A,$B)
end

time2gflops(mnk, time) = 2 * mnk^3 / time / 10^9

jflops = time2gflops.(mnks, jbtimes)
oflops = time2gflops.(mnks, obtimes)
plot(mnks, jflops, lab="JuliaBLAS")
plot!(mnks, oflops, lab="OpenBLAS", ylabel="GFLOPS", xlabel="M=N=K", legend=:bottomright, dpi=400, ylims=(0,40), yticks=0:5:40)
savefig("bench1.png")
