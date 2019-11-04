using BenchmarkTools, LinearAlgebra, Plots
gr()
import JuliaBLAS: addmul!, Block

function bench(nn=30)
    time2gflops(mnk, time) = 2 * mnk^3 / time

    BLAS.set_num_threads(1)
    mnks = 48*(1:2:nn)
    obtimes = zeros(length(mnks)); jbtimes = zeros(length(mnks))

    for (i,siz) in enumerate(mnks)
        A,B,C = (zeros(siz,siz) for i in 1:3)
        bj = @benchmarkable addmul!($C,$A,$B) samples=20 time_tolerance=0.1 seconds=0.8
        jbtimes[i] = run(bj) |> minimum |> time
        bo = @benchmarkable mul!($C,$A,$B) samples=20 time_tolerance=0.1 seconds=0.8
        obtimes[i] = run(bo) |> minimum |> time
    end


    jflops = time2gflops.(mnks, jbtimes)
    oflops = time2gflops.(mnks, obtimes)
    plot(mnks, jflops, lab="JuliaBLAS")
    plot!(mnks, oflops, lab="OpenBLAS", ylabel="GFLOPS", xlabel="M=N=K", legend=:bottomright, dpi=400, ylims=(0,60), yticks=0:5:50)
end
plt = bench()
savefig(plt, "bench1.png")
