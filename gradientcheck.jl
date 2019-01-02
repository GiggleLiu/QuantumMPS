include("../load.jl")
using Yao, CuYao, Kernels, CircuitBuild, DataSets, Optimizers, Yao.Blocks
using Chemistry, DelimitedFiles, Statistics, Random

function gradstat(;V, Nmax, Nmin, R=10)
    means = zeros(Float64, R, Nmax-Nmin+1)
    Random.seed!(2)
    for nbit = Nmin:Nmax
        ground_state = heisenberg_ground_state(nbit)
        chem = chem_sample(:random, nbit, nbatch=1, nbit_virtual=V)
        nparam = chem.circuit |> nparameters
        println("Number of parameters = ", nparam)
        println("Number of site = ", nbit)

        for k in 1:R
            dispatch!(chem.circuit, :random)
            dbs = collect(chem.circuit, AbstractDiff).blocks[randperm(nparam)[1:100]]
            @time grad = gradients_exact(chem, dbs=dbs)
            #@time grad = gradient.(Ref(chem), dbs)
            mg = grad .|> abs2 |> mean
            println("Mean of gradients = ", mg)
            #println("rtol of gradient noises = ", sum(abs.(grad_noisy-grad))/sum(abs.(grad)))
            means[k, nbit-Nmin+1] = mg

            flush(stdout)
        end
    end

    for (token, var) in [("meangrads3_sq", means),
                        ]
        writedlm("data/chem_$(token)_V$V.dat", var)
    end
end

function gradvarstat(;V, Nmax, Nmin, R=200)
    vars = zeros(Float64, Nmax-Nmin+1)
    Random.seed!(2)
    for nbit = Nmin:Nmax
        ground_state = heisenberg_ground_state(nbit)
        chem = chem_sample(:random, nbit, nbatch=4096, nbit_virtual=V) |> cu
        nparam = chem.circuit |> nparameters
        println("Number of parameters = ", nparam)
        println("Number of site = ", nbit)

        db = collect(chem.circuit, AbstractDiff).blocks[randperm(nparam)[1]]
        @time grad = [gradient(chem, db) for i=1:R]
        varr = grad |> var
        println("Var of gradients = ", varr)
        vars[nbit-Nmin+1] = varr

        flush(stdout)
    end

    for (token, vi) in [("vargrad", vars),
                        ]
        #writedlm("data/chem_$(token)_V$V.dat", vi)
    end
end

gradstat(;V=4, Nmax=20, Nmin=19, R=10)
#gradvarstat(;V=4, Nmax=5, Nmin=5, R=200)
