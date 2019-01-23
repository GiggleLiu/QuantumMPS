push!(LOAD_PATH, abspath("src"))
using QMPS
using Yao, Yao.Blocks, CuArrays, CuYao
using DelimitedFiles, Statistics, Random
Random.seed!(5)

function gradstat(hami; VER, Vmin, Vmax, R=10)
    means = zeros(Float64, R, Vmax-Vmin+1)
    for V = Vmin:Vmax
        chem = model(Val(VER), ComplexF32; nbit=20, V=V, B=4096, pairs=pair_ring(V+1), nlayer=5) |> cu
        nparam = chem.circuit |> nparameters
        println("Number of parameters = ", nparam)
        println("Number of vbits = ", V)
        @time state_exact(chem)
        @time state_exact(chem)
        psi = state_exact(chem)
        @show typeof(psi)
        @time expect(hami, psi)
        @time expect(hami, psi)

        for k in 1:R
            dispatch!(chem.circuit, :random)
            dbs = collect(chem.circuit, AbstractDiff).blocks[randperm(nparam)[1:min(100, nparam)]]
            @time grad = gradients_exact(chem, hami; dbs=dbs)
            #@time grad = gradient.(Ref(chem), dbs)
            mg = grad .|> abs2 |> mean
            println("Mean of gradients = ", mg)
            #println("rtol of gradient noises = ", sum(abs.(grad_noisy-grad))/sum(abs.(grad)))
            means[k, V-Vmin+1] = mg

            flush(stdout)
        end
    end

    for (token, var) in [("vargrad", means),
                        ]
        writedlm("data/chem_$(VER)_d5_$(token)_V$Vmax.dat", var)
    end
end

const USE_CUDA = true
USE_CUDA && include("CuChem.jl")
USE_CUDA && device!(CuDevice(0))


function QMPS.state_exact(chem::QuantumMPS{<:GPUReg})
    circuit = chem2circuit(chem)
    nbit = nqubits(circuit)
    psi0 = product_state(ComplexF32, nbit, chem.input_state|>Yao.Intrinsics.packbits) |> cu
    if chem.nbit_ancilla == 0
        return psi0 |> circuit
    else
        return psi0 |> circuit |> focus!((1:nbit-chem.nbit_ancilla)...) |> QMPS.remove_env!
    end
end
const op = Heisenberg(20; periodic=false) |> hamiltonian
gradstat(op; Vmin=1, Vmax=19, VER=:su2)
