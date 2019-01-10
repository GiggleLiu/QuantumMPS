push!(LOAD_PATH, abspath("src"))
using Yao
using QMPS
using DelimitedFiles

function run_train(chem, model; VER, niter=500)
    qo = QMPSOptimizer(chem, model, Adam(lr=0.1))
    gs = ground_state(model)

    history = Float64[]
    fidelities = Float64[]
    nbit = nbit_simulated(chem)
    V = chem.nbit_virtual
    for (k, p) in enumerate(qo)
        curr_loss = energy(chem, model)/nbit
        fid = fidelity_exact(chem, gs)[]
        push!(history, curr_loss)
        push!(fidelities, fid)
        println("step = $k, energy/site = $curr_loss, fidelity = $(fid)")
        flush(stdout)
        k >= niter && break
    end

    for (token, var) in [("loss", history),
                         ("params", parameters(chem.circuit)),
                         ("fidelity", fidelities),
                        ]
        writedlm("data/_chem_$(VER)_$(token)_N$(nbit)_V$V.dat", var)
    end
end

const nbit = 16
const VER = :su2
const USE_CUDA = true
USE_CUDA && include("CuChem.jl")

# load predefined model
V = 7
pairs = pair_ring(V)
chem = model(Val(VER), ComplexF32; nbit=nbit, V=V, B=4096, pairs=pairs)
USE_CUDA && (chem = chem |> cu)
println("Number of parameters is ", chem.circuit |> nparameters)
flush(stdout)

#chem = model(Val(VER), ComplexF32; nbit=nbit, V=4, B=4096)
run_train(chem, Heisenberg(4, 4; periodic=false); VER=VER)
