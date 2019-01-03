push!(LOAD_PATH, abspath("src"))
using Yao
using QMPS
using DelimitedFiles

function run_train(chem, ground_state; VER, niter=500)
    qo = QMPSOptimizer(chem, Adam(lr=0.1))

    history = Float64[]
    fidelities = Float64[]
    nbit = nbit_simulated(chem)
    V = chem.nbit_virtual
    for (k, p) in enumerate(qo)
        curr_loss = heisenberg_energy(chem)/nbit
        fid = fidelity_exact(chem, ground_state)[]
        push!(history, curr_loss)
        push!(fidelities, fid)
        println("step = $k, energy/site = $curr_loss, fidelity = $(fid)")
        k >= niter && break
    end

    for (token, var) in [("loss", history),
                         ("params", parameters(chem.circuit)),
                         ("fidelity", fidelities),
                        ]
        writedlm("data/_chem_$(VER)_$(token)_N$(nbit)_V$V.dat", var)
    end
end

const nbit = 20
const VER = :su2
const USE_CUDA = true
USE_CUDA && include("CuChem.jl")

# load predefined model
chem = model(Val(VER), ComplexF32; nbit=nbit, V=5, B=4096)
USE_CUDA && (chem = chem |> cu)
println("Number of parameters is ", chem.circuit |> nparameters)

#chem = model(Val(VER), ComplexF32; nbit=nbit, V=4, B=4096)
@time heisenberg_energy(chem)
run_train(chem, heisenberg_ground_state(nbit); VER=VER)
