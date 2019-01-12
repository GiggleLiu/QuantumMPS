push!(LOAD_PATH, abspath("src"))
using Yao
using QMPS
using DelimitedFiles, JLD2, FileIO
using Random
Random.seed!(8)

function save_training(filename, qopt::Adam, loss, params, fidelity)
    save(filename, "qopt", qopt, "loss", loss, "params", params, "fidelity", fidelity)
end

function load_training(filename)
    res = load(filename)
    res["qopt"], res["loss"], res["params"], res["fidelity"]
end

function run_train(chem, model; VER, niter=500, start_point=0, save_step=10)
    nbit = nbit_simulated(chem)
    V = chem.nbit_virtual
    filename(k::Int) = "data/_chem_$(VER)_N$(nbit)_V$(V)_S$(k).jld2"
    if start_point==0
        optimizer = Adam(lr=0.1)
        history = Float64[]
        fidelities = Float64[]
    else
        optimizer, history, _params, fidelities = load_training(filename(start_point))
        dispatch!(chem.circuit, _params)
    end
    qo = QMPSOptimizer(chem, model, optimizer)
    gs = ground_state(model)

    for (k_, p) in enumerate(qo)
        k = k_ + start_point
        curr_loss = energy(chem, model)/nbit
        fid = fidelity_exact(chem, gs)[]
        push!(history, curr_loss)
        push!(fidelities, fid)
        println("step = $k, energy/site = $curr_loss, fidelity = $(fid)")
        flush(stdout)
        if k%save_step == 0 || k == niter
            save_training(filename(k), qo.optimizer, history, parameters(chem.circuit), fidelities)
            k >= niter && break
        end
    end
end

const nbit = 16
const VER = :random
const USE_CUDA = true
USE_CUDA && include("CuChem.jl")
USE_CUDA && device!(CuDevice(0))

# load predefined model
V = 4
pairs = pair_ring(VER==:su2 ? V : V+1)
chem = model(Val(VER), ComplexF32; nbit=nbit, V=V, B=4096, pairs=pairs)
USE_CUDA && (chem = chem |> cu)
println("Number of parameters is ", chem.circuit |> nparameters)
flush(stdout)

#chem = model(Val(VER), ComplexF32; nbit=nbit, V=4, B=4096)
run_train(chem, Heisenberg(4, 4; periodic=false); VER=VER, niter=500, start_point=0)
