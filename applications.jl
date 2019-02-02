push!(LOAD_PATH, abspath("src"))
using Yao
using QMPS
using DelimitedFiles, JLD2, FileIO

# CUDA switch
const USE_CUDA = true
USE_CUDA && device!(CuDevice(0))
USE_CUDA && include("CuChem.jl")

"""Heisenberg model without frustration, with open boundary condition."""
simple_model_heis(size...) = Heisenberg(size...; periodic=false)

"""Heisenberg model with frustration strenth J2 = 0.5, with open boundary condition."""
simple_model_j1j2(size...) = J1J2(size...; periodic=false, J2=0.5)

"""
    simple_ansatz(nbit::Int, symmetry::Symbol, depth::Int; load_params::Bool=false)

Load a predefined MPS inspired ansatz with 4 virtual qubits, batch size 4096.
If `load_params` is `true`, load parameters in training step 500.
"""
function simple_ansatz(nbit::Int, symmetry::Symbol, depth::Int; load_params::Bool=false)
    V = 4     # number of virtual qubits
    batch_size = 4096
    load_step = 500

    ansatz = model(Val(symmetry); nbit=nbit, nlayer=depth, V=V, B=batch_size, pairs=pair_ring(V+1))
    USE_CUDA && (ansatz = ansatz |> cu)

    if load_params
        filename = "data/chem_$(symmetry)_d$(depth)_N$(nbit)_V$(V)_S$(load_step).jld2"
        params = load_training(filename)[3]
        println("loading file $filename")
        println("Number of parameters is ", params |> length)
        dispatch!(ansatz.circuit, params)
    end

    ansatz
end


"""
    save_training(filename, qopt::Adam, loss, params, fidelity)

Save training status.
"""
function save_training(filename, qopt::Adam, loss, params, fidelity)
    save(filename, "qopt", qopt, "loss", loss, "params", params, "fidelity", fidelity)
end

"""
    load_training(filename) -> Tuple

Load training status (qopt, loss, params, fidelity).
"""
function load_training(filename)
    res = load(filename)
    res["qopt"], res["loss"], res["params"], res["fidelity"]
end

"""
    run_train(ansatz, model; SAVE_ID, niter=500, start_point=0, save_step=0)
"""
function run_train(ansatz, model; SAVE_ID, niter=500, start_point=0, save_step=10)
    nbit = nbit_simulated(ansatz)
    V = ansatz.nbit_virtual
    filename(k::Int) = "data/chem_$(SAVE_ID)_N$(nbit)_V$(V)_S$(k).jld2"
    if start_point==0
        optimizer = Adam(lr=0.1)
        history = Float64[]
        fidelities = Float64[]
    else
        optimizer, history, _params, fidelities = load_training(filename(start_point))
        dispatch!(ansatz.circuit, _params)
    end
    qo = QMPSOptimizer(ansatz, model, optimizer)
    gs = ground_state(model)

    for (k_, p) in enumerate(qo)
        k = k_ + start_point
        curr_loss = energy(ansatz, model)/nbit
        fid = fidelity_exact(ansatz, gs)[]
        push!(history, curr_loss)
        push!(fidelities, fid)
        println("step = $k, energy/site = $curr_loss, fidelity = $(fid)")
        flush(stdout)
        if k%save_step == 0 || k == niter
            save_training(filename(k), qo.optimizer, history, parameters(ansatz.circuit), fidelities)
            k >= niter && break
        end
    end
end

"""
    correlation_matrix(ansatz, op; SAVE_ID)

Calculate and save correlation matrix <Z_i Z_j>.
"""
function correlation_matrix(ansatz; SAVE_ID)
    nbit = nbit_simulated(ansatz)
    V = ansatz.nbit_virtual

    om = zeros(Float64, nbit, nbit)
    for i =1:nbit, j=1:nbit
        if i!=j
            om[i,j] = measure_corr(ansatz, i=>Z, j=>Z) |> real
            println("<σz($i)σz($j)> = $(om[i,j])")
        end
    end

    for (token, var) in [
                         ("om", om),
                        ]
        writedlm("data/chem_$(SAVE_ID)_$(token)$(tag)_N$(nbit)_V$V.dat", var)
    end
end
