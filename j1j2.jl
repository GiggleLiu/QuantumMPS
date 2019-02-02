include("applications.jl")

using Fire

@main function sample_cluster()
    m = model(Val(:cluster); nbit=3, B=10)
    @show gensample(m, X)
end

"""
    julia j1j2.jl train [--symmetry <su2|u1|general>] [--depth <Int>]

Train a 4 x 4 frustrated Heisenberg model with J2 = 0.5.
Available ansatz symmetries includes `general`, `u1` and `su2`.
"""
@main function train(;symmetry::Symbol=:su2, depth=5)
    USE_CUDA || @warn "You are not using GPU (35 x speed up), this training may take life long. Turn on the switch in file `applications.jl` if you have a GPU that supports CUDA!"
    model = simple_model_j1j2(4, 4)
    ansatz = simple_ansatz(16, symmetry, depth; load_params=false)

    run_train(ansatz, model; SAVE_ID=Symbol(symmetry,:_d,depth), niter=500, start_point=0)
end

"""
    julia j1j2.jl measure <energy|fidelity|szsz> [--symmetry <su2|u1|general>] [--depth <Int>]

Load a pre-trained ansatz for 4 x 4 frustrated Heisenberg model with J2 = 0.5, tasks includes
* energy, sample the energy.
* szsz, calculate the <sz(i)*sz(j)> correlation matrix.

Also, we can obtain some exact quantities in simulation for analysis
* fidelity, calculated the exact fidelity
* energy_exact, calculated the exact energy

Pre-trained ansatz includes
* --symmetry su2, --depth <1-5>
* --symmetry u1, --depth 5
* --symmetry random, --depth 5
"""
@main function measure(task::String; symmetry::Symbol=:su2, depth::Int=5)
    model = simple_model_j1j2(4, 4)
    ansatz = simple_ansatz(16, symmetry, depth; load_params=true)
    nbit = nbit_simulated(ansatz)

    if task == "energy"
        eng_sample = energy(ansatz, model)/nbit
        println("Sampled value of energy/site = $eng_sample")
    elseif task == "energy_exact"
        eng = energy_exact(ansatz, model)/nbit
        println("Exact value of energy/site = $eng")
    elseif task == "fidelity"
        gs = ground_state(model)
        fid = fidelity_exact(ansatz, gs)[]
        println("Exact value of fidelity = $fid")
    elseif task == "szsz"
        correlation_matrix(ansatz; SAVE_ID=symmetry)
    else
        throw(ArgumentError("$task is not defined!"))
    end
end
