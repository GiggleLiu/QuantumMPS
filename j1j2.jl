include("applications.jl")

using Fire

@main function train(symmetry::Symbol; depth=5)
    model = simple_model_j1j2(4, 4)
    ansatz = simple_ansatz(16, symmetry, depth; load_params=false)

    run_train(ansatz, model; SAVE_ID=Symbol(VER,:_d,nlayer), niter=500, start_point=0)
end

@main function trained(task::String, symmetry::Symbol=:su2; depth::Int=5)
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
        ansatz = simple_ansatz(16, symmetry, depth; load_params=true)
        correlation_matrix(ansatz, Z; SAVE_ID=symmetry)
    else
        throw(ArgumentError("$task is not defined!"))
    end
end
