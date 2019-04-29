include("applications.jl")

function train(nsite;depth::Int=2)
    symmetry = :twoqubit
    model = TFI(nsite; h=0.5, periodic=false)
    ansatz = simple_ansatz(nsite, symmetry, depth; load_params=false)

    run_train(ansatz, model; SAVE_ID=Symbol(symmetry,:_d,depth), niter=500, start_point=0)
end

function measure(task::String; symmetry::Symbol=:su2, depth::Int=5)
    model = simple_model_heis(6)
    ansatz = simple_ansatz(6, symmetry, depth; load_params=true)
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
