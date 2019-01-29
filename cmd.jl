include("applications.jl")

using Fire

@main function train(symmetry)
    model = simple_model_j1j2(4, 4)
    ansatz = simple_ansatz(16, Symbol(symmetry), 5, true)

    run_train(ansatz, model; SAVE_ID=Symbol(VER,:_d,nlayer), niter=500, start_point=0)
end

@main function zzcorrelation(symmetry)
    symmetry = Symbol(symmetry)
    ansatz = simple_ansatz(16, symmetry, 5, true)
    correlation_matrix(ansatz, Z; SAVE_ID=symmetry)
end

@main function energy_fidelity(symmetry, depth=5)
    model = simple_model_j1j2(4, 4)
    ansatz = simple_ansatz(16, Symbol(symmetry), depth, true)
    eng = energy_exact(ansatz, model)/nbit_simulated(ansatz)
    gs = ground_state(model)
    fid = fidelity_exact(ansatz, gs)[]
    println("Expectation value of energy/site = $eng, fidelity = $fid")
end
