export run_train, run_corr
using DelimitedFiles

function run_train(chem, ground_state; VER, niter=500)
    qo = TNChemOptimizer(chem, Adam(lr=0.1))

    history = Float64[]
    fidelities = Float64[]
    nbit = nbit_simulated(chem)
    V = chem.nbit_virtual
    @time for (k, p) in enumerate(qo)
        curr_loss = energy(chem)/(nbit-chem.nbit_ancilla)/4
        fid = fidelity_exact(chem, ground_state)[]
        push!(history, curr_loss)
        push!(fidelities, fid)
        println("k = $k, loss = $curr_loss, fidelity = $(fid)")
        flush(stdout)
        k >= niter && break
    end

    for (token, var) in [("loss", history),
                         ("params", parameters(chem.circuit)),
                         ("fidelity", fidelities),
                        ]
        writedlm("data/chem_$(VER)_$(token)_N$(nbit)_V$V.dat", var)
    end
end

function run_corr(chem; VER)
    nbit = nbit_simulated(chem)
    V = chem.nbit_virtual
    # load wave function
    params = readdlm("data/chem_$(VER)_params_N$(nbit)_V$(V).dat")
    dispatch!(chem.circuit, params)
    @show energy_exact(chem)

    om = zeros(Float64, nbit, nbit)
    for i =1:nbit-chem.nbit_ancilla, j=1:nbit-chem.nbit_ancilla
        if i!=j
            om[i,j] = measure_op2(chem, repeat(2, Z, 1:2), i, j) |> real
            println("<O($i,$j)> = $(om[i,j])")
            flush(stdout)
        end
    end

    for (token, var) in [
                         ("om", om),
                        ]
        writedlm("data/chem_goodqn$(VER)_$(token)_N$(nbit)_V$V.dat", var)
    end
end


