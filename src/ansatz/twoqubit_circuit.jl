export twoqubit_circuit

function twoqubit_circuit(nlayer::Int, nrepeat::Int)
    nbit_measure = nbit_virtual = 1
    nbit_used = nbit_measure + nbit_virtual
    circuit = chain(nbit_used)

    for i=1:nrepeat
        unit = chain(nbit_used)
        for j=1:nlayer
            push!(unit, put(nbit_used, 1=>rotor(true, false)))
            #push!(unit, put(nbit_used, 2=>rotor(true, false)))
            push!(unit, control(nbit_used, 1, 2=>(j%2==1 ? X : Z)))
            j == nlayer && push!(unit, put(nbit_used, 1=>rotor(false, true)))
            #j == nlayer && push!(unit, put(nbit_used, 2=>rotor(false, true)))
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function model(::Val{:twoqubit}; nbit::Int, B::Int=4096, nlayer::Int=5, kwargs...)
    V = 1
    c = twoqubit_circuit(nlayer, nbit-V) |> autodiff(:QC)
    chem = QuantumMPS(1, V, 0, c, zero_state(V+1, nbatch=B), zeros(Int, nbit))
    chem
end
