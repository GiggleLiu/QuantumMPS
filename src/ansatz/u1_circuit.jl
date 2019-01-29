function u1_unit(nbit::Int, i::Int, j::Int)
    chain(nbit, put(nbit, i=>Rz(0)),
    put(nbit, j=>Rz(0)),
    put(nbit, (i,j)=>rot(SWAP, 0))
    )
end

function u1_circuit(nbit_measure::Int, nbit_virtual::Int, nlayer::Int, nrepeat::Int, entangler_pairs)
    circuit = sequence()
    nbit_used = nbit_measure + nbit_virtual

    for i=1:nrepeat
        unit = chain(nbit_used)
        for j=1:nlayer
            push!(unit, chain(nbit_used, u1_unit(nbit_used, i, j) for (i,j) in entangler_pairs))
            for k=1:(i==nrepeat ? nbit_used : nbit_measure)
                put(nbit_used, k=>Rz(0))
            end
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function model(::Val{:u1}; nbit = 20, V=4, B=4096, nlayer=5, pairs)
    nrepeat = (nbit - V)
    c = u1_circuit(1, V, nlayer, nrepeat, pairs) |> autodiff(:QC)
    chem = QuantumMPS(1, V, 0, c, zero_state(V+1, B), [i%2 for i=1:nbit])
    chem
end
