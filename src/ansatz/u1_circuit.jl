function u1_unit(::Type{T}, nbit::Int, i::Int, j::Int) where T
    chain(nbit, put(nbit, i=>Rz(T, 0)),
    put(nbit, j=>Rz(T, 0)),
    put(nbit, (i,j)=>rot(SWAPGate{T}(), 0))
    )
end

function u1_circuit(::Type{T}, nbit_measure::Int, nbit_virtual::Int, nlayer::Int, nrepeat::Int, entangler_pairs) where T
    circuit = sequence()
    nbit_used = nbit_measure + nbit_virtual

    for i=1:nrepeat
        unit = chain(T, nbit_used)
        for j=1:nlayer
            push!(unit, chain(nbit_used, u1_unit(T, nbit_used, i, j) for (i,j) in entangler_pairs))
            for k=1:(i==nrepeat ? nbit_used : nbit_measure)
                put(nbit_used, k=>Rz(T, 0))
            end
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function model(::Val{:u1}, ::Type{T}; nbit = 20, V=4, B=4096, nlayer=5, pairs) where T
    nrepeat = (nbit - V)
    c = u1_circuit(T, 1, V, nlayer, nrepeat, pairs) |> autodiff(:QC)
    chem = QuantumMPS(1, V, 0, c, zero_state(T, V+1, B), [i%2 for i=1:nbit])
    chem
end
