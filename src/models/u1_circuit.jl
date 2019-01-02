function goodqn_unit(ver::Val{:u1}, nbit::Int, i::Int, j::Int)
    chain(nbit, put(nbit, i=>rot(Z, 0.0)),
    put(nbit, j=>rot(Z, 0.0)),
    put(nbit, (i,j)=>rot(SWAP, 0.0))
    )
end

function goodqn_circuit(ver::Val{:u1}, nbit_measure::Int, nbit_virtual::Int, nlayer::Int, nrepeat::Int, entangler_pairs)
    circuit = sequence()
    nbit_used = nbit_measure + nbit_virtual

    for i=1:nrepeat
        unit = chain(nbit_used)
        for j=1:nlayer
            push!(unit, chain(nbit_used, goodqn_unit(ver, nbit_used, i, j) for (i,j) in entangler_pairs))
            for k=1:(i==nrepeat ? nbit_used : nbit_measure)
                put(nbit_used, k=>rot(Z, 0.0))
            end
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function chem_sample(ver::Val{:u1}, nbit::Int=8; nbit_virtual::Int=4, nbatch=1000, nlayer=5, input_state=[i%2 for i=1:nbit])
    nrepeat = (nbit - nbit_virtual)
    basicpairs = pair_ring(1+nbit_virtual)
    c = goodqn_circuit(ver, 1, nbit_virtual, nlayer, nrepeat, basicpairs) |> autodiff(:QC)

    ei = eigen!(mat(heisenberg_term) |> Matrix)
    chem = TNChem(1, nbit_virtual, c, zero_state(nbit_virtual+1, nbatch), ei, input_state, 0)
end

function model(::Val{:u1}; nbit = 20, V = 4)
    chem = chem_sample(Val(:u1), nbit, nbatch=4096, nbit_virtual=V)
    println("Number of parameters is ", chem.circuit |> nparameters)
    #chem |> cu
    chem
end
