function goodqn_unit(ver::Val{:su2}, nbit::Int, i::Int, j::Int)
    put(nbit, (i,j)=>rot(SWAP, 0.0))
end

function su2_circuit(nbit_virtual::Int, nlayer::Int, nrepeat::Int)
    circuit = sequence()
    nbit_used = 1 + nbit_virtual
    for i=1:nrepeat
        unit = chain(nbit_used)
        if i==1
            for j=2:2:nbit_virtual
                push!(unit, singlet_block(nbit_used, j, j+1))
            end
        end
        if i%2 == 1
            push!(unit, singlet_block(nbit_used, 1, nbit_used))
        else
            push!(unit, swap(nbit_used, 1, nbit_used))
        end
        for j=1:nlayer
            nring = nbit_virtual
            nring <= 1 && continue
            ops = [goodqn_unit(Val(:su2), nbit_used, i, j) for (i,j) in pair_ring(nring)]
            push!(unit, chain(nbit_used, ops))
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function singlet_block(nbit::Int, i::Int, j::Int)
    unit = chain(nbit)
    push!(unit, put(nbit, i=>chain(X, H)))
    push!(unit, control(nbit, -i, j=>X))
end

function chem_sample(ver::Val{:su2}, nbit::Int=21; nbit_virtual::Int=5, nbatch=4096, nlayer=5, input_state=[0 for i=1:nbit])
    nrepeat = (nbit - nbit_virtual)
    c = su2_circuit(nbit_virtual, nlayer, nrepeat) |> autodiff(:QC)

    ei = eigen!(mat(heisenberg_term) |> Matrix)
    chem = TNChem(1, nbit_virtual, c, zero_state(nbit_virtual+1, nbatch), ei, input_state, 1)
end

function model(::Val{:su2}; nbit=21, VER=4, V=5, B=4096)
    chem = chem_sample(Val(:su2), nbit, nbatch=B, nbit_virtual=V)
    println("Number of parameters is ", chem.circuit |> nparameters)
    #chem |> cu
    chem
end
