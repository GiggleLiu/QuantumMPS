function su2_unit(nbit::Int, i::Int, j::Int)
    put(nbit, (i,j)=>rot(SWAP, 0.0))
end

"""
    su2_circuit(nbit_virtual::Int, nlayer::Int, nrepeat::Int, pairs::Vector) -> Sequence

SU(2) symmetry quantum circuit ansatz for evolving states in S^2 = 0 good quantum number block. It requires `2+nbit_virtual` qubits, `pairs` is the geometry of entanglements.
"""
function su2_circuit(nbit_virtual::Int, nlayer::Int, nrepeat::Int, pairs::Vector)
    circuit = sequence()
    nbit_used = 2 + nbit_virtual
    for i=1:nrepeat
        unit = chain(nbit_used)
        if i==1
            for j=2-(nrepeat%2):2:nbit_virtual
                push!(unit, singlet_block(nbit_used, j, j+1))
            end
        end
        if i%2 != nrepeat%2
            push!(unit, singlet_block(nbit_used, 1, nbit_used))
        else
            if i!=1
                push!(unit, swap(nbit_used, 1, nbit_used))   # fix swap parameter order!
            end
        end
        for j=1:nlayer
            nring = nbit_virtual+1
            nring <= 1 && continue
            ops = [su2_unit(nbit_used, i, j) for (i,j) in pairs]
            push!(unit, chain(nbit_used, ops))
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

"""construct a circuit for generating singlets."""
function singlet_block(nbit::Int, i::Int, j::Int)
    unit = chain(nbit)
    push!(unit, put(nbit, i=>chain(X, H)))
    push!(unit, control(nbit, -i, j=>X))
end

function model(::Val{:su2}; nbit, V, B=4096, nlayer=5, pairs)
    nrepeat = nbit - V
    c = su2_circuit(V, nlayer, nrepeat, pairs) |> autodiff(:QC)
    chem = QuantumMPS(1, V, 1, c, zero_state(V+2, B), zeros(Int, nbit+1))
    chem
end
