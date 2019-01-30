export getblock, nbit_used, nbit_simulated, nrepeat, chem2circuit, QuantumMPS
export state_exact, fidelity_exact

"""
    QuantumMPS{RT}

Members:
    `nbit_measure`, number of qubits measured in a single iteration, or physical qubits.
    `nbit_virtual`, number of virtual qubits to represent the virtual bond dimension in quantum MPS.
    `circuit`, the circuit structure (measurements are not included).
    `initial_reg`, the initial state (GPUReg or a regular one), always prepaired in |0>.
    `nbit_ancilla`, number of ancilla qubits.
"""
struct QuantumMPS{RT}
    nbit_measure::Int
    nbit_virtual::Int
    nbit_ancilla::Int

    circuit::AbstractBlock
    initial_reg::RT

    input_state::Vector{Int}
end

getblock(chem::QuantumMPS, i::Int) = chem.circuit[i]
nrepeat(chem::QuantumMPS) = length(chem.circuit)
nbit_used(chem::QuantumMPS) = nqubits(chem.circuit[1])
nbit_simulated(chem::QuantumMPS) = chem.nbit_measure*nrepeat(chem) + chem.nbit_virtual

"""convert a chem circuit to a circuit with no reuse"""
function chem2circuit(tnchem)
    nbit = nbit_simulated(tnchem) + tnchem.nbit_ancilla
    nm = tnchem.nbit_measure
    nv = tnchem.nbit_virtual + tnchem.nbit_ancilla
    c = chain(nbit)
    for (i, blk) in enumerate(tnchem.circuit)
        push!(c, concentrate(nbit, blk, [(i-1)*nm+1:i*nm..., nbit-nv+1:nbit...]))
    end
    c
end

function state_exact(chem::QuantumMPS)
    circuit = chem2circuit(chem)
    if chem.nbit_ancilla == 0
        return product_state(nqubits(circuit), chem.input_state|>Yao.Intrinsics.packbits) |> circuit
    else
        nbit = nqubits(circuit)
        product_state(nbit, chem.input_state|>Yao.Intrinsics.packbits) |> circuit |> focus!((1:nbit-chem.nbit_ancilla)...) |> remove_env!
    end
end

function remove_env!(reg::DefaultRegister)
    reg.state = dropdims(sum(reg |> rank3, dims=2), dims=2)
    reg
end

function fidelity_exact(chem::QuantumMPS, ground_state::AbstractRegister)
    fidelity(ground_state, state_exact(chem))
end
