export getblock, nbit_used, nbit_simulated, nrepeat, chem2circuit, TNChem
export energy, state_exact, energy_exact, fidelity_exact, heisenberg_ground_state, measure_op2

struct TNChem{RT}
    nbit_measure::Int
    nbit_virtual::Int

    circuit::AbstractBlock
    initial_reg::RT

    op_eigen::Eigen
    input_state::Vector{Int}
    nbit_ancilla::Int
end

getblock(chem::TNChem, i::Int) = chem.circuit[i]
nrepeat(chem::TNChem) = length(chem.circuit)
nbit_used(chem::TNChem) = nqubits(chem.circuit[1])
nbit_simulated(chem::TNChem) = chem.nbit_measure*nrepeat(chem) + chem.nbit_virtual

##################### Type Conversion ######################
"""convert a chem circuit to a circuit with no reuse"""
function chem2circuit(tnchem)
    nbit = nbit_simulated(tnchem)
    nm = tnchem.nbit_measure
    nv = tnchem.nbit_virtual
    c = chain(nbit)
    for (i, blk) in enumerate(tnchem.circuit)
        push!(c, concentrate(nbit, blk, [(i-1)*nm+1:i*nm..., nbit-nv+1:nbit...]))
    end
    c
end

"""sample the energy for the chemistry application"""
energy(chem::TNChem) = energy(chem, Val(:even)) + energy(chem, Val(:odd))
function energy(chem::TNChem, subset::Union{Val{:even}, Val{:odd}})
    input_state = chem.input_state
    reg = chem.initial_reg |> copy
    insert_qubit!(reg, 2)  # add an additional qubit
    nv = chem.nbit_virtual
    nrep = nrepeat(chem)
    local eng = 0.0
    for i = nrep+1:nrep+nv
        input_state[i] == 1 && apply!(reg, put(nv+2, (i-nrep+2)=>X))
    end
    if subset == Val(:even)
        focus!(reg, 2:2+nv) do reg
            input_state[1] == 1 && apply!(reg, put(nv+1, 1=>X))
            reg |> getblock(chem, 1)
            measure_reset!(reg, 1)
            reg
        end
        start = 2
    else
        start = 1
    end
    for i=start:2:nrep-1
        focus!(reg, [1,(3:2+nv)...]) do reg
            input_state[i] == 1 && apply!(reg, put(nv+1, 1=>X))
            reg |> getblock(chem, i)
        end
        focus!(reg, 2:2+nv) do reg
            input_state[i+1] == 1 && apply!(reg, put(nv+1, 1=>X))
            reg |> getblock(chem, i+1)
        end
        eng += measure_reset!(chem.op_eigen, reg, 1:2) |> real |> mean
    end
    if (nrep-start)%2 == 0
        focus!(reg, 2:2+nv) do reg
            input_state[nrep] == 1 && apply!(reg, put(nv+1, 1=>X))
            reg |> getblock(chem, nrep)
        end
        projected_bits = 1:1
    else
        projected_bits = 1:2
    end
    reg = select(focus!(reg, projected_bits), 0) |> relax!
    nbit = nqubits(reg)
    for i=1:2:nbit-1-chem.nbit_ancilla
        eng += measure_remove!(chem.op_eigen, reg, 1:2) |> real |> mean
        reg
    end
    eng
end

function state_exact(chem::TNChem)
    circuit = chem2circuit(chem)
    if chem.nbit_ancilla == 0
        return product_state(nqubits(circuit), chem.input_state|>Yao.Intrinsics.packbits) |> circuit
    else
        nbit = nqubits(circuit)
        product_state(nbit, chem.input_state|>Yao.Intrinsics.packbits) |> circuit |> focus!((1:nbit-chem.nbit_ancilla)...) |> remove_env!
    end
end

function energy_exact(tc::TNChem)
    nbit = nbit_simulated(tc)-tc.nbit_ancilla
    expect(heisenberg(nbit, periodic=false), state_exact(tc)) |> real
end

function fidelity_exact(chem::TNChem, ground_state::AbstractRegister)
    fidelity(ground_state, state_exact(chem))
end

function heisenberg_ground_state(nbit::Int)
    # get the ground state
    hami = heisenberg(nbit; periodic=false)
    E, v = eigsolve(mat(hami), 1, :SR)
    @show E[1]
    register(v[1])
end

function measure_op2(chem::TNChem, op::MatrixBlock, i::Int, j::Int; reverse::Bool=false)
    input_state = chem.input_state
    i > j && return measure_op2(chem, op, j, i)
    i == j && throw(ArgumentError("i should not equal to j!"))
    reg = chem.initial_reg |> copy
    nv = chem.nbit_virtual
    nrep = nrepeat(chem)

    for k = nrep+1:nrep+nv
        input_state[k] == 1 && apply!(reg, put(nv+1, (k-nrep+1)=>X))
    end

    for k=1:min(i, nrep)-1
        input_state[k] == 1 && apply!(reg, put(nv+1, 1=>X))
        reg |> getblock(chem, k)
        measure_reset!(reg, 1)
    end

    k = min(i, nrep)
    input_state[k] == 1 && apply!(reg, put(nv+1, 1=>X))
    reg |> getblock(chem, k)
    #addbit!(reg, 1)
    #reorder!(reg, [1, nv+2, (2:1+nv)...])  # this is wrong!
    insert_qubit!(reg, 2)
    focus!(reg, 2:2+nv) do reg
        for k = i+1:min(j, nrep)
            input_state[k] == 1 && apply!(reg, put(nv+1, 1=>X))
            reg |> getblock(chem, k)
            k==j || measure_reset!(reg, 1)
        end
        reg
    end
    II = i<=nrep ? 1 : i-nrep+2
    JJ = max(0, j-nrep)+2
    if reverse
        op = put(nv+2, (JJ, II)=>op)
    else
        op = put(nv+2, (II, JJ)=>op)
    end
    val = expect(op, reg) |> mean
    val
end

function remove_env!(reg::DefaultRegister)
    reg.state = dropdims(sum(reg |> rank3, dims=2), dims=2)
    reg
end
