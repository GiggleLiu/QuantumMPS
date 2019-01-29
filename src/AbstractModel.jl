export AbstractModel, Heisenberg
export heisenberg_ij, hamiltonian, heisenberg_term, ground_state, energy, energy_exact, get_bonds, energy, heisenberg_2d, nspin

abstract type AbstractModel{D} end
nspin(model::AbstractModel) = prod(size(model))

"""
Exact ground state energy.
"""
function energy_exact(tc::QuantumMPS, model::AbstractModel)
    nbit = nbit_simulated(tc)
    expect(hamiltonian(model), state_exact(tc)) |> real
end

"""
Ground state energy by sampling Quantum MPS.
The hamiltonian is limited to Heisenberg and J1J2 Type.
"""
function energy(chem::QuantumMPS, model::AbstractModel)
    T = datatype(chem.initial_reg)
    energy(chem, XGate{T}(), model) + energy(chem, YGate{T}(), model) + energy(chem, ZGate{T}(), model)
end

function energy(chem::QuantumMPS, pauli::PauliGate, model::AbstractModel)
    input_state = chem.input_state
    reg = chem.initial_reg |> copy
    nv = chem.nbit_virtual + chem.nbit_ancilla
    nrep = nrepeat(chem)
    T = datatype(chem.initial_reg)

    op = eigen!(pauli |> mat |>Matrix)
    rotor = put(nv+1, 1=>matrixgate(T.(op.vectors' |> Matrix)))
    local eng = 0.0
    local res = similar(reg |> state, Int, nbatch(reg), nbit_simulated(chem))
    for i = nrep+1:nrep+nv
        input_state[i] == 1 && apply!(reg, put(nv+1, (i-nrep+1)=>X))
    end
    input_state[1] == 1 && apply!(reg, put(nv+1, 1=>X))
    for i=1:nrep
        reg |> getblock(chem, i)
        if i!=nrep
            reg |> rotor
            @inbounds res[:,i] = 1 .- 2 .* measure_reset!(reg, 1, val=input_state[i+1])
        end
    end
    for i=1:nv+1-chem.nbit_ancilla
        reg |> rotor
        @inbounds res[:,i+nrep-1] = 1 .- 2 .* measure_remove!(reg, 1)
    end
    for bond in get_bonds(model)
        eng += bond[3]*mean(res[:,bond[1]].*res[:,bond[2]])
    end
    eng/4
end

function ground_state(model::AbstractModel)
    # get the ground state
    hami = hamiltonian(model)
    E, v = eigsolve(mat(hami), 1, :SR)
    register(v[1])
end


include("Heisenberg.jl")
include("J1J2.jl")
