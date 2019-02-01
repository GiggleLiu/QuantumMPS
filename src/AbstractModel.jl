export AbstractModel, Heisenberg
export heisenberg_ij, hamiltonian, heisenberg_term, ground_state, energy, energy_exact, get_bonds, energy, heisenberg_2d, nspin

abstract type AbstractModel{D} end
abstract type AbstractHeisenberg{D} <: AbstractModel{D} end

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
function energy(chem::QuantumMPS, model::AbstractHeisenberg)
    energy(chem, X, model) + energy(chem, Y, model) + energy(chem, Z, model)
end

function energy(chem::QuantumMPS, pauli::PauliGate, model::AbstractHeisenberg)
    res = gensample(chem, pauli)
    local eng = 0.0
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
