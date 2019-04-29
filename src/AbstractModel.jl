export AbstractModel, Heisenberg
export heisenberg_ij, hamiltonian, heisenberg_term, ground_state, energy, energy_exact, get_bonds, energy, heisenberg_2d, nspin

abstract type AbstractModel{D} end
abstract type AbstractHeisenberg{D} <: AbstractModel{D} end

nspin(model::AbstractModel) = prod(size(model))

"""
    energy_exact(tc::QuantumMPS, model::AbstractModel) -> Float64

Exact ground state energy.
"""
function energy_exact(tc::QuantumMPS, model::AbstractModel)
    nbit = nbit_simulated(tc)
    expect(hamiltonian(model), state_exact(tc)) |> real
end

"""
    ground_state(model::AbstractModel) -> DefaultRegister

Get the exact ground state of a model.
"""
function ground_state(model::AbstractModel)
    # get the ground state
    hami = hamiltonian(model)
    E, v = eigsolve(mat(hami), 1, :SR)
    register(v[1])
end

"""
    get_bonds(model::AbstractHeisenberg) -> Vector

Get the weighted bonds of a Heisenberg model in the form Tuple(i,j,w_ij).
"""
function get_bonds end

include("Heisenberg.jl")
include("J1J2.jl")
include("TFI.jl")
