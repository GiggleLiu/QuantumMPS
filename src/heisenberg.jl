export AbstractModel, Heisenberg
export heisenberg_ij, hamiltonian, heisenberg_term, ground_state, energy, energy_exact, get_bonds, energy, heisenberg_2d, nspin

abstract type AbstractModel{D} end

struct Heisenberg{D} <: AbstractModel{D}
    size::NTuple{D, Int}
    periodic::Bool
    Heisenberg(size::Int...; periodic::Bool) = new{length(size)}(size, periodic)
end

nspin(model) = prod(model.size)
heisenberg_ij(nbit::Int, i::Int, j::Int=i+1) = put(nbit, i=>X)*put(nbit, j=>X) + put(nbit, i=>Y)*put(nbit, j=>Y) + put(nbit, i=>Z)*put(nbit, j=>Z)
const heisenberg_term = repeat(2, X, 1:2) + repeat(2, Y, 1:2) + repeat(2, Z, 1:2)

function get_bonds(model::Heisenberg{2})
    m, n = model.size
    cis = LinearIndices(model.size)
    bonds = Pair{Int, Int}[]
    for i=1:model.size[1], j=1:model.size[2]
        (i!=m || model.periodic) && push!(bonds, cis[i,j] => cis[i%m+1,j])
        (j!=n || model.periodic) && push!(bonds, cis[i,j] => cis[i,j%n+1])
    end
    bonds
end

function get_bonds(model::Heisenberg{1})
    nbit, = model.size
    [i=>i%nbit+1 for i in 1:(model.periodic ? nbit : nbit-1)]
end

function hamiltonian(model::Heisenberg)
    nbit = nspin(model)
    sum(x->heisenberg_ij(nbit, x.first, x.second), get_bonds(model))*0.25
end

function ground_state(model::Heisenberg)
    # get the ground state
    hami = hamiltonian(model)
    E, v = eigsolve(mat(hami), 1, :SR)
    @show E[1]
    register(v[1])
end

function energy_exact(tc::QuantumMPS, model::AbstractModel)
    nbit = nbit_simulated(tc)
    expect(hamiltonian(model), state_exact(tc)) |> real
end

"""
Ground state energy for Heisenberg chain.
"""
# 3x3 heisenberg hamiltonian
function energy(chem::QuantumMPS, model::AbstractModel)
    T = datatype(chem.initial_reg)
    energy(chem, XGate{T}(), model) + energy(chem, YGate{T}(), model) + energy(chem, ZGate{T}(), model)
end

#=
function energy(chem::QuantumMPS, pauli::PauliGate)
    input_state = chem.input_state
    reg = chem.initial_reg |> copy
    nv = chem.nbit_virtual
    nrep = nrepeat(chem)
    T = datatype(chem.initial_reg)

    op = eigen!(pauli |> mat |>Matrix)
    rotor = put(nv+1, 1=>matrixgate(T.(op.vectors' |> Matrix)))
    local eng = 0.0
    local res_pre
    local res
    for i = nrep+1:nrep+nv
        input_state[i] == 1 && apply!(reg, put(nv+1, (i-nrep+1)=>X))
    end
    input_state[1] == 1 && apply!(reg, put(nv+1, 1=>X))
    for i=1:nrep
        reg |> getblock(chem, i)
        if i!=nrep
            reg |> rotor
            @inbounds res = 1 .- 2 .* measure_reset!(reg, 1, val=input_state[i+1])
            i>1 && (eng += mean(res_pre.*res))
            res_pre = res
        end
    end
    for i=1:nv+1-chem.nbit_ancilla
        reg |> rotor
        @inbounds res = 1 .- 2 .* measure_remove!(reg, 1)
        eng += mean(res_pre.*res)
        res_pre = res
    end
    eng/4
end
=#

function energy(chem::QuantumMPS, pauli::PauliGate, model::AbstractModel)
    input_state = chem.input_state
    reg = chem.initial_reg |> copy
    nv = chem.nbit_virtual
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
        eng += mean(res[:,bond.first].*res[:,bond.second])
    end
    eng/4
end
