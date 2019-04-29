using Yao

export random_circuit, pair_ring

"""
    pair_ring(n::Int) -> Vector

Pair ring.
"""
pair_ring(n::Int) = [i=>mod(i, n)+1 for i=1:n]

"""
    cnot_entangler(n::Int, pairs::Vector{Pair}) = ChainBlock

Arbitrary entangler unit, support lazy construction.
"""
cnot_entangler(n::Int, pairs) = chain(n, control(n, [ctrl], target=>X) for (ctrl, target) in pairs)

"""
    rotor(noleading::Bool=false, notrailing::Bool=false) -> MatrixBlock

`Rz(η)⋅Rx(θ)⋅Rz(ξ)`, remove the first Rz gate if `noleading == true`, remove the last Rz gate if `notrailing == true`.
"""
rotor(noleading::Bool=false, notrailing::Bool=false) = noleading ? (notrailing ? Rx(0) : chain(Rx(0), Rz(0))) : (notrailing ? chain(Rz(0), Rx(0)) : chain(Rz(0), Rx(0), Rz(0)))

"""
    rotorset(noleading::Bool=false, notrailing::Bool=false) -> ChainBlock

A sequence of rotors applied on all sites.
"""
rotorset(nbit::Int, noleading::Bool=false, notrailing::Bool=false) = chain(nbit, [put(nbit, j=>rotor(noleading, notrailing)) for j=1:nbit])

"""
A kind of widely used differentiable quantum circuit, angles in the circuit are randomely initialized.

ref:
    1. Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Chow, J. M., & Gambetta, J. M. (2017).
       Hardware-efficient Quantum Optimizer for Small Molecules and Quantum Magnets. Nature Publishing Group, 549(7671), 242–246.
       https://doi.org/10.1038/nature23879.
"""
function random_circuit(nbit_measure::Int, nbit_virtual::Int, nlayer::Int, nrepeat::Int, entangler_pairs)
    nbit_used = nbit_measure + nbit_virtual
    circuit = chain(nbit_used)
    entangler = cnot_entangler(nbit_used, entangler_pairs)

    for i=1:nrepeat
        unit = chain(nbit_used)
        for j=1:nlayer
            push!(unit, rotorset(nbit_used, false, false))
            push!(unit, entangler)
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function model(::Val{:general}; nbit::Int, V::Int, B::Int=4096, nlayer::Int=5, pairs)
    c = random_circuit(1, V, nlayer, nbit-V, pairs) |> autodiff(:QC)
    chem = QuantumMPS(1, V, 0, c, zero_state(V+1, nbatch=B), zeros(Int, nbit))
    chem
end
