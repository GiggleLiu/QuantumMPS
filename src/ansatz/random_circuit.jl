using Yao, Yao.Blocks

export random_circuit, pair_ring

"""
    pair_ring(n::Int) -> Vector

Pair ring.
"""
pair_ring(n::Int) = [i=>mod(i, n)+1 for i=1:n]

"""
    cnot_entangler(T, n::Int, pairs::Vector{Pair}) = ChainBlock

Arbitrary rotation unit, support lazy construction.
"""
cnot_entangler(::Type{T}, n::Int, pairs) where T = chain(n, control(n, [ctrl], target=>XGate{T}()) for (ctrl, target) in pairs)

"""
    rotor(T, noleading::Bool=false, notrailing::Bool=false) -> MatrixBlock

`Rz(η)⋅Rx(θ)⋅Rz(ξ)`, remove the first Rz gate if `noleading == true`, remove the last Rz gate if `notrailing == true`.
"""
rotor(::Type{T}, noleading::Bool=false, notrailing::Bool=false) where T = noleading ? (notrailing ? Rx(T, 0) : chain(Rx(T, 0), Rz(T, 0))) : (notrailing ? chain(Rz(T, 0), Rx(T, 0)) : chain(Rz(T, 0), Rx(T, 0), Rz(T, 0)))

"""
    rotorset(T, noleading::Bool=false, notrailing::Bool=false) -> ChainBlock

A sequence of rotors applied on all sites.
"""
rotorset(::Type{T}, nbit::Int, noleading::Bool=false, notrailing::Bool=false) where T = chain(nbit, [put(nbit, j=>rotor(T, noleading, notrailing)) for j=1:nbit])

"""
A kind of widely used differentiable quantum circuit, angles in the circuit are randomely initialized.

ref:
    1. Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Chow, J. M., & Gambetta, J. M. (2017).
       Hardware-efficient Quantum Optimizer for Small Molecules and Quantum Magnets. Nature Publishing Group, 549(7671), 242–246.
       https://doi.org/10.1038/nature23879.
"""
function random_circuit(::Type{T}, nbit_measure::Int, nbit_virtual::Int, nlayer::Int, nrepeat::Int, entangler_pairs) where T
    circuit = sequence()
    nbit_used = nbit_measure + nbit_virtual
    entangler = cnot_entangler(T, nbit_used, entangler_pairs)

    for i=1:nrepeat
        unit = chain(T, nbit_used)
        for j=1:nlayer
            push!(unit, rotorset(T, nbit_used, false, false))
            push!(unit, entangler)
            if i == nrepeat
                push!(unit, rotorset(T, nbit_used, false, false))
            else
                for i = 1:nbit_measure
                    push!(unit, put(nbit_used, i=>rotor(T, false, false)))
                end
            end
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function model(::Val{:random}, ::Type{T}; nbit::Int, V::Int, B::Int=4096, nlayer::Int=5) where T
    c = random_circuit(T, 1, V, nlayer, nbit-V, pair_ring(V+1)) |> autodiff(:QC)
    chem = QuantumMPS(1, V, 0, c, zero_state(T,V+1, B), zeros(Int, nbit))
    chem
end
