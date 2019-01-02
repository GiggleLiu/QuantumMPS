using Yao, Yao.Blocks

export random_circuit, pair_ring

"""
    pair_ring(n::Int) -> Vector

Pair ring.
"""
pair_ring(n::Int) = [i=>mod(i, n)+1 for i=1:n]

"""
    cnot_entangler([n::Int, ] pairs::Vector{Pair}) = ChainBlock

Arbitrary rotation unit, support lazy construction.
"""
cnot_entangler(n::Int, pairs) = chain(n, control(n, [ctrl], target=>X) for (ctrl, target) in pairs)
cnot_entangler(pairs) = n->cnot_entangler(n, pairs)

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
    circuit = sequence()
    entangler = cnot_entangler(entangler_pairs)
    #use_cache && (entangler = entangler |> cache)
    nbit_used = nbit_measure + nbit_virtual

    for i=1:nrepeat
        unit = chain(nbit_used)
        for j=1:nlayer
            push!(unit, rotorset(nbit_used, false, false))
            push!(unit, entangler)
            if i == nrepeat
                push!(unit, rotorset(nbit_used, false, false))
            else
                for i = 1:nbit_measure
                    push!(unit, put(nbit_used, i=>rotor(false, false)))
                end
            end
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function toy_chem(nbit_virtual::Int, nlayer::Int, nrepeat::Int;
                  entangle_pairs=pair_ring(nbit_virtual+1), nbatch::Int=1000, use_cache=false, input_state=zeros(Int, nbit_virtual+nrepeat), nbit_ancilla::Int=0)
    c = random_circuit(1, nbit_virtual, nlayer, nrepeat, entangle_pairs) |> autodiff(:QC)

    ei = eigen!(mat(heisenberg_term) |> Matrix)
    chem = TNChem(1, nbit_virtual, c, zero_state(nbit_virtual+1, nbatch), ei, input_state, nbit_ancilla)
end

function chem_sample(::Val{:random}, nbit::Int=8; nbit_virtual::Int=4, nbatch=4096, nlayer=5, input_state=zeros(Int, nbit))
    nrepeat = (nbit - nbit_virtual)
    basicpairs = pair_ring(1+nbit_virtual)
    toy_chem(nbit_virtual, nlayer, nrepeat, nbatch=nbatch, use_cache=false, entangle_pairs=basicpairs, input_state=input_state)
end

function model(::Val{:random}; nbit = 20, V = 4, B=4096)
    chem = chem_sample(Val(:random), nbit, nbatch=B, nbit_virtual=V)
    println("Number of parameters is ", chem.circuit |> nparameters)
    #chem |> cu
    chem
end
