using Yao, Yao.Blocks, Yao.Intrinsics
using Plots, LinearAlgebra, Statistics

CZ(pairs) = chain(control([ctrl, ], target => Z) for (ctrl, target) in pairs)
cnot_entangler(pairs) = chain(control([ctrl, ], target => X) for (ctrl, target) in pairs)

function cluster_generator(nqubit)
    reg0 = zero_state(nqubit)
    circuit = chain(nqubit)
    push!(circuit, rollrepeat(nqubit, H))
    for i in 2:nqubit
        push!(circuit, CZ([i => i - 1]))
    end
    reg0 |> circuit
    reg0
end

function correlation_func(reg, observable1, observable2, pos1, pos2)
    reg0 = copy(reg)
    reg_temp = copy(reg)
    reg_temp |> put(nqubits(reg), pos1 => observable1)
    reg_temp |> put(nqubits(reg), pos2 => observable2)
    reg0' * reg_temp
end

function Ureg(reg, pos)
    nbit = nqubits(reg)
    Urt = chain(nbit, [put(nbit, pos=>Rz(0.0)), put(nbit, pos=>Rx(0.0)), put(nbit, pos=>Rz(0.0))])
    dispatch!(Urt, :random)
    reg |> Urt
    reg
end

reg = cluster_generator(20)
Ureg(reg, 10)
correlation_func(reg, Z, Z, 9, 11)
