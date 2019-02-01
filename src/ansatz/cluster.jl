cluster_block(isfirst::Val{true}) = chain(2, [repeat(2, H, 1:2), control(2, 1, 2=>Z)])
cluster_block(isfirst::Val{false}) = chain(2, [swap(2, 1, 2), put(2, 2=>H), control(2, 1, 2=>Z)])

function cluster_circuit(nrepeat::Int)
    sequence([cluster_block(Val(i==1)) for i=1:nrepeat])
end

function model(::Val{:cluster}; nbit, B=4096)
    nrepeat = nbit - 1
    c = cluster_circuit(nrepeat)
    chem = QuantumMPS(1, 1, 0, c, zero_state(2, B), zeros(Int, nbit))
    chem
end
