export model

"""
    model(which::Symbol; nbit::Int, V::Int, B::Int=4096, nlayer::Int=5)

predefined models, `which` should be one of :random, :u1, :su2.
* `nbit` is the system size (length of MPS),
* `V` is the number of virtual qubits,
* `B` is the batch size.
* `nlayer` is the number of layers in a block.
"""
model(which::Symbol, args...; kwargs...) = model(Val(which), args...; kwargs...)

include("general_circuit.jl")
include("u1_circuit.jl")
include("su2_circuit.jl")
include("cluster.jl")
