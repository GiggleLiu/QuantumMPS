export chem_sample, model
chem_sample(which::Symbol, args...; kwargs...) = chem_sample(Val(which), args...; kwargs...)

include("random_circuit.jl")
include("u1_circuit.jl")
include("su2_circuit.jl")
