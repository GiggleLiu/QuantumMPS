export TFI

struct TFI{D} <: AbstractModel{D}
    size::NTuple{D, Int}
    h::Float64
    periodic::Bool
    TFI(size::Int...; h::Real, periodic::Bool) = new{length(size)}(size, Float64(h), periodic)
end

function get_bonds(model::TFI{1})
    nbit, = model.size
    [(i, i%nbit+1, 1.0) for i in 1:(model.periodic ? nbit : nbit-1)]
end

Base.size(model::TFI) = model.size

function hamiltonian(model::TFI{1})
    model.periodic && throw()
    nbit = nspin(model)
    sum(repeat(nbit, Z, (i,i+1)) for i=1:nbit-1)*0.25 +
    sum(put(nbit, i=>X) for i=1:nbit)*0.5model.h
end


function energy(chem::QuantumMPS, model::TFI)
   energy(chem, Z, model) +
    energy(chem, X, model)
end

function energy(chem::QuantumMPS, pauli::ZGate, model::TFI)
    res = gensample(chem, pauli)
    local eng = 0.0
    for bond in ((i, i+1, 1.0) for i=1:nspin(model)-1)
        eng += bond[3]*mean(res[:,bond[1]].*res[:,bond[2]])
    end
    eng/4
end

function energy(chem::QuantumMPS, pauli::XGate, model::TFI)
    res = gensample(chem, pauli)
    eng = mean(sum(res, dims=2))
    model.h*eng/2
end
