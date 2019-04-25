using Yao
using Statistics: mean
using LinearAlgebra

rotor(noleading::Bool=false, notrailing::Bool=false) = noleading ? (notrailing ? Rx(0) : chain(Rx(0), Rz(0))) : (notrailing ? chain(Rz(0), Rx(0)) : chain(Rz(0), Rx(0), Rz(0)))

function twoqubit_circuit(nlayer::Int, nrepeat::Int)
    nbit_measure = nbit_virtual = 1
    nbit_used = nbit_measure + nbit_virtual
    circuit = chain(nbit_used)

    for i=1:nrepeat
        unit = chain(nbit_used)
        for j=1:nlayer
            push!(unit, put(nbit_used, 1=>rotor(true, false)))
            #push!(unit, put(nbit_used, 2=>rotor(true, false)))
            push!(unit, control(nbit_used, 1, 2=>(j%2==1 ? X : Z)))
            j == nlayer && push!(unit, put(nbit_used, 1=>rotor(false, true)))
            #j == nlayer && push!(unit, put(nbit_used, 2=>rotor(false, true)))
        end
        push!(unit, Measure{nbit_used, 1, AbstractBlock}(Z, (1,), 0, false))
        if i==nrepeat
            for k=2:nbit_used
                push!(unit, Measure{nbit_used, 1, AbstractBlock}(Z, (k,), 0, false))
            end
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

circuit = twoqubit_circuit(2, 2)

function gensample(circuit, basis; nbatch=1024)
    mblocks = collect_blocks(Measure, circuit)
    for m in mblocks
        m.operator = basis
    end
    reg = zero_state(nqubits(circuit); nbatch=nbatch)
    reg |> circuit
    mblocks
end

function energy(circuit, model::TFIChain; nbatch=1024)
    # measuring Z basis
    mblocks = gensample(circuit, Z; nbatch=nbatch)
    local eng = 0.0
    for (a, b, v) in ((i, i+1, 1.0) for i=1:model.length-1)
        eng += v*mean(mblocks[a].results .* mblocks[b].results)
    end
    eng/=4

    # measuring X basis
    mblocks = gensample(circuit, X; nbatch=nbatch)
    engx = sum(mean.([m.results for m in mblocks]))
    eng + model.h*engx/2
end

struct TFIChain
    length::Int
    h::Float64
    periodic::Bool
    TFIChain(length::Int; h::Real, periodic::Bool) = new(length, Float64(h), periodic)
end

function hamiltonian(model::TFIChain)
    model.periodic && throw()
    nbit = model.length
    sum(repeat(nbit, Z, (i,i+1)) for i=1:nbit-1)*0.25 +
    sum(put(nbit, i=>X) for i=1:nbit)*0.5model.h
end

using Test, Random

nbit_simulated(qmps) = length(collect_blocks(Measure, qmps))
function chem2circuit(circuit)
    nbit = nbit_simulated(circuit)
    nm = 1
    nv = 1
    c = chain(nbit)
    for (i, blk) in enumerate(circuit)
        blk = chain([b for b in blk if !(b isa Measure)]...)
        push!(c, concentrate(nbit, blk, [(i-1)*nm+1:i*nm..., nbit-nv+1:nbit...]))
    end
    c
end

function train(circuit, model; maxiter=200, α=0.1, nbatch=1024)
    rots = collect(RotationGate, circuit)
    for i in 1:maxiter
        for r in rots
            r.theta += π/2
            E₊ = energy(circuit, model; nbatch=nbatch)
            r.theta -= π
            E₋ = energy(circuit, model; nbatch=nbatch)
            r.theta += π/2
            g = 0.5(E₊ - E₋)
            r.theta -= g*α
        end
        println("Iter $i, E/N = $(energy(circuit, model, nbatch=nbatch)/model.length)")
    end
    circuit
end

nspin = 4
model = TFIChain(nspin; h=0.5, periodic=false)
h = hamiltonian(model)
EG = eigen(mat(h) |> Matrix).values[1]
@show EG/nspin

circuit = twoqubit_circuit(2, nspin-1)
train(circuit, model; α=0.5)

@testset "energy-goodqn tfi" begin
    Random.seed!(4)
    hei = TFIChain(4; h=0., periodic=false)
    nbit = hei.length
    circuit = twoqubit_circuit(2, nbit-1)
    println("Number of parameters is ", circuit|> nparameters)
    bigc = chem2circuit(circuit)
    eng = energy(circuit, hei; nbatch=10000)
    hami = hamiltonian(hei)
    @show bigc
    @show parameters(circuit)
    @show parameters(bigc)
    eng_exact = expect(hami, zero_state(nbit) |> bigc) |> real
    @show eng, eng_exact
    @test isapprox(eng, eng_exact, rtol=0.2)
end
