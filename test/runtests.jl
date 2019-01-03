push!(LOAD_PATH, abspath("src"))
using Yao, Yao.Blocks
using LinearAlgebra, Statistics
using QMPS
using Test, Random

@testset "energy-goodqn" begin
    Random.seed!(2)
    nbit = 10
    for xmodel in [:random, :u1, :su2]
        chem = model(:random, ComplexF32; nbit=nbit, B=4096, V=4)
        println("Number of parameters is ", chem.circuit |> nparameters)
        circuit = chem2circuit(chem)
        eng = heisenberg_energy(chem)
        hami = reduce(+, heisenberg_ij(nbit, i) for i=1:nbit-1)*0.25
        eng_exact = expect(hami, product_state(nbit, chem.input_state |> Yao.Intrinsics.packbits) |> circuit) |> real
        @test isapprox(eng, eng_exact, rtol=0.3)
    end
end

@testset "energy" begin
    Random.seed!(2)
    chem = model(:random, ComplexF32; nbit=9, V=4, nlayer=2, B=10000)
    circuit = chem2circuit(chem)
    eng = heisenberg_energy(chem)
    hami = reduce(+, heisenberg_ij(9, i) for i=1:8)*0.25
    eng_exact = expect(hami, zero_state(9) |> circuit) |> real
    @test isapprox(eng, eng_exact, rtol=0.2)
end

# make it cluster state
@testset "convert wave function check" begin
    chem = model(:random, ComplexF32; nbit=9, nlayer=2, B=10, V=4)
    c = random_circuit(ComplexF32, 1, 4, 2, 5, pair_ring(5))
    circuit = chem2circuit(chem)
    @test zero_state(nqubits(circuit)) |> circuit |> statevec |> length == 2^9
end

@testset "measure check" begin
    Random.seed!(7)
    chem = model(:random, ComplexF64; nbit=9, nlayer=2, B=10000, V=4)
    circuit = chem2circuit(chem)

    for (i, j) in [(3,5), (5,3), (3,7), (7,3), (6,8), (8,6)]
        mean35 = expect(heisenberg_ij(nqubits(circuit), i, j), zero_state(nqubits(circuit)) |> circuit) |> real
        eng = sum(g->measure_corr(chem, i=>g, j=>g), [X, Y, Z])
        @test isapprox(mean35, eng, rtol=0.2)
    end
end


