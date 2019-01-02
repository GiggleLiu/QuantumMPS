push!(LOAD_PATH, abspath("src"))
using Yao, Yao.Blocks
using LinearAlgebra, Statistics
using QMPS
using Test, Random

@testset "energy-goodqn" begin
    Random.seed!(9)
    nbit = 8
    inputs = [i%2 for i=1:nbit]
    chem = chem_sample(:random, nbit, nbatch=4096, nbit_virtual=4, input_state=inputs)
    println("Number of parameters is ", chem.circuit |> nparameters)
    circuit = chem2circuit(chem)
    eng = energy(chem)
    hami = reduce(+, heisenberg_ij(nbit, i) for i=1:nbit-1)
    eng_exact = expect(hami, product_state(nbit, inputs |> Yao.Intrinsics.packbits) |> circuit) |> real
    @test isapprox(eng, eng_exact, rtol=0.2)
end

@testset "measure check" begin
    Random.seed!(5)
    chem = chem_sample(:random, 9; nlayer=2, nbatch=10000, nbit_virtual=4)
    circuit = chem2circuit(chem)

    for (i, j) in [(3,5), (5,3), (3,7), (7,3), (6,8), (8,6)]
        mean35 = expect(heisenberg_ij(nqubits(circuit), i, j), zero_state(nqubits(circuit)) |> circuit) |> real
        eng = measure_op2(chem, heisenberg_term, i, j)
        @test isapprox(mean35, eng, rtol=0.1)
    end
end

@testset "expect" begin
    reg = rand_state(3,10)
    e1 = expect(put(2, 2=>X), reg |> copy |> focus!(1,2) |> ρ)
    e2 = expect(put(2, 2=>X), reg |> copy |> focus!(1,2))
    e3 = expect(put(3, 2=>X), reg |> ρ)
    e4 = expect(put(3, 2=>X), reg)
    @test e1 ≈ e2
    @test e1 ≈ e3
    @test e1 ≈ e4
end

@testset "energy" begin
    Random.seed!(2)
    chem = chem_sample(:random, 9; nlayer=2, nbatch=10000)
    circuit = chem2circuit(chem)
    eng = energy(chem)
    hami = reduce(+, heisenberg_ij(9, i) for i=1:8)
    eng_exact = expect(hami, zero_state(9) |> circuit) |> real
    @test isapprox(eng, eng_exact, rtol=0.2)
end

# make it cluster state
@testset "convert wave function check" begin
    chem = chem_sample(:random, 9; nlayer=2, nbatch=10, nbit_virtual=4)
    c = random_circuit(1, 4, 2, 5, pair_ring(5))
    circuit = chem2circuit(chem)
    @test zero_state(nqubits(circuit)) |> circuit |> statevec |> length == 2^9
end
