push!(LOAD_PATH, abspath("src"))
using Yao
using LinearAlgebra, Statistics
using BitBasis: packbits
using QMPS
using Test, Random

# make it cluster state
@testset "convert wave function check" begin
    chem = model(:su2; nbit=9, nlayer=2, B=10, V=5, pairs=pair_ring(5))
    c = random_circuit(1, 4, 2, 5, pair_ring(5))
    circuit = expand_circuit(chem)
    @test zero_state(nqubits(circuit)) |> circuit |> statevec |> length == 2^10
end

@testset "measure check" begin
    Random.seed!(5)
    chem = model(:su2; nbit=9, nlayer=2, B=10000, V=5, pairs=pair_ring(5))
    circuit = expand_circuit(chem)

    for (i, j) in [(3,5), (5,3), (3,7), (7,3), (6,8), (8,6)]
        @show (i,j)
        mean35 = expect(heisenberg_ij(nqubits(circuit), i, j), zero_state(nqubits(circuit)) |> circuit) |> real
        eng = sum(g->measure_corr(chem, i=>g, j=>g), [X, Y, Z])
        @test isapprox(mean35, eng, rtol=0.4)
    end
end

@testset "j1j2" begin
    j1j2 = J1J2(4; periodic=false, J2=0.5)
    @test get_bonds(j1j2) == [(1, 2, 1.0),(2, 3, 1.0), (3, 4, 1.0), (1,3, 0.5), (2,4, 0.5)]
    j1j2 = J1J2(4; periodic=true, J2=0.5)
    @test get_bonds(j1j2) == [(1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 1, 1.0), (1,3, 0.5), (2,4, 0.5), (3,1, 0.5), (4,2, 0.5)]

    j1j2 = J1J2(3, 3; periodic=false, J2=0.5)
    prs = [1=>2, 1=>4, 2=>3, 2=>5, 2=>4, 3=>6, 3=>5, 4=>5, 4=>7, 5=>6, 5=>8, 5=>1, 5=>7, 6=>9, 6=>2, 6=>8, 7=>8, 8=>9, 8=>4, 9=>5]
    vs =  [1.0,   1,    1,    1,   0.5,  1.0,   0.5,  1.0,  1.0,  1.0, 1.0,  0.5,   0.5,  1.0,  0.5, 0.5,   1.0, 1.0,  0.5,  0.5]
    tps = [(i,j,v) for ((i,j), v) in zip(prs, vs)]
    @test sort(get_bonds(j1j2)) == sort(tps)
    j1j2 = J1J2(3, 3; periodic=true, J2=0.5)
    @test sort(get_bonds(j1j2)) == sort([(1,2,1.0), (1,4,1.0), (1,9,0.5), (1,6,0.5), (2,3,1.0), (2,5,1.0), (2,7,0.5), (2,4,0.5), (3,1,1.0), (3,6,1.0), (3,8,0.5), (3,5,0.5),
        (4,5,1.0), (4,7,1.0), (4,3,0.5), (4,9,0.5), (5,6,1.0), (5,8,1.0), (5,1,0.5), (5,7,0.5), (6,4,1.0), (6,9,1.0), (6,2,0.5), (6,8,0.5), (7,8,1.0), (7,1,1.0), (7,6,0.5), (7,3,0.5),
        (8,9,1.0), (8,2,1.0), (8,4,0.5), (8,1,0.5), (9,7,1.0), (9,3,1.0), (9,5,0.5), (9,2,0.5)])
end

@testset "energy-goodqn" begin
    Random.seed!(2)
    for hei in [Heisenberg(10; periodic=false), Heisenberg(3, 3; periodic=false), J1J2(3,3; J2=0.5, periodic=false)]
        nbit = nspin(hei)
        for xmodel in [:u1, :su2]
            @show xmodel
            pairs = pair_ring(xmodel==:su2 ? 4 : 5)
            chem = model(:general; nbit=nbit, B=10000, V=4, pairs=pairs)
            println("Number of parameters is ", chem.circuit |> nparameters)
            circuit = expand_circuit(chem)
            eng = energy(chem, hei)
            hami = hamiltonian(hei)
            eng_exact = expect(hami, product_state(nbit, chem.input_state |> packbits) |> circuit) |> real
            @test isapprox(eng, eng_exact, rtol=0.3)
        end
    end
end

@testset "energy-goodqn tfi" begin
    Random.seed!(11)
    for hei in [TFI(2; h=0.5, periodic=false)]
        nbit = nspin(hei)
        for xmodel in [:twoqubit]
            @show xmodel
            chem = model(xmodel; nbit=nbit, B=10000)
            println("Number of parameters is ", chem.circuit |> nparameters)
            circuit = expand_circuit(chem)
            eng = energy(chem, hei)
            hami = hamiltonian(hei)
            @show circuit
            eng_exact = expect(hami, product_state(nbit, chem.input_state |> packbits) |> circuit) |> real
            @show eng, eng_exact
            @test isapprox(eng, eng_exact, rtol=0.3)
        end
    end
end
