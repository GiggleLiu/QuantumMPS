struct J1J2{D} <: AbstractModel{D}
    size::NTuple{D, Int}
    periodic::Bool
    J1J2(size::Int...; periodic::Bool) = new{length(size)}(size, periodic)
end

Base.size(model::J1J2) = model.size

function hamiltonian(model::J1J2)
    nbit = nspin(model)
    sum(x->heisenberg_ij(nbit, x.first, x.second), get_bonds(model))*0.25
end

@inline function get_site(ij, mn, pbc::Val{true})
    Tuple((i-1)%m+1 for (i,m) in zip(ij, mn))
end

@inline function get_site(ij, mn, pbc::Val{true})
    Tuple(i<=m ? i : 0 for (i,m) in zip(ij, mn))
end

function get_bonds(model::J1J2{2})
    m, n = model.size
    cis = LinearIndices(model.size)
    bonds = Pair{Int, Int}[]
    for i=1:m, j=1:n
        for (_i, _j) in [(i+1, j), (i, j+1), (i-1, j+1), (i+1, j+1)]
            sites = get_site((_i, _j), (m, n), Val(model.periodic))
            if all(sites .> 0)
                push!(bonds, cis[i,j] => cis[sites...])
            end
        end
    end
    bonds
end

function get_bonds(model::J1J2{1})
    nbit, = model.size
    cat([i=>i%nbit+1 for i in 1:(model.periodic ? nbit : nbit-1)], [i=>i%(nbit+1)+1 for i in 1:(model.periodic ? nbit : nbit-2)])
end

using Test
@testset "j1j2" begin
    j1j2 = J1J2(4, periodic=false)
    @test get_bonds(j1j2) == [1=>2,2=>3,3=>4, 1=>3, 2=>4]
    j1j2 = J1J2(4, periodic=true)
    @test get_bonds(j1j2) == [1=>2,2=>3,3=>4, 4=>1, 1=>3, 2=>4, 3=>1, 4=>2]

    j1j2 = J1J2(3, 3, periodic=false)
    @test get_bonds(j1j2) == [1=>2, 1=>4, 2=>3, 2=>5, 2=>4, 3=>6, 3=>5, 4=>5, 4=>7, 5=>6, 5=>8, 5=>1, 5=>7, 6=>9, 6=>2, 6=>8, 7=>8, 8=>9, 8=>4, 9=>5]
    j1j2 = J1J2(3, 3, periodic=true)
    @test get_bonds(j1j2) == [1=>2, 1=>4, 1=>9, 1=>6, 2=>3, 2=>5, 2=>7, 2=>4, 3=>1, 3=>6, 3=>8, 3=>5, 4=>5, 4=>7, 4=>3, 4=>9, 5=>6, 5=>8, 5=>1, 5=>7, 6=>4, 6=>9, 6=>2, 6=>8, 7=>8, 7=>1, 7=>6, 7=>3, 8=>9, 8=>2, 8=>4, 8=>1, 9=>7, 9=>3, 9=>5, 9=>2]
end
