export J1J2

"""
    J1J2{D} <: AbstractHeisenberg{D}

frustrated Heisenberg model.
"""
struct J1J2{D} <: AbstractHeisenberg{D}
    size::NTuple{D, Int}
    periodic::Bool
    J2::Float64
    J1J2(size::Int...; J2::Real, periodic::Bool) = new{length(size)}(size, periodic, Float64(J2))
end

Base.size(model::J1J2) = model.size

function hamiltonian(model::J1J2)
    nbit = nspin(model)
    sum(x->x[3]*heisenberg_ij(nbit, x[1], x[2]), get_bonds(model))*0.25
end

@inline function get_site(ij, mn, pbc::Val{true})
    Tuple(mod(i-1,m)+1 for (i,m) in zip(ij, mn))
end

@inline function get_site(ij, mn, pbc::Val{false})
    Tuple(i<=m ? i : 0 for (i,m) in zip(ij, mn))
end

function get_bonds(model::J1J2{2})
    m, n = model.size
    cis = LinearIndices(model.size)
    bonds = Tuple{Int, Int, Float64}[]
    for i=1:m, j=1:n
        for (_i, _j) in [(i+1, j), (i, j+1)]
            sites = get_site((_i, _j), (m, n), Val(model.periodic))
            if all(sites .> 0)
                push!(bonds, (cis[i,j], cis[sites...], 1.0))
            end
        end
        for (_i, _j) in [(i-1, j-1), (i-1, j+1)]
            sites = get_site((_i, _j), (m, n), Val(model.periodic))
            if all(sites .> 0)
                push!(bonds, (cis[i,j], cis[sites...], model.J2))
            end
        end
    end
    bonds
end

function get_bonds(model::J1J2{1})
    nbit, = model.size
    vcat([(i, i%nbit+1, 1.0) for i in 1:(model.periodic ? nbit : nbit-1)], [(i, (i+1)%nbit+1, model.J2) for i in 1:(model.periodic ? nbit : nbit-2)])
end
