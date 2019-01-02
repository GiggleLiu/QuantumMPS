export heisenberg_ij, heisenberg, heisenberg_term

heisenberg_ij(nbit::Int, i::Int, j::Int=i+1) = put(nbit, i=>X)*put(nbit, j=>X) + put(nbit, i=>Y)*put(nbit, j=>Y) + put(nbit, i=>Z)*put(nbit, j=>Z)
const heisenberg_term = repeat(2, X, 1:2) + repeat(2, Y, 1:2) + repeat(2, Z, 1:2)

function heisenberg(nbit::Int; periodic::Bool=true)
    sx = i->put(nbit, i=>X)
    sy = i->put(nbit, i=>Y)
    sz = i->put(nbit, i=>Z)
    mapreduce(i->(j=i%nbit+1; sx(i)*sx(j)+sy(i)*sy(j)+sz(i)*sz(j)), +, 1:(periodic ? nbit : nbit-1))
end
