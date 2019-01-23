push!(LOAD_PATH, abspath("src"))
using Yao
using QMPS
using DelimitedFiles

function szsz_correlation(ground_state::AbstractRegister)
    nbit = nqubits(ground_state)
    om = zeros(Float64, nbit, nbit)
    for i =1:nbit, j=1:nbit
        if i!=j
            om[i,j] = expect(put(nbit, i=>Z)*put(nbit, j=>Z), ground_state) |> real
            println("<σz($i)σz($j)> = $(om[i,j])")
        end
    end

    for (token, var) in [
                         ("om", om),
                        ]
        writedlm("data/_chem_j1j2_exact_$(token)_N$(nbit).dat", var)
    end
end

# load the model
#heis = Heisenberg(4, 4; periodic=false)
heis = J1J2(4, 4; periodic=false, J2=0.5)
using KrylovKit
m = mat(hamiltonian(heis))
E, V = eigsolve(m, 1, :SR)
gs = register(V[1])
szsz_correlation(gs)
