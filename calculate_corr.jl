push!(LOAD_PATH, abspath("src"))
using Yao
using QMPS
using DelimitedFiles

function correlation(chem, op; VER)
    nbit = nbit_simulated(chem)
    V = chem.nbit_virtual
    if op == X
        tag = :X
    elseif op == Y
        tag = :Y
    elseif op == Z
        tag = :Z
    end

    om = zeros(Float64, nbit, nbit)
    for i =1:nbit, j=1:nbit
        if i!=j
            om[i,j] = measure_corr(chem, i=>op, j=>op) |> real
            println("<σz($i)σz($j)> = $(om[i,j])")
        end
    end

    for (token, var) in [
                         ("om", om),
                        ]
        writedlm("data/_chem_$(VER)_$(token)$(tag)_N$(nbit)_V$V.dat", var)
    end
end

const USE_CUDA = false
USE_CUDA && include("CuChem.jl")

# load the model
nbit = 16
V = 5
VER = :su2
pairs = pair_ring(VER==:su2 ? V : V+1)
heis = Heisenberg(4, 4; periodic=false)
chem = model(Val(VER), ComplexF32; nbit=nbit, V=V, B=20000, pairs=pairs)
USE_CUDA && (chem = chem |> cu)
filename = "data/chem_$(VER)_params_N$(nbit)_V$V.dat"
println("loading file $filename")
println("Number of parameters is ", chem.circuit |> nparameters)
params = readdlm(filename)
dispatch!(chem.circuit, params)

@time println("expectation value of energy/site (exact) = $(energy_exact(chem, heis)/nbit)")
exit()
@time println("expectation value of energy/site (sampled) = $(energy(chem, heis)/nbit)")

correlation(chem, Z; VER=:su2)
#run_train(chem, heisenberg_ground_state(nbit-1); VER=:su2)
