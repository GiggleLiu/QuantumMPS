push!(LOAD_PATH, abspath("src"))
using Yao
using QMPS
using DelimitedFiles

include("CuChem.jl")

function szsz_correlation(chem; VER)
    nbit = nbit_simulated(chem)
    V = chem.nbit_virtual

    om = zeros(Float64, nbit, nbit)
    for i =1:nbit, j=1:nbit
        if i!=j
            om[i,j] = measure_corr(chem, i=>Z, j=>Z) |> real
            println("<σz($i)σz($j)> = $(om[i,j])")
        end
    end

    for (token, var) in [
                         ("om", om),
                        ]
        writedlm("data/_chem_$(VER)_$(token)_N$(nbit)_V$V.dat", var)
    end
end

# load the model
chem = model(Val(:su2); nbit=20, V=5, B=1024)
chem = chem |> cu
filename = "data/chem_su2_params_N20_V5.dat"
println("loading file $filename")
println("Number of parameters is ", chem.circuit |> nparameters)
params = readdlm(filename)
dispatch!(chem.circuit, params)

@time println("expectation value of energy/site (exact) = $(energy_exact(chem)/20)")
@time println("expectation value of energy/site (sampled) = $(heisenberg_energy(chem)/20)")

szsz_correlation(chem; VER=:su2)
#run_train(chem, heisenberg_ground_state(nbit-1); VER=:su2)
