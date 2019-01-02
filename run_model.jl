push!(LOAD_PATH, abspath("src"))
using Yao
using QMPS
using DelimitedFiles

chem = model(Val(:su2); nbit=21, VER=4, V=5, B=1024)
dispatch!(chem.circuit, :random)
@time energy_exact(chem)
@time energy(chem)
#include("sampled_energy.jl")

run_corr(chem; VER=:su2)
#run_train(chem, heisenberg_ground_state(nbit-1); VER=:su2)
