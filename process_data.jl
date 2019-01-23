using DelimitedFiles, JLD2, FileIO

const nbit = 16
const VER = :(su2_d4)
V = 5

filename(k::Int) = "data/_chem_$(VER)_N$(nbit)_V$(V)_S$(k).jld2"
function load_training(filename)
    res = load(filename)
    res["qopt"], res["loss"], res["params"], res["fidelity"]
end

opt, history, params, fidelities = load_training(filename(500))

for (token, var) in [("loss", history),
                        ("params", params),
                        ("fidelity", fidelities),
                        ]
    writedlm("data/_chem_j1j2_$(VER)_$(token)_N$(nbit)_V$V.dat", var)
end
