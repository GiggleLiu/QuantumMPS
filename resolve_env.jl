using Pkg
deps = ["Yao", "DelimitedFiles", "FileIO", "Fire", "JLD2", "KrylovKit", "StatsBase"]
extras = ["CUDAnative", "CuArrays"]

USE_CUDA = !("nocuda" in ARGS)

if USE_CUDA
    deps = vcat(deps, extras)
end

for x in deps
    println("Installing $x ...")
    Pkg.add(x)
end

if USE_CUDA
    println("Installing CuYao ...")
    Pkg.clone("https://github.com/QuantumBFS/CuYao.jl")
end
