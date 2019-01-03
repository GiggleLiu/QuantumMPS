export heisenberg_ij, heisenberg, heisenberg_term, heisenberg_ground_state, heisenberg_energy, energy_exact

heisenberg_ij(nbit::Int, i::Int, j::Int=i+1) = put(nbit, i=>X)*put(nbit, j=>X) + put(nbit, i=>Y)*put(nbit, j=>Y) + put(nbit, i=>Z)*put(nbit, j=>Z)
const heisenberg_term = repeat(2, X, 1:2) + repeat(2, Y, 1:2) + repeat(2, Z, 1:2)

function heisenberg(nbit::Int; periodic::Bool=true)
    sx = i->put(nbit, i=>X)
    sy = i->put(nbit, i=>Y)
    sz = i->put(nbit, i=>Z)
    mapreduce(i->(j=i%nbit+1; sx(i)*sx(j)+sy(i)*sy(j)+sz(i)*sz(j)), +, 1:(periodic ? nbit : nbit-1))*0.25
end

function heisenberg_ground_state(nbit::Int)
    # get the ground state
    hami = heisenberg(nbit; periodic=false)
    E, v = eigsolve(mat(hami), 1, :SR)
    @show E[1]
    register(v[1])
end

function energy_exact(tc::QuantumMPS)
    nbit = nbit_simulated(tc)
    expect(heisenberg(nbit, periodic=false), state_exact(tc)) |> real
end

"""
Ground state energy for Heisenberg chain.
"""
function heisenberg_energy(chem::QuantumMPS)
    T = datatype(chem.initial_reg)
    heisenberg_energy(chem, XGate{T}()) + heisenberg_energy(chem, YGate{T}()) + heisenberg_energy(chem, ZGate{T}())
end

function heisenberg_energy(chem::QuantumMPS, pauli::PauliGate)
    input_state = chem.input_state
    reg = chem.initial_reg |> copy
    nv = chem.nbit_virtual
    nrep = nrepeat(chem)
    T = datatype(chem.initial_reg)

    op = eigen!(pauli |> mat |>Matrix)
    rotor = put(nv+1, 1=>matrixgate(T.(op.vectors' |> Matrix)))
    local eng = 0.0
    local res_pre
    local res
    for i = nrep+1:nrep+nv
        input_state[i] == 1 && apply!(reg, put(nv+1, (i-nrep+1)=>X))
    end
    input_state[1] == 1 && apply!(reg, put(nv+1, 1=>X))
    for i=1:nrep
        reg |> getblock(chem, i)
        if i!=nrep
            reg |> rotor
            @inbounds res = 1 .- 2 .* measure_reset!(reg, 1, val=input_state[i+1])
            i>1 && (eng += mean(res_pre.*res))
            res_pre = res
        end
    end
    for i=1:nv+1-chem.nbit_ancilla
        reg |> rotor
        @inbounds res = 1 .- 2 .* measure_remove!(reg, 1)
        eng += mean(res_pre.*res)
        res_pre = res
    end
    eng/4
end
