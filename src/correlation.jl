export measure_corr

"""
measure correlator.

e.g. measure_corr(chem, 1=>X, 3=>X) will measure <σₓ¹σₓ³> from a quantum MPS.
"""
function measure_corr(chem::QuantumMPS, si::Pair{Int, <:PauliGate}, sj::Pair{Int, <:PauliGate})
    si.first > sj.first && return measure_corr(chem, sj, si)
    si.first == sj.first && throw(ArgumentError("Qubit address conflict error!"))
    T = datatype(chem.initial_reg)

    input_state = chem.input_state
    reg = chem.initial_reg |> copy
    nv = chem.nbit_virtual + chem.nbit_ancilla
    nrep = nrepeat(chem)

    for i = nrep+1:nrep+nv
        input_state[i] == 1 && apply!(reg, put(nv+1, (i-nrep+1)=>XGate{T}()))
    end
    local res = nothing
    input_state[1] == 1 && apply!(reg, put(nv+1, 1=>XGate{T}()))
    for i=1:nrep
        reg |> getblock(chem, i)
        if i!=nrep
            res_i = _measure!(reg, i, [si, sj], input_state, true)
            if res_i != nothing
                if res != nothing
                    return mean(res.*res_i)
                else
                    res = res_i
                end
            end
        end
    end
    for i=1:nv+1-chem.nbit_ancilla
        res_i = _measure!(reg, i+nrep-1, [si, sj], input_state, false)
        if res_i != nothing
            if res != nothing
                return mean(res.*res_i)
            else
                res = res_i
            end
        end
    end
    throw()
end

function _measure!(reg::AbstractRegister{B, T}, i, sl, input_state, reset) where {B, T}
    for si in sl
        if i==si.first
            op_i = eigen!(si.second |> mat |>Matrix)
            reg |> put(nqubits(reg), 1=>matblock(T.(op_i.vectors' |> Matrix)))
            return @inbounds 1 .- 2 .* (reset ? measure_collapseto!(reg, 1; config=input_state[i+1]) : measure_remove!(reg, 1))
        end
    end
    reset ? measure_collapseto!(reg, 1; config=input_state[i+1]) : measure_collapseto!(reg, 1)
    nothing
end
