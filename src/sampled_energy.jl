import Chemistry: energy
using Statistics, Yao

energy(chem::TNChem) = energy(chem, Val(:X)) + energy(chem, Val(:Y)) + energy(chem, Val(:Z))
function energy(chem::TNChem, which::Union{Val{:X}, Val{:Y}, Val{:Z}})
    input_state = chem.input_state
    reg = chem.initial_reg |> copy
    nv = chem.nbit_virtual
    nrep = nrepeat(chem)

    op = eigen!(_op(which) |> mat |>Matrix)
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
            res = measure_reset!(op, reg, 1, val=input_state[i+1])
            i>1 && (eng += mean(res_pre.*res))
            res_pre = res
        end
    end
    for i=1:nv+1-chem.nbit_ancilla
        res = measure_remove!(op, reg, 1)
        eng += mean(res_pre.*res)
        res_pre = res
    end
    eng
end

_op(::Val{:X}) = X
_op(::Val{:Y}) = Y
_op(::Val{:Z}) = Z
