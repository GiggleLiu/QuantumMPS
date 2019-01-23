function su2_unit(::Type{T}, nbit::Int, i::Int, j::Int) where T
    put(nbit, (i,j)=>rot(SWAPGate{T}(), 0.0))
end

function su2_circuit(::Type{T}, nbit_virtual::Int, nlayer::Int, nrepeat::Int, pairs::Vector) where T
    circuit = sequence()
    nbit_used = 1 + nbit_virtual
    for i=1:nrepeat
        unit = chain(T, nbit_used)
        if i==1
            for j=2-(nrepeat%2):2:nbit_virtual-1
                push!(unit, singlet_block(T, nbit_used, j, j+1))
            end
        end
        if i%2 != nrepeat%2
            push!(unit, singlet_block(T, nbit_used, 1, nbit_used))
        else
            if i!=1
                push!(unit, swap(nbit_used, T, 1, nbit_used))   # fix swap parameter order!
            end
        end
        for j=1:nlayer
            nring = nbit_virtual
            nring <= 1 && continue
            ops = [su2_unit(T, nbit_used, i, j) for (i,j) in pairs]
            push!(unit, chain(nbit_used, ops))
        end
        push!(circuit, unit)
    end
    dispatch!(circuit, :random)
end

function singlet_block(::Type{T}, nbit::Int, i::Int, j::Int) where T
    unit = chain(nbit)
    push!(unit, put(nbit, i=>chain(XGate{T}(), HGate{T}())))
    push!(unit, control(nbit, -i, j=>XGate{T}()))
end

function model(::Val{:su2}, ::Type{T}; nbit, V, B=4096, nlayer=5, pairs) where T
    nrepeat = nbit - V
    c = su2_circuit(T, V+1, nlayer, nrepeat, pairs) |> autodiff(:QC)
    chem = QuantumMPS(1, V+1, 1, c, zero_state(T, V+2, B), zeros(Int, nbit+1))
    chem
end
