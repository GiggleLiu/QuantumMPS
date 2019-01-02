export TNChemOptimizer, gradients_exact

struct TNChemOptimizer
    chem::TNChem
    optimizer
    diff_blocks
    params::Vector
    TNChemOptimizer(chem::TNChem, optimizer) = new(chem, optimizer, collect(chem.circuit, AbstractDiff), parameters(chem.circuit))
end

import Yao: gradient
# TODO: setiparameters! throw number of parameters mismatch error!
function gradient(chem::TNChem, db::AbstractDiff)
    db.block.theta += π/2
    epos = energy(chem)
    db.block.theta -= π
    eneg = energy(chem)
    db.block.theta += π/2
    real(epos-eneg)/2
end

import Base: iterate
function iterate(qo::TNChemOptimizer, state::Int=1)
    # initialize the parameters
    grad = gradient.(Ref(qo.chem), qo.diff_blocks)
    update!(qo.params, grad, qo.optimizer)
    dispatch!(qo.chem.circuit, qo.params)
    (grad, state+1)
end

function gradients_exact(chem; dbs=nothing)
    nbit = nbit_simulated(chem)
    circuit = chem2circuit(chem)
    if dbs == nothing
        dbs = collect(circuit, AbstractDiff)
    end
    opdiff.(()->state_exact(chem), dbs, Ref(heisenberg(nbit-chem.nbit_ancilla, periodic=false)))
end
