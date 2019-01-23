export QMPSOptimizer, gradients_exact

struct QMPSOptimizer
    chem::QuantumMPS
    model::AbstractModel
    optimizer
    diff_blocks
    params::Vector
    QMPSOptimizer(chem::QuantumMPS, model::AbstractModel, optimizer) = new(chem, model, optimizer, collect(chem.circuit, AbstractDiff), parameters(chem.circuit))
end

import Yao: gradient
# TODO: setiparameters! throw number of parameters mismatch error!
function gradient(chem::QuantumMPS, db::AbstractDiff, model::AbstractModel)
    db.block.theta += π/2
    epos = energy(chem, model)
    db.block.theta -= π
    eneg = energy(chem, model)
    db.block.theta += π/2
    real(epos-eneg)/2
end

import Base: iterate
function iterate(qo::QMPSOptimizer, state::Int=1)
    # initialize the parameters
    grad = gradient.(Ref(qo.chem), qo.diff_blocks, Ref(qo.model))
    update!(qo.params, grad, qo.optimizer)
    dispatch!(qo.chem.circuit, qo.params)
    (grad, state+1)
end

function gradients_exact(chem, hami; dbs=nothing)
    nbit = nbit_simulated(chem)
    circuit = chem2circuit(chem)
    if dbs == nothing
        dbs = collect(circuit, AbstractDiff)
    end
    opdiff.(()->state_exact(chem), dbs, Ref(hami))
end
