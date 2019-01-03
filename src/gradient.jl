export QMPSOptimizer, gradients_exact

struct QMPSOptimizer
    chem::QuantumMPS
    optimizer
    diff_blocks
    params::Vector
    QMPSOptimizer(chem::QuantumMPS, optimizer) = new(chem, optimizer, collect(chem.circuit, AbstractDiff), parameters(chem.circuit))
end

import Yao: gradient
# TODO: setiparameters! throw number of parameters mismatch error!
function gradient(chem::QuantumMPS, db::AbstractDiff)
    db.block.theta += π/2
    epos = heisenberg_energy(chem)
    db.block.theta -= π
    eneg = heisenberg_energy(chem)
    db.block.theta += π/2
    real(epos-eneg)/2
end

import Base: iterate
function iterate(qo::QMPSOptimizer, state::Int=1)
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
