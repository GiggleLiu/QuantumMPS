using CUDAnative, GPUArrays, CuArrays
using CUDAnative: device!, devices, CuDevice
using CuYao
import CuYao: cu
device!(CuDevice(2))
CuArrays.allowscalar(false)

function cu(chem::QuantumMPS)
    QuantumMPS(chem.nbit_measure, chem.nbit_virtual, chem.nbit_ancilla, chem.circuit, chem.initial_reg |> cu, chem.input_state)
end
