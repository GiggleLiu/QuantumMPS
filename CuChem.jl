using CUDAnative, GPUArrays, CuArrays
using CuYao
import CuYao: cu
CuArrays.allowscalar(false)

function cu(chem::TNChem)
    ei = Eigen(chem.op_eigen.values|>cu, chem.op_eigen.vectors|>cu)
    TNChem(chem.nbit_measure, chem.nbit_virtual, chem.circuit, chem.initial_reg |> cu, ei, chem.input_state, chem.nbit_ancilla)
end
