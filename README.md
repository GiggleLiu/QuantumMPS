# QuantumMPS
Matrix product state (MPS) inspired quantum circuits for variational quantum eigensolver (VQE).

Physical models includes: 1D and 2D Heisenberg model, with and without frustration, OBC and PBC,
Circuit Block ansatz includes: General, U(1) and SU(2) symmetric ansatz.

## Setup Guide
Set up your julia environment

* [julia 1.0+](https://julialang.org/)
* install required julia libraries: `Yao`, `DelimitedFiles`, `FileIO`, `Fire`, `JLD2`, `KrylovKit` and `StatsBase`. e.g. to install the high performance variational quantum simulation package `Yao`,
one can open a julia REPL and type `]` to enter the `pkg` mode, keep your internet connected and type
```julia console
pkg> add Yao
```

To access GPU, you need the extra packages: `CUDAnative`, `CuArrays` and `CuYao`.
CuYao has not been registered yet, so 
```julia console
pkg> dev CuYao
```

Clone this repository [https://github.com/GiggleLiu/QuantumCircuitBornMachine.git](https://github.com/GiggleLiu/QuantumCircuitBornMachine.git) to your local host.

## Run an Example
As an example, we solve the ground state and get the ground state property of frustrated Heisenberg model with J2 = 0.5 on 4 x 4 lattice,
to run the training, one can type
```bash
$ julia j1j2.jl train su2      # train a SU2 symmetric model
```

The above training can be very very slow, please turn on your GPU if you have one, which can be achieved by setting `USE_CUDA = true` in file `applications.jl`.
With GPU acceleration, models can be trained in 5-48 hours.
With or without GPU, you can calculate the correlation function and energy using pre-trained data in `data/`.
```bash
$ julia j1j2.jl trained szsz su2     # Sz(i)*Sz(j) correlation matrix
$ julia j1j2.jl trained energy su2   # sample energy expectation value
$ julia j1j2.jl trained fidelity su2   # fidelity with respect to the exact ground state
$ julia j1j2.jl trained fidelity su2 --depth=1
```

## Documentations

* paper: Variational Quantum Eigensolver with Fewer Qubits ([pdf]()), [arXiv:xxxxxx](https://arxiv.org/abs/xxxxxx), Jin-Guo Liu, Yi-Hong Zhang, Yuan Wang and Lei Wang
* slides: [online]()

## Citation

If you use this code for your research, please cite our paper:

```
@article{Liu2019,
}
```

## Authors
* Jin-Guo Liu <cacate0129@iphy.ac.cn>
* Yi-Hong Zhang <yh-zhang17@mails.tsinghua.edu.cn>
