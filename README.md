# QuantumMPS
Matrix product state (MPS) inspired quantum circuits ansatz for variational quantum eigensolver (VQE).

Physical Hamiltonian includes: 1D and 2D Heisenberg model, with or without frustration, OBC or PBC.

Circuit Block includes: General, U(1) and SU(2) symmetric ansatz.

## Setup Guide

Clone this repository [https://github.com/GiggleLiu/QuantumMPS.git](https://github.com/GiggleLiu/QuantumMPS.git) to your local host.

Set up your julia environment

* [julia 1.0+](https://julialang.org/)
* install required julia libraries: `Yao`, `DelimitedFiles`, `FileIO`, `Fire`, `JLD2`, `KrylovKit` and `StatsBase`. To access GPU, you need the extra packages: `CUDAnative`, `CuArrays` and `CuYao`. They can be resolved by typing
```bash
$ julia resolve_env.jl # if a GPU is available
$ julia resolve_env.jl nocuda
```

## Run an Example
As an example, we solve the ground state of frustrated Heisenberg model with J2 = 0.5 on 4 x 4 lattice.
to run

```bash
$ julia j1j2.jl train --symmetry su2 --depth 1
```
Here, `symmetry` and `depth` are optional parameters to specify symmetry of ansatz and depth of circuit block.
The default symmetry is `su2` and the default circuit depth is 5.
The above training with default setting can be very very slow. With Nvidia Titan V GPU, training can be accelerated by a factor of ~35 comparing with the sequential CPU version, but still takes several hours. Decreasing the circuit depth can also accelerate the training.

You can meaure the correlation function and energy per site using pre-trained model stored in `data/`
```bash
$ julia j1j2.jl measure szsz -- depth 1         # Sz(i)*Sz(j) correlation matrix, default depth is 5.
$ julia j1j2.jl measure energy --symmetry su2   # sample energy expectation value
```

To get help on input parameters, please type
```bash
$ julia j1j2.jl train --help
$ julia j1j2.jl measure --help
```

## Documentations

* paper: Variational Quantum Eigensolver with Fewer Qubits ([pdf](https://arxiv.org/pdf/1902.02663.pdf)), [arXiv:1902.02663](https://arxiv.org/abs/1902.02663), Jin-Guo Liu, Yihong Zhang, Yuan Wan and Lei Wang

## Citation

You are welcome to use this code for your research. Please kindly cite:

```
@article{Liu2019,
  author = {Jin-Guo Liu, Yi-Hong Zhang, Yuan Wan and Lei Wang},
  title = {Variational Quantum Eigensolver with Fewer Qubits},
  eprint = {arXiv:1902.02663},
  url = {https://arxiv.org/abs/1902.02663}
}
```
