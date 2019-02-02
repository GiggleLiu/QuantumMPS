# QuantumMPS
Matrix product state (MPS) inspired quantum circuits for variational quantum eigensolver (VQE).

Physical models includes: 1D and 2D Heisenberg model, with or without frustration, OBC or PBC,

Circuit Block ansatz includes: General, U(1) and SU(2) symmetric ansatz.

## Setup Guide
Set up your julia environment

* [julia 1.0+](https://julialang.org/)
* install required julia libraries: `Yao`, `DelimitedFiles`, `FileIO`, `Fire`, `JLD2`, `KrylovKit` and `StatsBase`. For example, to install `Yao`,
one can open a julia REPL and type `]` to enter the `pkg` mode, keep your internet connected and type
```julia console
pkg> add Yao
```

To access GPU, you need the extra packages: `CUDAnative`, `CuArrays` and `CuYao`.
Since `CuYao` has not been registered yet, please use
```julia console
pkg> dev CuYao
```
To install this CUDA extension for Yao.

Clone this repository [https://github.com/GiggleLiu/QuantumMPS.git](https://github.com/GiggleLiu/QuantumMPS.git) to your local host.

## Run an Example
As an example, we solve the ground state and get the ground state property of frustrated Heisenberg model with J2 = 0.5 on 4 x 4 lattice,
to run the training, one can type

```bash
$ julia j1j2.jl train --symmetry su2 --depth 1
```
Here, `symmetry` and `depth` are optional parameters to specify symmetry of ansatz and depth of circuit block.
The default symmetry is `su2` and the default circuit depth is 5.
The above training with default setting can be very very slow. Please turn on the GPU switch by setting `USE_CUDA = true` in file `applications.jl` if you have a GPU that supports CUDA. With Nvidia Titan V, training can be accelerated by a factor of ~35 comparing with the sequential CPU version, but still takes several hours. Decreasing the circuit depth can also accelerate the training.

With or without GPU, you can calculate the correlation function and energy per site using pre-trained data in `data/`
```bash
$ julia j1j2.jl measure szsz -- depth 1         # Sz(i)*Sz(j) correlation matrix, default depth is 5.
$ julia j1j2.jl measure energy --symmetry su2   # sample energy expectation value
```

To get help on input parameters, you can type
```bash
$ julia j1j2.jl train --help
$ julia j1j2.jl measure --help
```

## Documentations

* paper: Variational Quantum Eigensolver with Fewer Qubits ([pdf]()), [arXiv:xxxxxx](https://arxiv.org/abs/xxxxxx), Jin-Guo Liu, Yihong Zhang, Yuan Wang and Lei Wang
* slides: [online]()

## Citation

If you use this code for your research, please cite our paper:

```
<arXiv citation>
```

## Authors
See [contributors](https://github.com/GiggleLiu/QuantumMPS/graphs/contributors) page.

