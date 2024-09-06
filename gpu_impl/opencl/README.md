# GPU README

## Todo

- [x] Try colwise sum on GPU
  - [x] Benchmark CPU v GPU
- [x] Then move to getting incoming connections on a basic graph
- [x] Then move to calculating electrical input values given basic graph, voltages, and gap conductance kernel
- [x] GPU leaky integrate and fire kernel
  - Test kernel alone and then lattice execution
- [ ] Then move to a more advanced graph with a seperate key set
- [ ] Benchmark calculation of inputs
- [ ] GPU iterate and spike kernels in crate (trait that returns the compiled kernel, not the string, with a `fn` or `lazy_static`) (should be integrated as optional feature)
  - [ ] Electrical synapses
  - [ ] Chemical synapses
    - Neurotransmitter concentraions represented as flattened 2D array, outermost array corresponds to what neuron it is input to, inner array represents the individual concentrations
    - Ligand gates have on and off flags to determine whether to iterate those gates
- [ ] GPU history tracking
- [ ] Executing simulation timestep multiple times over
  - If executing multiple kernels seperately takes too long, combine into single kernel that takes in a number of iterations to execute for
  - Add verbose option with a progress bar (and to CPU version)
- [ ] Integrating into main crate (as optional feature)
  - To gpu lattice function, eventually a conversion back when all features added, remove gaussian params from neuron model
- [ ] GPU spike train iteration
- [ ] GPU plasticity
  - Plasticity should be updated by first updating every incoming connection for neurons to update, then outgoing connections
- [ ] Reward modulation interfacing
  - Specialized readout layer to read for reward calculation that sends data back to the CPU
  - Similarly, a specialized input layer (probably for Poisson neurons), needs to be generated
- [ ] Automatic GPU and CPU code generation given the differential equations

## Notes

- Note that using `cl_bool` resulted in incorrect computation, switching to `cl_uint` fixed this
