# TODO

- [x] Try colwise sum on GPU
  - [x] Benchmark CPU v GPU
- [ ] Then move to getting incoming connections on a basic graph
- [ ] Then move to calculating input values given basic graph, voltages, and gap conductance kernel
- [ ] Then move to a more advanced graph with a seperate key set
- [ ] Benchmark calculation of inputs
- [ ] GPU iterate and spike kernels (trait that returns the compiled kernel, not the string, with a `fn` or `lazy_static`)
  - [ ] Electrical synapses
  - [ ] Chemical synapses
- [ ] GPU history tracking
- [ ] GPU spike train iteration
- [ ] GPU plasticity
- [ ] Reward modulation interfacing
  - Specialized readout layer to read for reward calculation that sends data back to the CPU
  - Similarly, a specialized input layer (probably for Poisson neurons), needs to be generated
- [ ] Automatic GPU and CPU code generation given the differential equations