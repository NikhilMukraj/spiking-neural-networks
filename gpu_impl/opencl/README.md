# TODO

- [x] Try colwise sum on GPU
  - [x] Benchmark CPU v GPU
- [ ] Then move to getting incoming connections on a basic graph
- [ ] Then move to calculating input values given basic graph, voltages, and gap conductance kernel
- [ ] Then move to a more advanced graph with a seperate key set
- [ ] Benchmark calculation of inputs
- [ ] GPU iterate and spike kernels
  - [ ] Electrical synapses
  - [ ] Chemical synapses
- [ ] GPU spike train iteration
- [ ] GPU history tracking
- [ ] GPU plasticity
- [ ] Reward modulation interfacing
  - Specialized readout layer to read for reward calculation that sends data back to the CPU
