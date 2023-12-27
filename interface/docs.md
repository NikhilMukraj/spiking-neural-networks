# Documentation

## IF Cell Model

### Basic

### Adaptive

### Adaptive Exponentational

### Izhikevich

### Izhikevich Leaky

### IF Mode

### IF Parameters

### IF Cell

- `iterate_and_return_spike(i: float, bayesian: bool) -> bool` : Runs one iteration of the model and returns whether it spikes
  - `i: float` : Input voltage
  - `bayesian: bool` : Whether to add noise from `if_params`
- `run_static_input(i: float, iterations: int, bayesian: bool = false) -> (list[float], list[bool])` : Runs model for a selected number of iterations and returns how voltage changes and when the model spiked
  - `i: float` : Input voltage
  - `iterations: int` : Number of times to run model (must be positive)
  - `bayesian: bool` : Whether to add noise from `if_params`
- `determine_neurotransmitter_concentration(is_spiking: bool)` : Updates neurotransmitter concentration within synapse based on cellular parameters
  - `is_spiking: bool` : Whether the neuron is spiking

## IF Cell Related Methods

### Test Coupled Neurons

### Test STDP

### Test Lattice

## Hodgkin Huxley Model

### Gates

### Hodgkin Huxley

### Todo

- Iterate with neurotransmitter concentration consideration with IF Cell
- Static input with neurotransmitter concentration consideration with IF Cell
- Hodgkin Huxley interface
- Lattice interface
- Testing for all class methods, with test file
- Documentation for all of IF Cell
