# Documentation

## IF Cell Model

### IF Mode

- `Basic`: An unmodified leaky integrate and fire
  - Only modifies voltage over time
- `Adaptive`: An adaptive leaky integrate and fire
  - Modifies voltage and a recovery variable over time
- `Adaptive Exponential`: An adaptive exponential integrate and fire
  - Modifies voltage and a recovery variable over time, the recovery variable is modified with an exponential function
- `Izhikevich`: An adaptive quadratic integrate and fire, or Izhikevich neuron
  - Modifies voltage and a recovery variable over time, voltage is changed with a quadratic function, can be parameterized to model tonic and bursting firing modes
- `Izhikevich Leaky`: A hybrid adaptive quadratic integrate and fire, or Izhikevich neuron with a leak
  - Modifies voltage and a recovery variable over time, voltage is changed with a quadratic function and takes into account leak, can be parameterized to model tonic and bursting firing modes

### IF Parameters

### Basic

- `get_dv_change_and_spike(i: float) -> (f64, bool)`: Returns the change in voltage and whether the neuron spikes when in `Basic` mode
  - `i: float`: Input voltage

### Adaptive

- `adaptive_get_dv_change(i: float) -> f64`: Returns the change in voltage in `Adaptive` mode
  - `i: float`: Input voltage
- `apply_dw_change_and_get_spike() -> bool`: Returns whether the neuron spikes when in `Adaptive` mode

### Adaptive Exponentational

- `exp_adaptive_get_dv_change(i: float) -> f64`: Returns the change in voltage in `Adaptive Exponentational` mode
  - `i: float`: Input voltage
- `apply_dw_change_and_get_spike() -> bool`: Returns whether the neuron spikes when in `Adaptive Exponentational` mode

### Izhikevich

- `izhikevich_get_dv_change(i: float) -> f64`: Returns the change in voltage in `Izhikevich` mode
  - `i: float`: Input voltage
- `izhikevich_apply_dw_and_get_spike() -> bool`: Returns whether the neuron spikes when in `Izhikevich` mode

### Izhikevich Leaky

- `izhikevich_leaky_get_dv_change(i: float) -> f64`: Returns the change in voltage in `Izhikevich Leaky` mode
  - `i: float`: Input voltage
- `izhikevich_apply_dw_and_get_spike() -> bool`: Returns whether the neuron spikes when in `Izhikevich Leaky` mode

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
