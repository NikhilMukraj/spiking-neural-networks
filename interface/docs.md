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

- `v_th: float=-55.`: spike threshold (mV)
- `v_reset: float=-75.`: reset potential (mV)
- `tau_m: float=10.`: membrane time constant (ms)
- `g_l: float=10.`: leak conductance (nS)
- `v_init: float=-75.`: initial potential (mV)
- `e_l: float=-75.`: leak reversal potential (mV)
- `tref: float=10.`: refractory time (ms), could rename to refract_time
- `w_init: float=0.`: initial w value
- `alpha_init: float=6.`: arbitrary a value for `Izhikevich` mode
- `beta_init: float=10.`: arbitrary b value for `Izhikevich` mode
- `d_init: float=2.`: arbitrary d value for `Izhikevich` mode
- `dt: float=0.1`: simulation time step (ms)
- `exp_dt: float=1.`: exponential time step (ms) for `Adaptive Exponenial` mode
- `bayesian_mean: float=1.`: Mean when applying noise
- `bayesian_std: float=0.`: Standard deviation when applying noise
- `bayesian_max: float=2.`: Maximum noise
- `bayesian_min: float=0.`: Minimum noise

### Cell Parameters

- `current_voltage: float`: Membrane potential
- `refractory_count: float`: Keeps track of refractory period
- `leak_constant: float`: Leak constant gene
- `integration_constant: float`: Integration constant gene
- `is_excitatory: bool`: Whether neuron is an excitatory (`true`) or inhibitory (`false`) potentiation type
- `neurotransmission_concentration: float`: Concentration of neurotransmitter in synapse
- `neurotransmission_release: float`: Concentration of neurotransmitter released at spiking
- `receptor_density: float`: Factor of how many receiving receptors for a given neurotransmitter
- `chance_of_releasing: float`: Chance cell can produce neurotransmitter
- `dissipation_rate: float`: How quickly neurotransmitter concentration decreases
- `chance_of_random_release: float`: Likelyhood of neuron randomly releasing neurotransmitter
- `random_release_concentration: float`: How much neurotransmitter is randomly released
- `w_value: float`: Adaptive value
- `a_minus: float`: STDP parameter for scaling weight if postsynaptic neuron fires first
- `a_plus: float`: STDP parameters for scaling weight if presynaptic neuron fires first
- `tau_minus: float`: stdp parameters for decay if postsynaptic neuron fires first
- `tau_plus: float`: stdp parameters for decay if presynaptic neuron fires first
- `stdp_weight_mean: float=1.`: Mean when initializing weight
- `stdp_weight_std: float=0.`: Standard deviation when initializing weight
- `stdp_weight_max: float=2.`: Maximum when initializing weight
- `stdp_weight_min: float=0.`: Minimum when initializing weight
- `last_firing_time: Optional[int]`: Last time step that spike occurred
- `alpha: float`: Arbitrary value that controls speed in `Izhikevich` mode
- `beta: float`: Arbitrary value that controls sensitivity to `w` in `Izhikevich` mode
- `c: float`: After spike reset value for voltage in `Izhikevich` mode
- `d: float`: After spike reset value for `w` in `Izhikevich` mode

### Basic

- `get_dv_change_and_spike(i: float) -> (float, bool)`: Returns the change in voltage and whether the neuron spikes when in `Basic` mode
  - `i: float`: Input voltage

### Adaptive

- `adaptive_get_dv_change(i: float) -> float`: Returns the change in voltage in `Adaptive` mode
  - `i: float`: Input voltage
- `apply_dw_change_and_get_spike() -> bool`: Returns whether the neuron spikes when in `Adaptive` mode

### Adaptive Exponentational

- `exp_adaptive_get_dv_change(i: float) -> float`: Returns the change in voltage in `Adaptive Exponentational` mode
  - `i: float`: Input voltage
- `apply_dw_change_and_get_spike() -> bool`: Returns whether the neuron spikes when in `Adaptive Exponentational` mode

### Izhikevich

- `izhikevich_get_dv_change(i: float) -> float`: Returns the change in voltage in `Izhikevich` mode
  - `i: float`: Input voltage
- `izhikevich_apply_dw_and_get_spike() -> bool`: Returns whether the neuron spikes when in `Izhikevich` mode

### Izhikevich Leaky

- `izhikevich_leaky_get_dv_change(i: float) -> float`: Returns the change in voltage in `Izhikevich Leaky` mode
  - `i: float`: Input voltage
- `izhikevich_apply_dw_and_get_spike() -> bool`: Returns whether the neuron spikes when in `Izhikevich Leaky` mode

### IF Cell

- `iterate_and_return_spike(i: float, bayesian: bool) -> bool` : Runs one iteration of the model and returns whether it spikes
  - `i: float` : Input voltage
  - `bayesian: bool=false` : Whether to add noise from `if_params`
- `run_static_input(i: float, iterations: int, bayesian: bool = false) -> (list[float], list[bool])` : Runs model for a selected number of iterations and returns how voltage changes and when the model spiked
  - `i: float` : Input voltage
  - `iterations: int` : Number of times to run model (must be positive)
  - `bayesian: bool=false` : Whether to add noise from `if_params`
- `determine_neurotransmitter_concentration(is_spiking: bool)` : Updates neurotransmitter concentration within synapse based on cellular parameters
  - `is_spiking: bool` : Whether the neuron is spiking

## IF Cell Related Methods

### Test Coupled Neurons

- `test_coupled_if_cells(pre_synaptic_neuron: IFCell, post_synaptic_neuron: IFCell, iterations: int, input_voltage: float, input_equation: str) -> list[(float, float)]` : Feeds the membrane potential of one neuron into the input of the next and returns the membrane potentials over time
  - `pre_synaptic_neuron: IFCell` : An `IFCell` that is used as an input
  - `post_synaptic_neuron: IFCell` : An `IFCell` that is used as an input
  - `iterations: int` : Number of times to run model (must be positive)
  - `input_voltage: float` : Input voltage
  - `input_equation: str` : Textual representation of an equation that modifies the voltage input into the post synaptic neuron

### Test STDP

### Test Lattice

## Hodgkin Huxley Model

### Gates

### Hodgkin Huxley

### Todo

- Iterate with neurotransmitter concentration consideration with IF Cell
- Static input with neurotransmitter concentration consideration with IF Cell
- Hodgkin Huxley interface
- Hodgkin Huxley coupled test
- Lattice interface
- Testing for all class methods, with test file
- Documentation for all of IF Cell
