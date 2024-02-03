# Rust Documentation

## Integrate and Fire Neuron

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

## Hodgkin Huxley Neuron

### Gates

### Ligand Gated Channels

## TOML Methods

Can be run with:

```bash
cargo run --release filename.toml
```

### Parameters

#### IF Parameters

- `v_th: Float=-55.`: Spike threshold (mV)
- `v_reset: Float=-75.`: Reset potential (mV)
- `tau_m: Float=10.`: Membrane time constant (ms)
- `g_l: Float=10.`: Leak conductance (nS)
- `v_init: Float=-75.`: Initial potential (mV)
- `e_l: Float=-75.`: Leak reversal potential (mV)
- `tref: Float=10.`: Refractory time (ms), could rename to refract_time
- `w_init: Float=0.`: Initial w value
- `alpha_init: Float=6.`: Arbitrary $\alpha$ value for `Izhikevich` mode
- `beta_init: Float=10.`: Arbitrary $\beta$ value for `Izhikevich` mode
- `d_init: Float=2.`: After spike reset value for `w` in `Izhikevich` mode
- `dt: Float=0.1`: Simulation timestep (ms)
- `exp_dt: Float=1.`: Exponential time step (ms) for `Adaptive Exponenial` mode
- `a_minus: float`: STDP parameter for scaling weight if postsynaptic neuron fires first
- `a_plus: float`: STDP parameters for scaling weight if presynaptic neuron fires first
- `tau_minus: float`: stdp parameters for decay if postsynaptic neuron fires first
- `tau_plus: float`: stdp parameters for decay if presynaptic neuron fires first
- `stdp_weight_mean: float=1.`: Mean when initializing weight
- `stdp_weight_std: float=0.`: Standard deviation when initializing weight
- `stdp_weight_max: float=2.`: Maximum when initializing weight
- `stdp_weight_min: float=0.`: Minimum when initializing weight
- `last_firing_time: Optional[int]`: Last time step that spike occurred
- `bayesian_mean: Float=1.`: Mean when applying noise
- `bayesian_std: Float=0.`: Standard deviation when applying noise
- `bayesian_max: Float=2.`: Maximum noise
- `bayesian_min: Float=0.`: Minimum noise

#### Hodgkin Huxley Parameters

- `current_voltage: Float=0.` : Current membrane potential (mV)
- `input_resistance: Float=1.` : Resistance value on input voltage
- `dt: Float=0.1` : Simulation timestep (ms)
- `cm: Float=1.` : Capacitance
- `e_na: Float=115.` : Channel sodium
- `e_k: Float=-12.` : Channel potassium
- `e_k_leak: Float=10.6` : Channel potassium leak
- `g_na: Float=120.` : Sodium conductance
- `g_k: Float=36.` : Potassium conductance
- `g_k_leak: Float=0.3` : Potassium leak condutance
- `bayesian_mean: Float=1.`: Mean when applying noise
- `bayesian_std: Float=0.`: Standard deviation when applying noise
- `bayesian_max: Float=2.`: Maximum noise
- `bayesian_min: Float=0.`: Minimum noise

### Gates Parameters

- `state: Float=0.` : Arbitrary gate parameter
- `alpha: Float=0.` : Arbitrary gate parameter
- `beta: Float=0.` : Arbitrary gate parameter

### Ligand Gated Channels Parameters

- `AMPA: Boolean=false` : Ligand gated channel fitted to AMPA channels
- `GABAa: Boolean=false` : Ligand gated channel fitted to GABAa channels
<!-- - `NMDA` -->
- **todo create custom ligand gated channel from toml**

### Run Static Input

#### Static Input Integrate and Fire

Runs a single neuron with a static voltage input

- `if_type: String` : Type of integrate and fire neuron
- `input_voltage: Float` : Input voltage to neuron
- `iterations: String` : Amount of iterations to do
- `filename: String` : What to name output file

Example with non Izhikevich Type:

```toml
[single_neuron_test]
if_type = "adaptive"
input_voltage = 300.0
iterations = 4000
bayesian_std = 0.2
filename = "adaptive.txt"
```

Example with Izhikevich Type:

```toml
[single_neuron_test]
if_type = "adaptive quadratic"
input_voltage = 40.0
dt = 0.5
alpha_init = 0.01
beta_init = 0.25 
v_reset = -55.0
d_init = 8.0
iterations = 4000
filename = "aqif.txt"
```

#### Static Input Hodgkin Huxley

Runs a single Hodgkin Huxley model with a static input

- `iterations: Float` : Amount of iterations to do
- `filename: String` : What to name output file
- `input_current: Float` : Current to input into the model
  - **todo convert this to voltage based on internal resistance**
- `full: Boolean=false` : Whether to write state of gates to file along with voltage

Example:

```toml
[hodgkin_huxley]
iterations = 10000
filename = "hodgkin_huxley_test.txt"
g_na = 120.0
g_k = 36.0
e_k = -12.0
dt = 0.01
input_current = 50.0
bayesian_std = 0.1
```

### Run Coupled Test

#### Coupled Integrate and Fire

Runs two integrate and fire neurons, one presynaptic connected to a postsynaptic neuron for a given amount of iterations

- Prefix any presynaptic neuron arguments with `pre_` and any postsynaptic neuron arguments with `post_` except for the `if_type` as both must be of the same type
- `input_voltage: Float` : Input voltage to presynaptic neuron
- `filename: String` : What to name output file
- `input_equation: String` : What equation to use to modify input voltage into next neuron
  - Defaults to `"(sign * mp + 65) / 15."` if `if_type` is `Izhikevich` or `Izhikevich Leaky`
  - Defaults to `"sign * mp + 100 + rd * (nc^2 * 200)"` if `if_type` is not `Izhikevich` or `Izhikevich Leaky`

Example:

```toml
[coupled_test]
if_type = "izhikevich"
iterations = 10000
input_voltage = 25.0
pre_dt = 0.5
post_dt = 0.5
pre_alpha_init = 0.01
pre_beta_init = 0.25
pre_w_init = 20.0
pre_v_reset = -50.0
pre_d_init = 2.0
filename = "coupled tests/bursting_in_tonic_out.txt"
input_equation = "sign * mp + 65"
```

#### Coupled Hodgkin Huxley

Runs two Hodgkin Huxley neurons, one presynaptic connected to a postsynaptic neuron for a given amount of iterations

- Prefix any presynaptic neuron arguments with `pre_` and any postsynaptic neuron arguments with `post_`
- `input_voltage: Float` : Input voltage to presynaptic neuron neuron
- `iterations: String` : Amount of iterations to do
- `filename: String` : What to name output file
- `bayesian: Boolean=false` : Whether or not to add noise to inputs
- `full: Boolean=false` : Whether or not to write ligand gates states to file

Example:

```toml
[coupled_hodgkin_huxley]
post_AMPA = true
post_GABAa = true
# post_NMDA = true
iterations = 10000
filename = "coupled_hodgkin_huxley.csv"
bayesian = false
full = true
```

### Run STDP Test

Iterates an isolated postsynaptic integrate and fire neuron with a set amount of presynaptic neurons undergoing STDP

- `output_type: String` : One of the following formats to dump the simulation data into
  - `"Averaged Text"` : An average of all the voltages of each neuron per time step in plain text
  - `"Grid Text"` : A matrix of  all the voltages of each neuron per time step in plain text
  - `"Avergaed Binary"` : An average of all the voltages of each neuron per time step in binary format
  - `"Avergaed Binary"` : A matrix of all the voltages of each neuron per time step in binary format
- `input_equation=String` : What equation to use to modify input voltage into next neuron
  - Defaults to `"(sign * mp + 65) / 15."` if `if_type` is `Izhikevich` or `Izhikevich Leaky`
  - Defaults to `"sign * mp + 100 + rd * (nc^2 * 200)"` if `if_type` is not `Izhikevich` or `Izhikevich Leaky`
- `iterations: String` : Amount of iterations to do
- `input_voltage: Float` : Input voltage to neuron
- `n: Int` : Number of presynaptic input neurons (must be >=1)
- `filename: String` : What to name output file
- `if_type: String` : Type of integrate and fire neuron

Example:

```toml
[stdp_test]
if_type = "izhikevich"
iterations = 10000
stdp_weight_mean = 5.0
n = 1
input_voltage = 30.0
filename = "izhikevich_stdp.txt"
a_plus = 2.0
a_minus = 2.0
bayesian_std = 0.3
```

### Run R-STDP Test

- **todo**

### Run Lattice Simulation

Generates a lattice of randomly connected neurons that is simulated for a given amount of time steps

- `output_type: String` : One of the following formats to dump the simulation data into
  - `"Averaged Text"` : An average of all the voltages of each neuron per time step in plain text
  - `"Grid Text"` : A matrix of  all the voltages of each neuron per time step in plain text
  - `"Avergaed Binary"` : An average of all the voltages of each neuron per time step in binary format
  - `"Avergaed Binary"` : A matrix of all the voltages of each neuron per time step in binary format
- `tag=String` : What string to prefix the output files with
- `input_equation=String` : What equation to use to modify input voltage into next neuron
  - Defaults to `"(sign * mp + 65) / 15."` if `if_type` is `Izhikevich` or `Izhikevich Leaky`
  - Defaults to `"sign * mp + 100 + rd * (nc^2 * 200)"` if `if_type` is not `Izhikevich` or `Izhikevich Leaky`
  - Depends on the following variables:
    - `sign` : Whether the neuron is excitatory (`-1.`) or inhibitory (`1.`)
    - `mp` : Membrane potential voltage of the neuron
    - `rd` : Receptor density of the neuron
    - `nc` : Neurotransmitter concentration in synapse from input neuron
- `num_rows: Integer` : How many rows in the lattice (must be >=1)
- `num_cols: Integer` : How many rows in the lattice (must be >=1)
- `radius: Integer=1` : How far to connect possible neurons to (radius of surrounding square, must be >=1)
- `iterations: String` : Amount of iterations to do
- `if_type: String` : Type of integrate and fire neuron

Example with non Izhikevich Type:

```toml
[lattice_simulation]
num_rows = 20
num_cols = 20
dt = 0.1
iterations = 400
radius = 2
tag = "adaptive_exp"
output_type = "grid"
if_type = "adaptive exponential"
random_volt_initialization = true
```

Example with Izhikevich Type:

```toml
[lattice_simulation]
num_rows = 10
num_cols = 10
radius = 1
tag = "stdp_lattice"
output_type = "grid binary"
if_type = "izhikevich"
random_volt_initialization = true
do_stdp = true
weight_init = 1.0
a_plus = 0.5
a_minus = 0.5
iterations = 1000
input_equation = """
    (sign * mp + 65) / 15.
"""
```

<!-- ### Run Lattice Genetic Algorithm Fit -->
