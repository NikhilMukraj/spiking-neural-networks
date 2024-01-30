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
- `alpha_init: Float=6.`: Arbitrary a value for `Izhikevich` mode
- `beta_init: Float=10.`: Arbitrary b value for `Izhikevich` mode
- `d_init: Float=2.`: Arbitrary d value for `Izhikevich` mode
- `dt: Float=0.1`: Simulation timestep (ms)
- `exp_dt: Float=1.`: Exponential time step (ms) for `Adaptive Exponenial` mode
- `bayesian_mean: Float=1.`: Mean when applying noise
- `bayesian_std: Float=0.`: Standard deviation when applying noise
- `bayesian_max: Float=2.`: Maximum noise
- `bayesian_min: Float=0.`: Minimum noise

#### Hodgkin Huxley Parameters

- `current_voltage: float` : Current membrane potential
- `dt: float` : Simulation timestep (ms)
- `cm: float` : Capacitance
- `e_na: float` : Channel sodium
- `e_k: float` : Channel potassium
- `e_k_leak: float` : Channel potassium leak
- `g_na: float` : Sodium conductance
- `g_k: float` : Potassium conductance
- `g_k_leak: float` : Potassium leak condutance
- `bayesian_mean: float=1.`: Mean when applying noise
- `bayesian_std: float=0.`: Standard deviation when applying noise
- `bayesian_max: float=2.`: Maximum noise
- `bayesian_min: float=0.`: Minimum noise

### Gates Parameters

- `state: float` : Arbitrary gate parameter
- `alpha: float` : Arbitrary gate parameter
- `beta: float` : Arbitrary gate parameter

### Ligand Gated Channels Parameters

### Run Static Input

#### Static Input Integrate and Fire

- `if_type`: `String` - Type of integrate and fire neuron
- `input` : `Float` - Input voltage to neuron
- `iterations` : `String` - Amount of iterations to do
- `filename` : `String` - What to name output file

Example with non Izhikevich Type:

```toml
[single_neuron_test]
if_type = "adaptive"
input = 300.0
iterations = 4000
bayesian_std = 0.2
filename = "adaptive.txt"
```

Example with Izhikevich Type:

```toml
[single_neuron_test]
if_type = "adaptive quadratic"
input = 40.0
dt = 0.5
alpha_init = 0.01
beta_init = 0.25 
v_reset = -55.0
d_init = 8.0
iterations = 4000
filename = "aqif.txt"
```

#### Static Input Hodgkin Huxley

### Run Coupled Test

#### Coupled Integrate and Fire

#### Coupled Hodgkin Huxley

### Run STDP Test

### Run Lattice Simulation

Example with non Izhikevich Type:

```toml
# to be written
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

### Run Lattice Fit
