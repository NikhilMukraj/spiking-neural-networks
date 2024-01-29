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

### Get Parameters

#### Get IF Parameters

#### Get Hodgkin Huxley Parameters

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
