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

### Run Static Input

#### Static Input Integrate and Fire

#### Static Input Hodgkin Huxley

### Run Coupled Test

#### Coupled Integrate and Fire

#### Coupled Hodgkin Huxley

### Run STDP Test

### Run Lattice Simulation

### Run Lattice Fit
