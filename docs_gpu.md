# CUDA Documentation

## Izhikevich Neuron

### Iterate

```c
void iterate(
    float *input
    float *voltage,
    float *w,
    float *dv,
    float *dw,
    float *alpha,
    float *beta,
    float *c,
    float *d,
    float *dt,
    float *tau_m,
    float *v_th,
    int *is_spiking,
    int n
)
```

Iterates the given Izhikevich neurons with their specified parameters one time based on a seris of input voltages

- `input: *float` : Input voltages (mV)
- `voltage: *float` : Current voltage of the given Izhikevich neuron (mV)
- `w: *float` : Current adaptive value of the given Izhikevich neuron
- `dv: *float` : Array to write changes in voltage to
- `dw: *float` : Array to write changes in adaptive value to
- `alpha: *float` : Array of arbitrary $\alpha$ values that control speed in `Izhikevich` mode
- `beta: *float` : Array of arbitrary $\beta$ values that control sensitivity to `w` in `Izhikevich` mode
- `c: *float` : Array of after spike reset values for voltage in `Izhikevich` mode
- `d: *float` : Array of after spike reset values for `w` in `Izhikevich` mode
- `dt: float` : Simulation timestep value (ms)
- `tau_m: *float` : Neuronal ${\tau}_{m}$ value
- `v_th: *float` : Voltage threshold (mV)
- `is_spiking: *int` : Whether or not the given neuron is spiking
- `n: int` : Size of the neuron array

### Lattice

### Calculate Inputs

### Iterate Lattice
