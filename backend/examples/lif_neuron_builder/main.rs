#[cfg(feature = "neuron_builder")]
use spiking_neural_networks::neuron::iterate_and_spike_traits::neuron_builder;


#[cfg(feature = "neuron_builder")]
neuron_builder!(r#"
[neuron]
    type: BasicIntegrateAndFire
    vars: e = 0, v_reset = -75, v_th = -55
    on_spike: 
        v = v_reset
    spike_detection: v >= v_th
    on_iteration:
        dv/dt = (v - e) + i
[end]
"#);

fn main() {

}