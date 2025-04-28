#[cfg(feature = "py")]
#[cfg(test)]
pub mod test {
    use nb_macro::neuron_builder;


    // check if getter setters work and if iterate and spike works

    neuron_builder!(r#"
    [neuron]
        type: BasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = -(v - e) + i
    [end]
    "#);
}
