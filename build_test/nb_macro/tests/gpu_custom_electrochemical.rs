#[allow(clippy::assign_op_pattern)]
#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;


    neuron_builder!(r#"
    [neuron]
        type: ElectroChemicalIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, modifier = 1
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
        on_electrochemical_iteration:
            receptors.update_receptor_kinetics(t, dt)
            receptors.set_receptor_currents(v, dt)
            dv/dt = (v - e) + i
            v = (modifier * -receptors.get_receptor_currents(dt, c_m)) + v
            synaptic_neurotransmitters.apply_t_changes()
    [end]
    "#);
}