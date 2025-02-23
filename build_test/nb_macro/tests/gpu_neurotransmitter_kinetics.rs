#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder; 

    neuron_builder!(r#"
    [neurotransmitter_kinetics]
        type: BasicNeurotransmitterKinetics
        vars: t_max = 1, c = 0.001, conc = 0
        on_iteration:
            [if] is_spiking [then]
                conc = t_max
            [else]
                conc = 0
            [end]

            t = t + dt * -c * t + conc

            t = min(max(t, 0), t_max)
    [end]
    "#);
}
