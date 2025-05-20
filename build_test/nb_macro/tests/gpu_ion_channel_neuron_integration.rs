#[cfg(feature = "gpu")]
#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;


    neuron_builder!(r#"
        [ion_channel]
            type: TestLeak
            vars: e = 0, g = 1,
            on_iteration:
                current = g * (v - e)
        [end]

        [neuron]
            type: LIF
            ion_channels: l = TestLeak
            vars: v_reset = -75, v_th = -55
            on_spike: 
                v = v_reset
            spike_detection: v >= v_th
            on_iteration:
                l.update_current(v)
                dv/dt = l.current + i
        [end]
    "#);

    // #[test]
    // fn test_neuron_conversion_empty() {}   

    // #[test]
    // fn test_neuron_conversion() {}   

    // #[test]
    // fn test_neuron_conversion_non_square() {}   

    // #[test]
    // fn test_neuron_conversion_electrochemical_empty() {}   

    // #[test]
    // fn test_neuron_conversion_electrochemical() {}   

    // #[test]
    // fn test_neuron_conversion_electrochemical_non_square() {}   
}