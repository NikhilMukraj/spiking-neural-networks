#[cfg(test)]
mod tests {
    use nb_macro::neuron_builder;


    neuron_builder!(r#"
        [neuron]
            type: BasicIntegrateAndFire
            vars: e = 0, v_reset = -75, v_th = -55, dt = 100
            on_spike: 
                v = v_reset
            spike_detection: v >= v_th
            on_iteration:
                dv/dt = (v - e) + i
        [end]
    "#);

    #[test]
    fn test_custom_dt() {
        let lif = BasicIntegrateAndFire::default_impl();

        assert_eq!(lif.dt, 100.);
    }

    // test if gpu kernel and conversion works as expected
}
