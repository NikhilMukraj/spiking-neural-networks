mod lif_reference;


#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use crate::lif_reference::ReferenceIntegrateAndFire;


    neuron_builder!(r#"
    [neuron]
        type: BasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, flag = false, out = 0
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            [if] flag [then]
                out = 1
            [else]
                out = 2
            [end]

            dv/dt = (v - e) + i
    [end]
    "#);

    #[test]
    pub fn test_bool_var() {
        let voltages = [-50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.];
        let flag_types = [false, true];

        for i in voltages {
            for j in flag_types {
                let mut test_output: Vec<f32> = vec![];
                let mut test_is_spikings: Vec<bool> = vec![];
                let mut reference_output: Vec<f32> = vec![];
                let mut reference_is_spikings: Vec<bool> = vec![];

                let mut to_test: BasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                    BasicIntegrateAndFire::default();
                let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                    ReferenceIntegrateAndFire::default();

                to_test.flag = j;

                for _ in 0..1000 {
                    test_is_spikings.push(to_test.iterate_and_spike(i));
                    test_output.push(to_test.current_voltage);
                    reference_is_spikings.push(reference_neuron.iterate_and_spike(i));
                    reference_output.push(reference_neuron.current_voltage);
                }

                assert_eq!(test_is_spikings, reference_is_spikings);
                assert_eq!(test_output, reference_output);

                if j {
                    assert_eq!(to_test.out, 1.)
                } else {
                    assert_eq!(to_test.out, 2.);
                }
            }
        }
    }
}