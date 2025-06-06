// remove existing system to insert current update in neuron if ion channel exists
// user should input where and how current is added

mod lif_reference;


#[cfg(test)]
mod test {
    use nb_macro::neuron_builder; 
    use crate::lif_reference::ReferenceIntegrateAndFire;

    
    neuron_builder!(r#"
        [ion_channel]
            type: TestLeak
            vars: e = 0, g = 1,
            on_iteration:
                current = g * (v - e)
        [end]

        [neuron]
            type: BasicIntegrateAndFire
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

    #[test]
    pub fn test_electrical_accuracy() {
        let voltages = [-50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.];

        for i in voltages {
            let mut test_output: Vec<f32> = vec![];
            let mut test_is_spikings: Vec<bool> = vec![];
            let mut reference_output: Vec<f32> = vec![];
            let mut reference_is_spikings: Vec<bool> = vec![];

            let mut to_test: BasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                BasicIntegrateAndFire::default();
            let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ReferenceIntegrateAndFire::default();

            for _ in 0..1000 {
                test_is_spikings.push(to_test.iterate_and_spike(i));
                test_output.push(to_test.current_voltage);
                reference_is_spikings.push(reference_neuron.iterate_and_spike(i));
                reference_output.push(reference_neuron.current_voltage);
            }

            assert_eq!(test_is_spikings, reference_is_spikings);
            assert_eq!(test_output, reference_output);
        }
    }
}
