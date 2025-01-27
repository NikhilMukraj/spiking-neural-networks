mod lif_reference;


#[cfg(test)]
mod test {
    use nb_macro::neuron_builder_from_file; 
    use crate::lif_reference::ReferenceIntegrateAndFire;

    neuron_builder_from_file!("./tests/lif.nb");

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

    // #[test]
    // pub fn test_chemical_accuracy() {

    // }
}