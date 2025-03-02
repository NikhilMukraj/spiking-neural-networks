mod lif_reference;


#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use spiking_neural_networks::neuron::iterate_and_spike::{XReceptor, DefaultReceptorsType}; 
    use crate::lif_reference::ReferenceIntegrateAndFire;

    
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

    const VOLTAGES: [f32; 11] = [-50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.];
    const T: [f32; 11] = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.];

    #[test]
    pub fn test_electrical_accuracy() {
        for i in VOLTAGES {
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

    #[test]
    pub fn test_chemical_accuracy() {
        for has_receptors in [false, true] {
            for t in T {
                for i in VOLTAGES {
                    let mut test_output: Vec<f32> = vec![];
                    let mut test_is_spikings: Vec<bool> = vec![];
                    let mut reference_output: Vec<f32> = vec![];
                    let mut reference_is_spikings: Vec<bool> = vec![];

                    let mut to_test: BasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                        BasicIntegrateAndFire::default();
                    if has_receptors {
                        to_test.receptors.insert(
                            DefaultReceptorsNeurotransmitterType::X,
                            DefaultReceptorsType::X(XReceptor::default()),
                        ).expect("Could not insert receptor");
                    }
                    let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                        ReferenceIntegrateAndFire::default();
                    if has_receptors {
                        reference_neuron.receptors.insert(
                            DefaultReceptorsNeurotransmitterType::X,
                            DefaultReceptorsType::X(XReceptor::default()),
                        ).expect("Could not insert receptor");
                    }

                    let conc = HashMap::from([(DefaultReceptorsNeurotransmitterType::X, t)]);

                    for _ in 0..1000 {
                        test_is_spikings.push(
                            to_test.iterate_with_neurotransmitter_and_spike(i, &conc)
                        );
                        test_output.push(to_test.current_voltage);
                        reference_is_spikings.push(
                            reference_neuron.iterate_with_neurotransmitter_and_spike(i, &conc)
                        );
                        reference_output.push(reference_neuron.current_voltage);
                    }

                    test_output.retain(|&x| x.is_finite());
                    reference_output.retain(|&x| x.is_finite());

                    assert_eq!(test_is_spikings, reference_is_spikings);
                    assert_eq!(test_output, reference_output);
                }
            }
        }
    }
}