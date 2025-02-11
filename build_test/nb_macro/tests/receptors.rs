#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use spiking_neural_networks::neuron::iterate_and_spike::ApproximateReceptor;


    // test multiple receptor types, (X and Y)
    // test mismatching receptors
    // test getting and seting currents and iterating (with different concs of neurotransmitter)
    // then move to metabotropic stuff (with top level vars)

    neuron_builder!(r#"
    [receptors]
        type: BasicReceptors
        neurotransmitter: X
        vars: current = 0, g = 1, e = 0
        on_iteration:
            current = g * r * (v - e)
    [end]

    [receptors]
        type: MultipleReceptors
        neurotransmitter: A
        vars: current = 0, g = 1, e = 0
        on_iteration:
            current = g * r * (v - e)
        neurotransmitter: B
        vars: current = 0, g = 1, e = 0
        on_iteration:
            current = 2 * g * r * (v - e)
    [end]

    [receptors]
        type: OneMetabotropic
        neurotransmitter: M
        vars: m = 0
        on_iteration:
            m = r ^ 2
    [end]

    [receptors]
        type: Mixed
        vars: m = 0
        neurotransmitter: Iono
        vars: current = 0, g = 1, e = 0
        on_iteration:
            current = g * r * (v - e)
        neurotransmitter: Meta
        vars: s = 0
        on_iteration:
            s = 2 * s + 1
    [end]
    "#);

    #[test]
    fn test_no_receptors_current() {
        let receptors = BasicReceptors::<ApproximateReceptor>::default();

        assert_eq!(receptors.get_receptor_currents(1., 1.), 0.);
    }

    #[test]
    fn test_multi_no_receptors_current() {
        let receptors = MultipleReceptors::<ApproximateReceptor>::default();

        assert_eq!(receptors.get_receptor_currents(1., 1.), 0.);
    }

    #[test]
    fn test_mismatch() {
        let mut receptors = MultipleReceptors::<ApproximateReceptor>::default();

        assert!(
            receptors.insert(
                MultipleReceptorsNeurotransmitterType::A, 
                MultipleReceptorsType::B(BReceptor::default())
            ).is_err()
        );
        assert!(
            receptors.insert(
                MultipleReceptorsNeurotransmitterType::B, 
                MultipleReceptorsType::A(AReceptor::default())
            ).is_err()
        );

        assert!(
            receptors.insert(
                MultipleReceptorsNeurotransmitterType::A, 
                MultipleReceptorsType::A(AReceptor::default())
            ).is_ok()
        );
        assert!(
            receptors.insert(
                MultipleReceptorsNeurotransmitterType::B, 
                MultipleReceptorsType::B(BReceptor::default())
            ).is_ok()
        );
    }

    #[test]
    fn test_simple_metabotropic() {
        let mut receptors = OneMetabotropic::<ApproximateReceptor>::default();

        receptors.insert(
            OneMetabotropicNeurotransmitterType::M, 
            OneMetabotropicType::M(MReceptor::default())
        ).unwrap();

        let ts = [0., 0.25, 0.5, 0.75, 0.1, 0.75, 0.5, 0.25, 0.];

        for t in ts {
            let conc = HashMap::from([(OneMetabotropicNeurotransmitterType::M, t)]);
            receptors.update_receptor_kinetics(&conc, 1.);

            if let Some(OneMetabotropicType::M(receptor)) = receptors.get(&OneMetabotropicNeurotransmitterType::M) {
                assert_eq!(receptor.r.get_r(), t);
            } else {
                panic!("Receptor should have been inserted");
            }
        }
    }
}