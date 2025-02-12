#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use spiking_neural_networks::neuron::iterate_and_spike::ApproximateReceptor;

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
        type: MixedReceptors
        vars: m = 0
        neurotransmitter: Iono
        vars: current = 0, g = 1, e = 0
        on_iteration:
            current = g * m * r * (v - e)
        neurotransmitter: Meta
        vars: s = 1
        on_iteration:
            m = s * r
    [end]
    "#);

    const VOLTAGES: [f32; 11] = [-50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.];
    const T: [f32; 9] = [0., 0.25, 0.5, 0.75, 0.1, 0.75, 0.5, 0.25, 0.];

    #[test]
    fn test_no_receptors_current() {
        let receptors = BasicReceptors::<ApproximateReceptor>::default();

        assert_eq!(receptors.get_receptor_currents(1., 1.), 0.);
    }

    #[test]
    fn test_receptors_with_current() {
        let mut receptors = BasicReceptors::<ApproximateReceptor>::default();

        receptors.insert(
            BasicReceptorsNeurotransmitterType::X, 
            BasicReceptorsType::X(XReceptor::default())
        ).unwrap();

        for voltage in VOLTAGES {
            for t in T {
                let conc = HashMap::from([(BasicReceptorsNeurotransmitterType::X, t)]);
                receptors.update_receptor_kinetics(&conc, 1.);
                receptors.set_receptor_currents(voltage, 1.);

                assert_eq!(receptors.get_receptor_currents(1., 1.), t * voltage);
            }
        }
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
    fn test_multiple_receptors_with_current() {
        let mut receptors = MultipleReceptors::<ApproximateReceptor>::default();

        receptors.insert(
            MultipleReceptorsNeurotransmitterType::A, 
            MultipleReceptorsType::A(AReceptor::default())
        ).unwrap();

        for voltage in VOLTAGES {
            for t in T {
                let conc = HashMap::from([(MultipleReceptorsNeurotransmitterType::A, t)]);
                receptors.update_receptor_kinetics(&conc, 1.);
                receptors.set_receptor_currents(voltage, 1.);

                assert_eq!(receptors.get_receptor_currents(1., 1.), t * voltage);
            }
        }

        let mut receptors = MultipleReceptors::<ApproximateReceptor>::default();

        receptors.insert(
            MultipleReceptorsNeurotransmitterType::B, 
            MultipleReceptorsType::B(BReceptor::default())
        ).unwrap();

        for voltage in VOLTAGES {
            for t in T {
                let conc = HashMap::from([(MultipleReceptorsNeurotransmitterType::B, t)]);
                receptors.update_receptor_kinetics(&conc, 1.);
                receptors.set_receptor_currents(voltage, 1.);

                assert_eq!(receptors.get_receptor_currents(1., 1.), 2. * t * voltage);
            }
        }

        receptors.insert(
            MultipleReceptorsNeurotransmitterType::A, 
            MultipleReceptorsType::A(AReceptor::default())
        ).unwrap();

        for voltage in VOLTAGES {
            for t1 in T {
                for t2 in T {
                    let conc = HashMap::from(
                        [
                            (MultipleReceptorsNeurotransmitterType::A, t1),
                            (MultipleReceptorsNeurotransmitterType::B, t2)
                        ]
                    );
                    receptors.update_receptor_kinetics(&conc, 1.);
                    receptors.set_receptor_currents(voltage, 1.);

                    assert_eq!(
                        receptors.get_receptor_currents(1., 1.), 
                        t1 * voltage + 2. * t2 * voltage,
                    );
                }
            }
        }
    }

    #[test]
    fn test_simple_metabotropic() {
        let mut receptors = OneMetabotropic::<ApproximateReceptor>::default();

        receptors.insert(
            OneMetabotropicNeurotransmitterType::M, 
            OneMetabotropicType::M(MReceptor::default())
        ).unwrap();

        for t in T {
            let conc = HashMap::from([(OneMetabotropicNeurotransmitterType::M, t)]);
            receptors.update_receptor_kinetics(&conc, 1.);

            if let Some(OneMetabotropicType::M(receptor)) = receptors.get(&OneMetabotropicNeurotransmitterType::M) {
                assert_eq!(receptor.r.get_r(), t);
            } else {
                panic!("Receptor should have been inserted");
            }
        }
    }

    #[test]
    fn test_meta_iono_interaction() {
        let mut receptors = MixedReceptors::<ApproximateReceptor>::default();

        receptors.insert(
            MixedReceptorsNeurotransmitterType::Iono, 
            MixedReceptorsType::Iono(IonoReceptor::default())
        ).unwrap();

        receptors.insert(
            MixedReceptorsNeurotransmitterType::Meta, 
            MixedReceptorsType::Meta(MetaReceptor::default())
        ).unwrap();

        let m_ts = T;
        let i_ts = T;

        for voltage in VOLTAGES {
            for m_t in m_ts {
                for i_t in i_ts {
                    let conc = HashMap::from(
                        [
                            (MixedReceptorsNeurotransmitterType::Meta, m_t), 
                            (MixedReceptorsNeurotransmitterType::Iono, i_t),
                        ],
                    );

                    receptors.update_receptor_kinetics(&conc, 1.);
                    receptors.set_receptor_currents(voltage, 1.);
        
                    if let Some(MixedReceptorsType::Meta(receptor)) = receptors.get(&MixedReceptorsNeurotransmitterType::Meta) {
                        assert_eq!(receptor.r.get_r(), m_t);
                    } else {
                        panic!("Receptor should have been inserted");
                    }

                    if let Some(MixedReceptorsType::Iono(receptor)) = receptors.get(&MixedReceptorsNeurotransmitterType::Iono) {
                        assert_eq!(receptor.r.get_r(), i_t);
                    } else {
                        panic!("Receptor should have been inserted");
                    }

                    assert_eq!(receptors.get_receptor_currents(1., 1.), m_t * i_t * voltage);
                }
            }
        }
    }
}