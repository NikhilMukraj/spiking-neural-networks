// default receptors
// adding receptors
// default on electrochemical iteration and custom on electrochemical iteration
// move existing receptors to test in receptors test file to a shared file
// to use here for testing neuron receptor integrations

mod shared_receptors;

#[allow(clippy::assign_op_pattern)]
#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use nb_macro::neuron_builder;
    use crate::shared_receptors::{
        AReceptor, BReceptor, CombinedReceptor, CombinedReceptors, 
        CombinedReceptorsNeurotransmitterType, CombinedReceptorsType, 
        IonoReceptor, MetaReceptor, MixedReceptors, MixedReceptorsNeurotransmitterType, 
        MixedReceptorsType, MultipleReceptors, MultipleReceptorsNeurotransmitterType, 
        MultipleReceptorsType,
    };

    neuron_builder!(r#"
    [neuron]
        type: MultiIntegrateAndFire
        receptors: MultipleReceptors
        vars: e = 0, v_reset = -75, v_th = -55
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
    [end]

    [neuron]
        type: MixedIntegrateAndFire
        receptors: MixedReceptors
        vars: e = 0, v_reset = -75, v_th = -55
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
    [end]

    [neuron]
        type: ElectroChemicalIntegrateAndFire
        receptors: MultipleReceptors
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

    [neuron]
        type: CombinedIntegrateAndFire
        receptors: CombinedReceptors
        vars: e = 0, v_reset = -75, v_th = -55
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
    [end]
    "#);

    fn sum_bool(vector: &[bool]) -> usize {
        vector.iter().map(|i| if *i { 1 } else { 0 }).sum()
    }

    #[test]
    fn test_multiple_receptors() {
        let mut spiking_data: HashMap<(bool, bool), Vec<bool>> = HashMap::new();

        for (has_a, has_b) in [(false, false), (true, false), (true, true)] {
            let mut to_test: MultiIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor> = 
                MultiIntegrateAndFire::default();

            if has_a {
                to_test.receptors.insert(
                    MultipleReceptorsNeurotransmitterType::A,
                    MultipleReceptorsType::A(AReceptor { g: 2., ..AReceptor::default() }),
                ).unwrap()
            }
            if has_b {
                to_test.receptors.insert(
                    MultipleReceptorsNeurotransmitterType::B,
                    MultipleReceptorsType::B(BReceptor { g: 2., ..BReceptor::default() }),
                ).unwrap()
            }

            let conc = HashMap::from([
                (MultipleReceptorsNeurotransmitterType::A, 1.),
                (MultipleReceptorsNeurotransmitterType::B, 1.),
            ]);

            spiking_data.insert((has_a, has_b), vec![]);

            for _ in 0..100000 {
                let is_spiking = to_test.iterate_with_neurotransmitter_and_spike(0., &conc);
                spiking_data.get_mut(&(has_a, has_b)).unwrap().push(is_spiking);
            }
        }

        assert!(
            sum_bool(spiking_data.get(&(false, false)).unwrap()) < 
            sum_bool(spiking_data.get(&(true, false)).unwrap())
        );
        assert!(
            sum_bool(spiking_data.get(&(true, false)).unwrap()) < 
            sum_bool(spiking_data.get(&(true, true)).unwrap())
        );
    }

    #[test]
    fn test_mixed_receptors() {
        let mut spiking_data: HashMap<(bool, bool), Vec<bool>> = HashMap::new();

        for (has_meta, has_iono) in [(false, false), (true, false), (true, true)] {
            let mut to_test: MixedIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor> = 
                MixedIntegrateAndFire::default();

            if has_meta {
                to_test.receptors.insert(
                    MixedReceptorsNeurotransmitterType::Meta,
                    MixedReceptorsType::Meta(MetaReceptor::default()),
                ).unwrap()
            }
            if has_iono {
                to_test.receptors.insert(
                    MixedReceptorsNeurotransmitterType::Iono,
                    MixedReceptorsType::Iono(IonoReceptor { g: 2., ..IonoReceptor::default() }),
                ).unwrap()
            }

            let conc = HashMap::from([
                (MixedReceptorsNeurotransmitterType::Iono, 1.),
                (MixedReceptorsNeurotransmitterType::Meta, 1.),
            ]);

            spiking_data.insert((has_meta, has_iono), vec![]);

            for _ in 0..100000 {
                let is_spiking = to_test.iterate_with_neurotransmitter_and_spike(0., &conc);
                spiking_data.get_mut(&(has_meta, has_iono)).unwrap().push(is_spiking);
            }
        }

        assert_eq!(
            sum_bool(spiking_data.get(&(false, false)).unwrap()),
            sum_bool(spiking_data.get(&(true, false)).unwrap())
        );
        assert!(
            sum_bool(spiking_data.get(&(true, false)).unwrap()) < 
            sum_bool(spiking_data.get(&(true, true)).unwrap())
        );
    }

    #[test]
    fn test_custom_electrochemical_same() {
        for t in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]  {
            for i in (-50..50).step_by(10) {
                for (has_a, has_b) in [(false, false), (true, false), (true, true)] {
                    let mut to_test: ElectroChemicalIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor> = 
                        ElectroChemicalIntegrateAndFire::default();
                    let mut ref_neuron: MultiIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor> = 
                        MultiIntegrateAndFire::default();

                    to_test.dt = 1.;
                    ref_neuron.dt = 1.;
        
                    if has_a {
                        to_test.receptors.insert(
                            MultipleReceptorsNeurotransmitterType::A,
                            MultipleReceptorsType::A(AReceptor { g: 2., ..AReceptor::default() }),
                        ).unwrap();
                        ref_neuron.receptors.insert(
                            MultipleReceptorsNeurotransmitterType::A,
                            MultipleReceptorsType::A(AReceptor { g: 2., ..AReceptor::default() }),
                        ).unwrap();
                    }
                    if has_b {
                        to_test.receptors.insert(
                            MultipleReceptorsNeurotransmitterType::B,
                            MultipleReceptorsType::B(BReceptor { g: 2., ..BReceptor::default() }),
                        ).unwrap();
                        ref_neuron.receptors.insert(
                            MultipleReceptorsNeurotransmitterType::B,
                            MultipleReceptorsType::B(BReceptor { g: 2., ..BReceptor::default() }),
                        ).unwrap();
                    }
        
                    let conc = HashMap::from([
                        (MultipleReceptorsNeurotransmitterType::A, t),
                        (MultipleReceptorsNeurotransmitterType::B, t),
                    ]);

                    for _ in 0..1000 {
                        let is_spiking1 = to_test.iterate_with_neurotransmitter_and_spike(
                            i as f32, &conc
                        );
                        let is_spiking2 = ref_neuron.iterate_with_neurotransmitter_and_spike(
                            i as f32, &conc
                        );

                        assert_eq!(is_spiking1, is_spiking2);

                        if to_test.current_voltage.is_finite() && ref_neuron.current_voltage.is_finite() {
                            assert!(
                                ((to_test.current_voltage - ref_neuron.current_voltage) / ref_neuron.current_voltage).abs() < 0.01,
                                "{} != {}",
                                to_test.current_voltage,
                                ref_neuron.current_voltage,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_custom_electrochemical_differing() {
        let mut total_spikes1 = 0;
        let mut total_spikes2 = 0;

        for t in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]  {
            for i in (-50..50).step_by(10) {
                for (has_a, has_b) in [(false, false), (true, false), (true, true)] {
                    let mut to_test: ElectroChemicalIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor> = 
                        ElectroChemicalIntegrateAndFire::default();
                    let mut ref_neuron: MultiIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor> = 
                        MultiIntegrateAndFire::default();

                    to_test.modifier = 3.;
                    to_test.dt = 1.;
                    ref_neuron.dt = 1.;
        
                    if has_a {
                        to_test.receptors.insert(
                            MultipleReceptorsNeurotransmitterType::A,
                            MultipleReceptorsType::A(AReceptor { g: 2., ..AReceptor::default() }),
                        ).unwrap();
                        ref_neuron.receptors.insert(
                            MultipleReceptorsNeurotransmitterType::A,
                            MultipleReceptorsType::A(AReceptor { g: 2., ..AReceptor::default() }),
                        ).unwrap();
                    }
                    if has_b {
                        to_test.receptors.insert(
                            MultipleReceptorsNeurotransmitterType::B,
                            MultipleReceptorsType::B(BReceptor { g: 2., ..BReceptor::default() }),
                        ).unwrap();
                        ref_neuron.receptors.insert(
                            MultipleReceptorsNeurotransmitterType::B,
                            MultipleReceptorsType::B(BReceptor { g: 2., ..BReceptor::default() }),
                        ).unwrap();
                    }
        
                    let conc = HashMap::from([
                        (MultipleReceptorsNeurotransmitterType::A, t),
                        (MultipleReceptorsNeurotransmitterType::B, t),
                    ]);

                    for _ in 0..1000 {
                        let is_spiking1 = to_test.iterate_with_neurotransmitter_and_spike(
                            i as f32, &conc
                        );
                        let is_spiking2 = ref_neuron.iterate_with_neurotransmitter_and_spike(
                            i as f32, &conc
                        );

                        if is_spiking1 {
                            total_spikes1 += 1;
                        }
                        if is_spiking2 {
                            total_spikes2 += 1;
                        }
                    }
                }
            }
        }

        assert!(total_spikes1 > total_spikes2);
    }

    #[test]
    fn test_combined_receptors() {
        for t in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.] {
            let mut to_test: CombinedIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor> =
                CombinedIntegrateAndFire::default();
            let mut ref_neuron: MultiIntegrateAndFire::<ApproximateNeurotransmitter, ApproximateReceptor> =
                MultiIntegrateAndFire::default();

            to_test.dt = 1.;
            ref_neuron.dt = 1.;

            to_test.receptors.insert(
                CombinedReceptorsNeurotransmitterType::Combined, 
                CombinedReceptorsType::Combined(CombinedReceptor::default())
            ).unwrap();

            ref_neuron.receptors.insert(
                MultipleReceptorsNeurotransmitterType::A,
                MultipleReceptorsType::A(AReceptor { g: 2., ..AReceptor::default() }),
            ).unwrap();
            ref_neuron.receptors.insert(
                MultipleReceptorsNeurotransmitterType::B,
                MultipleReceptorsType::B(BReceptor { g: 1., ..BReceptor::default() }),
            ).unwrap();

            let to_test_conc = HashMap::from([
                (CombinedReceptorsNeurotransmitterType::Combined, t),
            ]);

            let ref_conc = HashMap::from([
                (MultipleReceptorsNeurotransmitterType::A, t),
                (MultipleReceptorsNeurotransmitterType::B, t),
            ]);

            for _ in 0..1000 {
                let is_spiking1 = to_test.iterate_with_neurotransmitter_and_spike(
                    0.,
                    &to_test_conc,
                );
                let is_spiking2 = ref_neuron.iterate_with_neurotransmitter_and_spike(
                    0.,
                    &ref_conc
                );

                // assert!(
                //     to_test.receptors.get(&CombinedReceptorsNeurotransmitterType::Combined).unwrap().get_r() ==
                //     to_test.receptors.get(&CombinedReceptorsNeurotransmitterType::Combined).unwrap().get_r() ==
                //     ref_neuron.receptors.get(&MultipleReceptorsNeurotransmitterType::A).unwrap().get_r() ==
                //     ref_neuron.receptors.get(&MultipleReceptorsNeurotransmitterType::B).unwrap().get_r()
                // );

                #[allow(irrefutable_let_patterns)]
                if let CombinedReceptorsType::Combined(receptor) = to_test.receptors.get(&CombinedReceptorsNeurotransmitterType::Combined).unwrap() {
                    assert_eq!(receptor.r1.get_r(), t);
                }
                #[allow(irrefutable_let_patterns)]
                if let CombinedReceptorsType::Combined(receptor) = to_test.receptors.get(&CombinedReceptorsNeurotransmitterType::Combined).unwrap() {
                    assert_eq!(receptor.r2.get_r(), t);
                }
                if let MultipleReceptorsType::A(receptor) = ref_neuron.receptors.get(&MultipleReceptorsNeurotransmitterType::A).unwrap() {
                    assert_eq!(receptor.r.get_r(), t);
                }
                if let MultipleReceptorsType::B(receptor) = ref_neuron.receptors.get(&MultipleReceptorsNeurotransmitterType::B).unwrap() {
                    assert_eq!(receptor.r.get_r(), t);
                }

                assert_eq!(is_spiking1, is_spiking2);

                if to_test.current_voltage.is_finite() && ref_neuron.current_voltage.is_finite() {
                    assert!(
                        ((to_test.current_voltage - ref_neuron.current_voltage) / ref_neuron.current_voltage).abs() < 0.01,
                        "{} != {}",
                        to_test.current_voltage,
                        ref_neuron.current_voltage,
                    );
                }
            }
        }
    }
}