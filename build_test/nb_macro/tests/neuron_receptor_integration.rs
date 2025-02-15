// default receptors
// adding receptors
// default on electrochemical iteration and custom on electrochemical iteration
// move existing receptors to test in receptors test file to a shared file
// to use here for testing neuron receptor integrations

mod shared_receptors;

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use nb_macro::neuron_builder;
    use crate::shared_receptors::{
        AReceptor, BReceptor, IonoReceptor, MetaReceptor, MixedReceptors, MixedReceptorsType,
        MixedReceptorsNeurotransmitterType, MultipleReceptors, MultipleReceptorsNeurotransmitterType, 
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
}