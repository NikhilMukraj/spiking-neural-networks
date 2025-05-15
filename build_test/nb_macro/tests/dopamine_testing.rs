mod izhikevich_dopamine;
mod ionotropic_channels;


#[cfg(test)]
mod test {
    use std::collections::HashMap;
    use crate::izhikevich_dopamine::{
        BoundedReceptorKinetics, DopaGluGABA, DopaGluGABANeurotransmitterType, DopaGluGABAType, DopamineReceptor, GlutamateReceptor, IzhikevichNeuron
    };
    use crate::ionotropic_channels::{AMPAReceptor, Ionotropic, IonotropicNeurotransmitterType, IonotropicType, NMDAReceptor};
    use spiking_neural_networks::neuron::iterate_and_spike::{
        IonotropicReception, IterateAndSpike, ReceptorKinetics, Receptors
    };


    const ITERATIONS: usize = 10_000;

    #[test]
    fn test_functionality_with_no_dopamine() {
        let ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let voltages: Vec<f32> = (0..400).map(|i| (i as f32 / 2.) - 100.).collect();

        for glu in ts {
            let mut dopamine_receptors = DopaGluGABA::default_impl();
            dopamine_receptors.insert(
                DopaGluGABANeurotransmitterType::Glutamate,
                DopaGluGABAType::Glutamate(GlutamateReceptor::default()),
            ).unwrap();
            dopamine_receptors.insert(
                DopaGluGABANeurotransmitterType::Dopamine,
                DopaGluGABAType::Dopamine(DopamineReceptor { 
                    s_d2: 1., 
                    s_d1: 1., 
                    r_d1: BoundedReceptorKinetics::default(), 
                    r_d2: BoundedReceptorKinetics::default(),
                })
            ).unwrap();

            let mut reference_receptors = Ionotropic::<BoundedReceptorKinetics>::default();
            reference_receptors.insert(
                IonotropicNeurotransmitterType::AMPA,
                IonotropicType::AMPA(AMPAReceptor::default()),
            ).unwrap();
            reference_receptors.insert(
                IonotropicNeurotransmitterType::NMDA,
                IonotropicType::NMDA(NMDAReceptor::default())
            ).unwrap();

            let t_total_dopa = HashMap::from([
                (DopaGluGABANeurotransmitterType::Glutamate, glu),
                (DopaGluGABANeurotransmitterType::Dopamine, 0.)
            ]);
            let t_total = HashMap::from([
                (IonotropicNeurotransmitterType::AMPA, glu),
                (IonotropicNeurotransmitterType::NMDA, glu)
            ]);

            dopamine_receptors.update_receptor_kinetics(&t_total_dopa, 0.);
            reference_receptors.update_receptor_kinetics(&t_total, 0.);

            for voltage in &voltages {
                dopamine_receptors.set_receptor_currents(*voltage, 0.1);
                reference_receptors.set_receptor_currents(*voltage, 0.1);

                let current = dopamine_receptors.get_receptor_currents(0.1, 25.);
                let reference_current = reference_receptors.get_receptor_currents(0.1, 25.);

                assert_eq!(current, reference_current);
            }
        }
    }

    #[test]
    fn test_d1_functionality() {
        let glu_ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let dopamine_ts = glu_ts.clone();

        let mut spike_counts: Vec<Vec<usize>> = (0..11).map(|_| (0..11).map(|_| 0).collect()).collect();
        
        for (n, glu) in glu_ts.iter().enumerate() {
            for (m, dopamine) in dopamine_ts.iter().enumerate() {
                let mut neuron = IzhikevichNeuron::default_impl();

                neuron.receptors
                    .insert(DopaGluGABANeurotransmitterType::Glutamate, DopaGluGABAType::Glutamate(GlutamateReceptor::default()))
                    .expect("Valid neurotransmitter pairing");
                neuron.receptors
                    .insert(
                        DopaGluGABANeurotransmitterType::Dopamine, 
                        DopaGluGABAType::Dopamine(DopamineReceptor { s_d2: 0., s_d1: 1., ..DopamineReceptor::default() })
                    )
                    .expect("Valid neurotransmitter pairing");

                let t_total = HashMap::from([
                    (DopaGluGABANeurotransmitterType::Glutamate, *glu),
                    (DopaGluGABANeurotransmitterType::Dopamine, *dopamine)
                ]);

                let mut spikes = 0;
                for _ in 0..ITERATIONS {
                    let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &t_total);
                    if is_spiking {
                        spikes += 1;
                    }
                    match neuron.receptors.get(&DopaGluGABANeurotransmitterType::Dopamine).unwrap() {
                        DopaGluGABAType::Dopamine(receptor) => assert_eq!(receptor.r_d1.get_r(), *dopamine),
                        _ => unreachable!()
                    }
                    assert_eq!(neuron.receptors.nmda_modifier, 1. - *dopamine);
                    assert_eq!(neuron.receptors.inh_modifier, 1.);
                }

                spike_counts[n][m] = spikes;
            }
        }

        for i in 1..11 {
            for j in 1..11 {
                assert!(spike_counts[i][j] >= spike_counts[i][j - 1]);
                assert!(spike_counts[j][i] >= spike_counts[j - 1][i]);
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 3..8 {
            assert!(
                spike_counts[i][0] < spike_counts[i][10], 
                "{}: {} < {}", 
                i, 
                spike_counts[i][0], 
                spike_counts[i][10]
            );
        }
    }

    #[test]
    fn test_d2_functionality() {
        let glu_ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let dopamine_ts = glu_ts.clone();

        let mut spike_counts: Vec<Vec<usize>> = (0..11).map(|_| (0..11).map(|_| 0).collect()).collect();

        for (n, glu) in glu_ts.iter().enumerate() {
            for (m, dopamine) in dopamine_ts.iter().enumerate() {
                let mut neuron = IzhikevichNeuron::default_impl();

                neuron.receptors
                    .insert(DopaGluGABANeurotransmitterType::Glutamate, DopaGluGABAType::Glutamate(GlutamateReceptor::default()))
                    .expect("Valid neurotransmitter pairing");
                neuron.receptors
                    .insert(
                        DopaGluGABANeurotransmitterType::Dopamine, 
                        DopaGluGABAType::Dopamine(DopamineReceptor { s_d2: 0.5, s_d1: 0., ..DopamineReceptor::default() })
                    )
                    .expect("Valid neurotransmitter pairing");

                let t_total = HashMap::from([
                    (DopaGluGABANeurotransmitterType::Glutamate, *glu),
                    (DopaGluGABANeurotransmitterType::Dopamine, *dopamine)
                ]);

                let mut spikes = 0;
                for _ in 0..ITERATIONS {
                    let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &t_total);
                    if is_spiking {
                        spikes += 1;
                    }
                    match neuron.receptors.get(&DopaGluGABANeurotransmitterType::Dopamine).unwrap() {
                        DopaGluGABAType::Dopamine(receptor) => assert_eq!(receptor.r_d2.get_r(), *dopamine),
                        _ => unreachable!()
                    }
                    assert_eq!(neuron.receptors.nmda_modifier, 1.);
                    assert_eq!(neuron.receptors.inh_modifier, 1. - (0.5 * *dopamine));
                }

                spike_counts[n][m] = spikes;
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 1..11 {
            for j in 1..11 {
                assert!(spike_counts[i][j] <= spike_counts[i][j - 1]);
            }
        }
    }
}