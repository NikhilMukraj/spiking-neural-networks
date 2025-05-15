mod ionotropic_channels;

#[cfg(test)]
mod tests {
    use nb_macro::neuron_builder;
    use crate::ionotropic_channels::{
        Ionotropic, IonotropicType, IonotropicNeurotransmitterType,
        AMPAReceptor, NMDAReceptor, GABAReceptor,
    };

    neuron_builder!("
        [neuron]
            type: LIF
            receptors: Ionotropic
            vars: v_reset = -75, v_th = -55, g = 0.1, e = 0
            on_spike: v = v_reset
            spike_detection: v > v_th
            on_iteration: dv/dt = -g * (v - e) + i
        [end]
    ");

    // more ampa = more spikes
    // more nmda = more spikes
    // more mg = less spikes
    // more ampa + nmda = more spikes
    // more gaba = less spikes

    const ITERATIONS: usize = 10_000;

    #[test]
    fn test_ampa() {
        let ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let mut spike_counts = vec![];

        for t in ts {
            let mut neuron = LIF {
                dt: 1.,
                ..LIF::default_impl()
            };
            neuron.receptors.insert(
                IonotropicNeurotransmitterType::AMPA,
                IonotropicType::AMPA(AMPAReceptor::default()),
            ).unwrap();
            let mut spikes = 0;
            let conc = HashMap::from([(IonotropicNeurotransmitterType::AMPA, t)]);

            for _ in 0..ITERATIONS {
                let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &conc);
                if is_spiking {
                    spikes += 1;
                }
            }

            spike_counts.push(spikes);
        }

        for i in 1..11 {
            assert!(spike_counts[i] >= spike_counts[i - 1]);
        }
        assert!(spike_counts[0] < *spike_counts.last().unwrap());
    }

    #[test]
    fn test_nmda() {
        let ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let mut spike_counts = vec![];

        for t in ts {
            let mut neuron = LIF {
                dt: 1.,
                ..LIF::default_impl()
            };
            neuron.receptors.insert(
                IonotropicNeurotransmitterType::NMDA,
                IonotropicType::NMDA(NMDAReceptor::default()),
            ).unwrap();
            let mut spikes = 0;
            let conc = HashMap::from([(IonotropicNeurotransmitterType::NMDA, t)]);

            for _ in 0..ITERATIONS {
                let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &conc);
                if is_spiking {
                    spikes += 1;
                }
            }

            spike_counts.push(spikes);
        }

        for i in 1..11 {
            assert!(spike_counts[i] >= spike_counts[i - 1]);
        }
        assert!(spike_counts[0] < *spike_counts.last().unwrap());
    }

    #[test]
    fn test_mg() {
        let mgs: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let mut spike_counts = vec![];

        for mg in mgs {
            let mut neuron = LIF {
                dt: 1.,
                ..LIF::default_impl()
            };
            neuron.receptors.insert(
                IonotropicNeurotransmitterType::NMDA,
                IonotropicType::NMDA(NMDAReceptor { mg, ..NMDAReceptor::default() }),
            ).unwrap();
            let mut spikes = 0;
            let conc = HashMap::from([(IonotropicNeurotransmitterType::NMDA, 1.)]);

            for _ in 0..ITERATIONS {
                let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &conc);
                if is_spiking {
                    spikes += 1;
                }
            }

            spike_counts.push(spikes);
        }

        for i in 1..11 {
            assert!(spike_counts[i] <= spike_counts[i - 1]);
        }
        assert!(spike_counts[0] > *spike_counts.last().unwrap());
    }

    #[allow(clippy::needless_range_loop)]
    #[test]
    fn test_ampa_nmda() {
        let ampa_ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let nmda_ts: Vec<f32> = ampa_ts.clone();

        let mut spike_counts: Vec<Vec<usize>> = (0..11).map(|_| (0..11).map(|_| 0).collect()).collect();

        for (n, ampa) in ampa_ts.iter().enumerate() {
            for (m, nmda) in nmda_ts.iter().enumerate() {
                let mut neuron = LIF {
                    dt: 1.,
                    ..LIF::default_impl()
                };
                neuron.receptors.insert(
                    IonotropicNeurotransmitterType::AMPA,
                    IonotropicType::AMPA(AMPAReceptor::default()),
                ).unwrap();
                neuron.receptors.insert(
                    IonotropicNeurotransmitterType::NMDA,
                    IonotropicType::NMDA(NMDAReceptor::default()),
                ).unwrap();

                let mut spikes = 0;
                let conc = HashMap::from([
                    (IonotropicNeurotransmitterType::AMPA, *ampa),
                    (IonotropicNeurotransmitterType::NMDA, *nmda),
                ]);
    
                for _ in 0..ITERATIONS {
                    let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &conc);
                    if is_spiking {
                        spikes += 1;
                    }
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
    }

    #[test]
    fn test_gaba() {
        let ts: Vec<f32> = (0..11).map(|i| i as f32 / 10.).collect();
        let mut spike_counts = vec![];

        for t in ts {
            let mut neuron = LIF {
                dt: 1.,
                ..LIF::default_impl()
            };
            neuron.receptors.insert(
                IonotropicNeurotransmitterType::AMPA,
                IonotropicType::AMPA(AMPAReceptor::default()),
            ).unwrap();
            neuron.receptors.insert(
                IonotropicNeurotransmitterType::GABA,
                IonotropicType::GABA(GABAReceptor::default()),
            ).unwrap();
            let mut spikes = 0;
            let conc = HashMap::from([
                (IonotropicNeurotransmitterType::AMPA, 1.0),
                (IonotropicNeurotransmitterType::GABA, t),
            ]);

            for _ in 0..ITERATIONS {
                let is_spiking = neuron.iterate_with_neurotransmitter_and_spike(0., &conc);
                if is_spiking {
                    spikes += 1;
                }
            }

            spike_counts.push(spikes);
        }

        for i in 1..11 {
            assert!(spike_counts[i] <= spike_counts[i - 1]);
        }
        assert!(spike_counts[0] > *spike_counts.last().unwrap());
    }
}
