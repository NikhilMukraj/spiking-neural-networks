#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use spiking_neural_networks::neuron::{iterate_and_spike::{ApproximateNeurotransmitter, IonotropicNeurotransmitterType}, spike_train::DeltaDiracRefractoriness};


    neuron_builder!(
        "[spike_train]
            type: RateSpikeTrain
            vars: step = 0., rate = 0.
            on_iteration:
                step += dt
                [if] rate != 0. && step >= rate [then]
                    step = 0
                    current_voltage = v_th
                    is_spiking = true
                [else]
                    current_voltage = v_resting
                    is_spiking = false
                [end]
        [end]"
    );

    const ITERATIONS: usize = 10_000;

    #[test]
    fn test_expected_rate() {
        let rates = [0, 100, 200, 300, 400, 500];

        for rate in rates {
            let mut spike_train: RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> = RateSpikeTrain {
                rate: rate as f32,
                ..Default::default()
            };

            let mut spikes = 0;
            for _ in 0..ITERATIONS {
                let is_spiking = spike_train.iterate();
                if is_spiking {
                    spikes += 1;
                }
            }

            if rate == 0 {
                assert_eq!(spikes, 0);
            } else {
                assert!((spikes as f32 - (ITERATIONS as f32 / (rate as f32 / 0.1))).abs() <= 1.);
            }
        }
    }

    #[test]
    fn test_spacing() {
        let mut spike_train: RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> = RateSpikeTrain {
            rate: 100.,
            dt: 1.,
            ..Default::default()
        };

        for i in 0..1001 {
            let is_spiking = spike_train.iterate();

            if i == 0 || (i + 1) % 100 != 0 {
                assert!(!is_spiking);
            } else {
                assert!(is_spiking);
            }
        }
    }

    #[test]
    fn test_neurotransmitter() {
        let mut spike_train: RateSpikeTrain<IonotropicNeurotransmitterType, ApproximateNeurotransmitter, DeltaDiracRefractoriness> = RateSpikeTrain {
            rate: 100.,
            dt: 1.,
            ..Default::default()
        };

        spike_train.synaptic_neurotransmitters.insert(IonotropicNeurotransmitterType::AMPA, ApproximateNeurotransmitter::default());

        for i in 0..1001 {
            let _ = spike_train.iterate();

            if i != 0 && (i + 1) % 100 == 0 {
                assert_eq!(spike_train.synaptic_neurotransmitters.get(&IonotropicNeurotransmitterType::AMPA).unwrap().t, 1.);
            } else {
                assert_ne!(spike_train.synaptic_neurotransmitters.get(&IonotropicNeurotransmitterType::AMPA).unwrap().t, 1.);
            }
        }
    }
}
