#[cfg(test)]
mod test {
    use spiking_neural_networks::neuron::{iterate_and_spike::{ApproximateNeurotransmitter, IonotropicNeurotransmitterType}, spike_train::{DeltaDiracRefractoriness, RateSpikeTrain, SpikeTrain}};


    // count number of spikes and compare against given rate and number of iterations

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
}