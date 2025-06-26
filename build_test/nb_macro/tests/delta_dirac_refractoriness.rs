#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;
    use rand::Rng;
    use spiking_neural_networks::neuron::spike_train::DeltaDiracRefractoriness;


    neuron_builder!(
        "[neural_refractoriness]
            type: TestRefractoriness
            effect: (v_th - v_resting) * exp((-1 / (decay / dt)) * (time_difference ^ 2)) + v_resting
        [end]"
    );

    #[test]
    fn test_effect() {
        for _ in 0..50 {
            let test_decay = rand::thread_rng().gen_range(0.0..20_000.);

            let reference_refractoriness = DeltaDiracRefractoriness { k: test_decay };
            let test_refractoriness = TestRefractoriness { decay: test_decay };

            let last_firing_time = rand::thread_rng().gen_range(0..1000);
            let timestep = rand::thread_rng().gen_range(last_firing_time..(last_firing_time + 1000));

            let v_max = rand::thread_rng().gen_range(10.0..30.);
            let v_resting = 0.;

            let dt = 0.1;

            let reference = reference_refractoriness.get_effect(timestep, last_firing_time, v_max, v_resting, dt);
            let actual = test_refractoriness.get_effect(timestep, last_firing_time, v_max, v_resting, dt);

            assert!((reference - actual).abs() < 0.01, "{} != {}", reference, actual);
        }
    }
}