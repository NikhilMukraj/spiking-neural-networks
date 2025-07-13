#[cfg(test)]
mod tests {
    use nb_macro::neuron_builder;
    use spiking_neural_networks::neuron::iterate_and_spike::IonotropicNeurotransmitterType;


    neuron_builder!(r#"
        [neuron]
            type: BasicIntegrateAndFire
            vars: e = 0, v_reset = -75, v_th = -55, dt = 100
            on_spike: 
                v = v_reset
            spike_detection: v >= v_th
            on_iteration:
                dv/dt = (v - e) + i
        [end]

        [ion_channel]
            type: TestLeak
            vars: e = 0, g = 1, current = 10
            on_iteration:
                current = g * (v - e)
        [end]

        [spike_train]
            type: RateSpikeTrain
            vars: step = 0, rate = 0, v_resting = 24
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
        [end]

        [neurotransmitter_kinetics]
            type: TestNeurotransmitterKinetics
            vars: t = 0.5, t_max = 1, c = 0.001, conc = 0
            on_iteration:
                [if] is_spiking [then]
                    conc = t_max
                [else]
                    conc = 0
                [end]

                t = t + dt * -c * t + conc

                t = min(max(t, 0), t_max)
        [end]

        [receptor_kinetics]
            type: TestReceptorKinetics
            vars: r = 0.5, r_max = 1
            on_iteration:
                r = min(max(t, 0), r_max)
        [end]
    "#);

    #[test]
    fn test_custom_dt() {
        let lif = BasicIntegrateAndFire::default_impl();

        assert_eq!(lif.dt, 100.);
    }

    #[test]
    fn test_custom_current() {
        let ion_channel = TestLeak::default();

        assert_eq!(ion_channel.current, 10.);
    }

    #[test]
    fn test_custom_v_resting() {
        let spike_train = RateSpikeTrain::default_impl();

        assert_eq!(spike_train.v_resting, 24.);
    }

    #[test]
    fn test_custom_neurotransmitter_kinetics() {
        let kinetics = TestNeurotransmitterKinetics::default();

        assert_eq!(kinetics.t, 0.5);
    }

    #[test]
    fn test_custom_receptor_kinetics() {
        let kinetics = TestReceptorKinetics::default();

        assert_eq!(kinetics.r, 0.5);
    }

    // test if gpu kernel and conversion works as expected
}
