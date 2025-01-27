// test exp, min, max, sinh, cosh, tanh, sin, cos, tanh, heaviside
// test each function in a seperate neuron model

#[cfg(test)]
mod test {
    use nb_macro::neuron_builder;

    neuron_builder!(r#"
    [neuron]
        type: ExpTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = exp(i)
    [end]

    [neuron]
        type: MinTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = min(0, i)
    [end]

    [neuron]
        type: MaxTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = max(0, i)
    [end]

    [neuron]
        type: TanhTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = tanh(i)
    [end]

    [neuron]
        type: SinhTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = sinh(i)
    [end]

    [neuron]
        type: CoshTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = cosh(i)
    [end]

    [neuron]
        type: TanTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = tan(i)
    [end]

    [neuron]
        type: SinTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = sin(i)
    [end]

    [neuron]
        type: CosTest
        vars: v_reset = -75, v_th = 50000
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            v = cos(i)
    [end]
    "#);

    #[test]
    fn test_exp() {
        let mut exp_test: ExpTest<ApproximateNeurotransmitter, ApproximateReceptor> = ExpTest::default();

        for i in 0..10 {
            let input = i as f32;
            exp_test.iterate_and_spike(input);

            assert_eq!(exp_test.current_voltage, input.exp());
        }
    }

    #[test]
    fn test_min() {
        let mut min_test: MinTest<ApproximateNeurotransmitter, ApproximateReceptor> = MinTest::default();

        for i in -10..10 {
            let input = i as f32;
            min_test.iterate_and_spike(input);

            assert_eq!(min_test.current_voltage, min(0., input));
        }
    }

    #[test]
    fn test_max() {
        let mut max_test: MaxTest<ApproximateNeurotransmitter, ApproximateReceptor> = MaxTest::default();

        for i in -10..10 {
            let input = i as f32;
            max_test.iterate_and_spike(input);

            assert_eq!(max_test.current_voltage, max(0., input));
        }
    }

    #[test]
    fn test_tanh() {
        let mut tanh_test: TanhTest<ApproximateNeurotransmitter, ApproximateReceptor> = TanhTest::default();

        for i in -10..10 {
            let input = i as f32;
            tanh_test.iterate_and_spike(input);

            assert_eq!(tanh_test.current_voltage, tanh(input));
        }
    }

    #[test]
    fn test_sinh() {
        let mut sinh_test: SinhTest<ApproximateNeurotransmitter, ApproximateReceptor> = SinhTest::default();

        for i in -10..10 {
            let input = i as f32;
            sinh_test.iterate_and_spike(input);

            assert_eq!(sinh_test.current_voltage, sinh(input));
        }
    }

    #[test]
    fn test_cosh() {
        let mut cosh_test: CoshTest<ApproximateNeurotransmitter, ApproximateReceptor> = CoshTest::default();

        for i in -10..10 {
            let input = i as f32;
            cosh_test.iterate_and_spike(input);

            assert_eq!(cosh_test.current_voltage, cosh(input));
        }
    }

    #[test]
    fn test_tan() {
        let mut tan_test: TanTest<ApproximateNeurotransmitter, ApproximateReceptor> = TanTest::default();

        for i in -10..10 {
            let input = i as f32;
            tan_test.iterate_and_spike(input);

            assert_eq!(tan_test.current_voltage, tan(input));
        }
    }

    #[test]
    fn test_sin() {
        let mut sin_test: SinTest<ApproximateNeurotransmitter, ApproximateReceptor> = SinTest::default();

        for i in -10..10 {
            let input = i as f32;
            sin_test.iterate_and_spike(input);

            assert_eq!(sin_test.current_voltage, sin(input));
        }
    }

    #[test]
    fn test_cos() {
        let mut cos_test: CosTest<ApproximateNeurotransmitter, ApproximateReceptor> = CosTest::default();

        for i in -10..10 {
            let input = i as f32;
            cos_test.iterate_and_spike(input);

            assert_eq!(cos_test.current_voltage, cos(input));
        }
    }
}
