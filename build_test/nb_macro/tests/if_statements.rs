// check if
// check if else
// check if else if 
// check if else if else if
// check if else if else if else
// check nested ifs

mod lif_reference;


#[cfg(test)]
mod test {
    use nb_macro::neuron_builder; 
    use crate::lif_reference::ReferenceIntegrateAndFire;


    neuron_builder!(r#"
    [neuron]
        type: BasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, flag = 0
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
            [if] i < 0 [then]
                flag = 1
            [end]
    [end]

    [neuron]
        type: NestedBasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, flag1 = 0, flag2 = 0
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
            [if] i < 0 [then]
                flag1 = 1
                [if] i > -30 [then]
                    flag2 = 2
                [end]
            [end]
    [end]

    [neuron]
        type: ElseBasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, flag = 0
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
            [if] i < 0 [then]
                flag = 1
            [else]
                flag = 2
            [end]
    [end]
    "#);

    #[test]
    pub fn test_if_statement() {
        let voltages = [-50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.];

        for i in voltages {
            let mut test_output: Vec<f32> = vec![];
            let mut reference_output: Vec<f32> = vec![];

            let mut to_test: BasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                BasicIntegrateAndFire::default();
            let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ReferenceIntegrateAndFire::default();

            assert_eq!(to_test.flag, 0.);

            for _ in 0..1000 {
                let _ = to_test.iterate_and_spike(i);
                test_output.push(to_test.current_voltage);
                let _ = reference_neuron.iterate_and_spike(i);
                reference_output.push(reference_neuron.current_voltage);

                if i < 0. {
                    assert_eq!(to_test.flag, 1.)
                }
            }

            assert_eq!(test_output, reference_output);
        }
    }

    #[test]
    pub fn test_nested_if_statement() {
        let voltages = [-50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.];

        for i in voltages {
            let mut test_output: Vec<f32> = vec![];
            let mut reference_output: Vec<f32> = vec![];

            let mut to_test: NestedBasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                NestedBasicIntegrateAndFire::default();
            let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ReferenceIntegrateAndFire::default();

            assert_eq!(to_test.flag1, 0.);
            assert_eq!(to_test.flag2, 0.);

            for _ in 0..1000 {
                let _ = to_test.iterate_and_spike(i);
                test_output.push(to_test.current_voltage);
                let _ = reference_neuron.iterate_and_spike(i);
                reference_output.push(reference_neuron.current_voltage);

                if i < 0. {
                    assert_eq!(to_test.flag1, 1.);
                    if i > -30. {
                        assert_eq!(to_test.flag2, 2.);
                    } else {
                        assert_eq!(to_test.flag2, 0.);
                    }
                } else {
                    assert_eq!(to_test.flag1, 0.);
                    assert_eq!(to_test.flag2, 0.);
                }
            }

            assert_eq!(test_output, reference_output);
        }
    }

    #[test]
    pub fn test_if_else_statement() {
        let voltages = [-50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.];

        for i in voltages {
            let mut test_output: Vec<f32> = vec![];
            let mut reference_output: Vec<f32> = vec![];

            let mut to_test: ElseBasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ElseBasicIntegrateAndFire::default();
            let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ReferenceIntegrateAndFire::default();

            assert_eq!(to_test.flag, 0.);

            for _ in 0..1000 {
                let _ = to_test.iterate_and_spike(i);
                test_output.push(to_test.current_voltage);
                let _ = reference_neuron.iterate_and_spike(i);
                reference_output.push(reference_neuron.current_voltage);

                if i < 0. {
                    assert_eq!(to_test.flag, 1.)
                } else {
                    assert_eq!(to_test.flag, 2.)
                }
            }

            assert_eq!(test_output, reference_output);
        }
    }

    // #[test]
    // pub fn test_if_if_else_statement() {

    // }
}