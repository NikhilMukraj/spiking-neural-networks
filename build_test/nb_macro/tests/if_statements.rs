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

    [neuron]
        type: ElseIfBasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, flag = 0
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
            [if] i < 0 [then]
                flag = 1
            [elseif] i > 30 [then]
                flag = 2
            [end]
    [end]

    [neuron]
        type: ElseIfElseBasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, flag = 0
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
            [if] i < 0 [then]
                flag = 1
            [elseif] i > 30 [then]
                flag = 2
            [else]
                flag = 3
            [end]
    [end]

    [neuron]
        type: ElseIfElseIfBasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, flag = 0
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
            [if] i < 0 [then]
                flag = 1
            [elseif] i > 20 [then]
                flag = 2
            [elseif] i > 0 [then]
                flag = 3
            [else]
                flag = 4
            [end]
    [end]

    [neuron]
        type: ElseIfNestedBasicIntegrateAndFire
        vars: e = 0, v_reset = -75, v_th = -55, flag = 0
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = (v - e) + i
            [if] i < 0 [then]
                flag = 1
            [elseif] i > 20 [then]
                [if] i >= 40 [then]
                    flag = 2
                [else]
                    flag = 3
                [end]
            [else]
                flag = 4
            [end]
    [end]
    "#);

    const VOLTAGES: [f32; 11] = [-50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50.];

    #[test]
    pub fn test_if_statement() {
        let mut hit = false;
        for i in VOLTAGES {
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
                    hit = true;
                    assert_eq!(to_test.flag, 1.);
                }
            }

            assert_eq!(test_output, reference_output);
        }

        assert!(hit);
    }

    #[test]
    pub fn test_nested_if_statement() {
        let (mut hit1, mut hit2, mut hit3, mut hit4) = (false, false, false, false);

        for i in VOLTAGES {
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
                    hit1 = true;
                    assert_eq!(to_test.flag1, 1.);
                    if i > -30. {
                        hit2 = true;
                        assert_eq!(to_test.flag2, 2.);
                    } else {
                        hit3 = true;
                        assert_eq!(to_test.flag2, 0.);
                    }
                } else {
                    hit4 = true;
                    assert_eq!(to_test.flag1, 0.);
                    assert_eq!(to_test.flag2, 0.);
                }
            }

            assert_eq!(test_output, reference_output);
        }

        assert!(hit1 && hit2 && hit3 && hit4);
    }

    #[test]
    pub fn test_if_else_statement() {
        let (mut hit1, mut hit2) = (false, false);

        for i in VOLTAGES {
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
                    hit1 = true;
                    assert_eq!(to_test.flag, 1.)
                } else {
                    hit2 = true;
                    assert_eq!(to_test.flag, 2.)
                }
            }

            assert_eq!(test_output, reference_output);
        }

        assert!(hit1 && hit2);
    }

    #[test]
    pub fn test_if_else_if_statement() {
        let (mut hit1, mut hit2) = (false, false);

        for i in VOLTAGES {
            let mut test_output: Vec<f32> = vec![];
            let mut reference_output: Vec<f32> = vec![];

            let mut to_test: ElseIfBasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ElseIfBasicIntegrateAndFire::default();
            let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ReferenceIntegrateAndFire::default();

            assert_eq!(to_test.flag, 0.);

            for _ in 0..1000 {
                let _ = to_test.iterate_and_spike(i);
                test_output.push(to_test.current_voltage);
                let _ = reference_neuron.iterate_and_spike(i);
                reference_output.push(reference_neuron.current_voltage);

                if i < 0. {
                    hit1 = true;
                    assert_eq!(to_test.flag, 1.)
                } else if i > 30. {
                    hit2 = true;
                    assert_eq!(to_test.flag, 2.)
                }
            }

            assert_eq!(test_output, reference_output);
        }

        assert!(hit1 && hit2);
    }

    #[test]
    pub fn test_if_else_if_else_statement() {
        let (mut hit1, mut hit2, mut hit3) = (false, false, false);

        for i in VOLTAGES {
            let mut test_output: Vec<f32> = vec![];
            let mut reference_output: Vec<f32> = vec![];

            let mut to_test: ElseIfElseBasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ElseIfElseBasicIntegrateAndFire::default();
            let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ReferenceIntegrateAndFire::default();

            assert_eq!(to_test.flag, 0.);

            for _ in 0..1000 {
                let _ = to_test.iterate_and_spike(i);
                test_output.push(to_test.current_voltage);
                let _ = reference_neuron.iterate_and_spike(i);
                reference_output.push(reference_neuron.current_voltage);

                if i < 0. {
                    hit1 = true;
                    assert_eq!(to_test.flag, 1.)
                } else if i > 30. {
                    hit2 = true;
                    assert_eq!(to_test.flag, 2.)
                } else {
                    hit3 = true;
                    assert_eq!(to_test.flag, 3.)
                }
            }

            assert_eq!(test_output, reference_output);
        }

        assert!(hit1 && hit2 && hit3);
    }

    #[test]
    pub fn test_if_else_if_else_if_else() {
        let (mut hit1, mut hit2, mut hit3, mut hit4) = (false, false, false, false);

        for i in VOLTAGES {
            let mut test_output: Vec<f32> = vec![];
            let mut reference_output: Vec<f32> = vec![];

            let mut to_test: ElseIfElseIfBasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
            ElseIfElseIfBasicIntegrateAndFire::default();
            let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ReferenceIntegrateAndFire::default();

            assert_eq!(to_test.flag, 0.);

            for _ in 0..1000 {
                let _ = to_test.iterate_and_spike(i);
                test_output.push(to_test.current_voltage);
                let _ = reference_neuron.iterate_and_spike(i);
                reference_output.push(reference_neuron.current_voltage);

                if i < 0. {
                    hit1 = true;
                    assert_eq!(to_test.flag, 1.)
                } else if i > 20. {
                    hit2 = true;
                    assert_eq!(to_test.flag, 2.);
                } else if i > 0. {
                    hit3 = true;
                    assert_eq!(to_test.flag, 3.)
                } else {
                    hit4 = true;
                    assert_eq!(to_test.flag, 4.)
                }
            }

            assert_eq!(test_output, reference_output);
        }

        assert!(hit1 && hit2 && hit3 && hit4);
    }

    #[test]
    pub fn test_if_else_if_nested() {
        let (mut hit1, mut hit2, mut hit3, mut hit4) = (false, false, false, false);

        for i in VOLTAGES {
            let mut test_output: Vec<f32> = vec![];
            let mut reference_output: Vec<f32> = vec![];

            let mut to_test: ElseIfNestedBasicIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ElseIfNestedBasicIntegrateAndFire::default();
            let mut reference_neuron: ReferenceIntegrateAndFire<ApproximateNeurotransmitter, ApproximateReceptor> = 
                ReferenceIntegrateAndFire::default();

            assert_eq!(to_test.flag, 0.);

            for _ in 0..1000 {
                let _ = to_test.iterate_and_spike(i);
                test_output.push(to_test.current_voltage);
                let _ = reference_neuron.iterate_and_spike(i);
                reference_output.push(reference_neuron.current_voltage);

                if i < 0. {
                    hit1 = true;
                    assert_eq!(to_test.flag, 1.)
                } else if i > 20. {
                    if i >= 40. {
                        hit2 = true;
                        assert_eq!(to_test.flag, 2.)
                    } else {
                        hit3 = true;
                        assert_eq!(to_test.flag, 3.)
                    }
                } else {
                    hit4 = true;
                    assert_eq!(to_test.flag, 4.)
                }
            }

            assert_eq!(test_output, reference_output);
        }

        assert!(hit1 && hit2 && hit3 && hit4);
    }
}