use nb_macro::neuron_builder;
// use pyo3::{types::PyTuple, exceptions::{PyKeyError, PyValueError}};
// mod lattices;
// use lattices::{
//     impl_lattice, impl_lattice_gpu, impl_spike_train_lattice,
//     impl_network, impl_network_gpu,
// };


neuron_builder!(r#"
    [neuron]
        type: BasicIntegrateAndFire
        kinetics: TestNeurotransmitterKinetics, TestReceptorKinetics
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

    [neuron]
        type: IonChannelNeuron
        kinetics: TestNeurotransmitterKinetics, TestReceptorKinetics
        ion_channels: l = TestLeak
        vars: v_reset = -75, v_th = -55
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            l.update_current(v)
            dv/dt = l.current + i
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

    [neural_refractoriness]
        type: TestRefractoriness
        vars: decay = 5000
        effect: (v_th - v_resting) * exp((-1 / (decay / dt)) * (time_difference ^ 2)) + v_resting
    [end]
"#);
