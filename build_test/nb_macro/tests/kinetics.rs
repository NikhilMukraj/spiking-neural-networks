#[cfg(test)]
pub mod test {
    use nb_macro::neuron_builder;

    neuron_builder!("
    [neurotransmitter_kinetics]
        type: BoundedNeurotransmitterKinetics
        vars: t_max = 1, c = 0.001, conc = 0
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
        type: BoundedReceptorKinetics
        vars: r_max = 1
        on_iteration:
            r = min(max(t, 0), r_max)
    [end]

    [neuron]
        type: BasicIntegrateAndFire
        kinetics: BoundedNeurotransmitterKinetics, BoundedReceptorKinetics
        vars: e = 0, v_reset = -75, v_th = -55
        on_spike: 
            v = v_reset
        spike_detection: v >= v_th
        on_iteration:
            dv/dt = -(v - e) + i
    [end]
    ");

    #[test]
    pub fn test_types() {
        assert_eq!(
            "kinetics::test::BasicIntegrateAndFire<kinetics::test::BoundedNeurotransmitterKinetics, kinetics::test::BoundedReceptorKinetics>", 
            std::any::type_name_of_val(&BasicIntegrateAndFire::default_impl()),
        );
    }
}